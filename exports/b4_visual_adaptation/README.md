# B4 Visual Adaptation

Qwen3-VL-8B 的 B4 主实现：共享四头视觉读取与条件编码，在四个解码器层的原生 FFN 输出上添加条件增量。仅用于 prefill，后续 decode 保持原生计算。

本目录包含独立的 B4 实现、训练入口和适配器权重。Qwen3-VL-8B 骨干与训练数据需自行准备。

## 文件

| 文件 | 内容 |
|---|---|
| `b4/model.py` | 状态 Query、视觉 Key/Value、多头读取、共享条件编码、四层输出适配器 |
| `b4/memory.py` | 原生视觉 memory、因果可见性、文本余弦先验、位置编码 |
| `b4/inference.py` | Qwen3-VL prefill 接入与 FFN 输出注入 |
| `train.py` | 单卡/torchrun 多卡训练、梯度累积、保存与恢复 |
| `b4/training.py` | 全词表 CE + hinge 损失、余弦日程、选择性激活检查点与视觉特征复用 |
| `b4/data.py` | MCQ 输入、整视频 CPU 预处理和有界预加载 |
| `prepare_data.py` | 将训练清单及训练标签转换为 JSONL |
| `watch.py` | 读取训练状态，显示实时进度 |
| `b4_weights.pt` | B4 的 FP32 适配器权重，纯 state_dict |
| `WEIGHTS.json` | 权重格式、结构、精度与文件大小 |

## 架构

| 设置 | 数值 |
|---|---|
| 骨干 | Qwen3-VL-8B |
| 注入层，0 起始 | 8 / 17 / 26 / 35 |
| 隐藏维度 / FFN 中间维度 | 4096 / 12288 |
| 读取头数 / 每头 Key 维度 | 4 / 128 |
| 共享条件维度 | 128 |
| 每层输出适配器宽度 | 745 |
| Query 分块大小 | 16 |
| 文本先验温度 | 0.1 |
| 新增参数量 | 19,405,440 |
| 注入位置 | 题干、选项文本 token，以及首答案预测位置；仅 prefill |
| 精度 | 原生骨干 BF16，读取、条件编码、输出适配器 FP32；增量相加前转回原生输出 dtype |

令 `x` 为当前层进入 FFN 的实际隐藏状态，`M` 为原生多模态 prefill 输入中的视觉 embedding：

```text
Q_l = RMS(reshape(RMS(x) @ W_Q + E_layer[l]))
K   = RMS(reshape(RMS(M) @ W_K) + positional_encoding)
V   = M

prior[t, i] = max cosine(M[i], text[j]) over public text positions j <= t
weights    = softmax(Q_l @ K^T / sqrt(128) + prior / 0.1 + causal_mask)
R_l        = weights @ V
c_l        = SiLU(concat(RMS(x), headwise_RMS(R_l)) @ W_C + b_C)
delta_l    = SiLU(c_l @ U_l) @ V_l
output_l   = native_FFN(x) + delta_l
```

`causal_mask` 只允许读取位置严格早于 Query 的视觉 token。固定位置编码使用原生 mRoPE 的时间、高、宽三个轴，每轴 42 维 sin/cos，另加 image/video 两维标识。没有额外学习的视觉 Value 投影。

这里的 `U_l`、`V_l` 是输出适配器矩阵；公式中的视觉 Value `V=M` 是另一项。原生 FFN 的 gate/up/down 计算保持完整，增量在 down projection 之后相加；外层残差仍由骨干处理。

文本余弦先验来自当前 token 可见的公开文本前缀；Query 来自当前隐藏状态。每题重建 memory 和当前 Key，不复用其他题的语言轨迹、条件向量或 KV。

## 接入

运行环境：Python 3.10、PyTorch 2.8.0、Transformers 4.57.6。进入本目录后安装依赖：

```sh
pip install -r requirements.txt
```

下面展示模型接入方式。`inputs` 是 Qwen3-VL 原生 processor 已准备好并放到模型设备上的输入，包含 `input_ids`、`attention_mask`、视觉 pixels 与对应 grid；视频解码和采样由调用端处理。

`question_option_positions` 是**视觉 token 展开后的最终 input_ids** 中题干、选项文本 token 的位置列表。它不包括系统文本、视觉占位 token、模板特殊 token 或答案；如有题内图片，位置列表跳过图片段。代码另外加入 `len(input_ids[0])-1`，即生成首答案所用的 prompt 最后位置。此位置表由调用端根据 prompt 分词结果提供，本包不包含数据集专用题目解析和评分逻辑。

```python
import torch
from transformers import Qwen3VLForConditionalGeneration
from b4 import B4Adapter, B4Inference

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
).to("cuda").eval().requires_grad_(False)

adapter = B4Adapter()
# b4_weights.pt contains the adapter state_dict, without the backbone/optimizer.
adapter.load_state_dict(torch.load("b4_weights.pt", map_location="cpu", weights_only=True))
adapter = adapter.to("cuda").eval().requires_grad_(False)  # Keep adapter FP32.

# inputs and question_option_positions come from the caller's prompt preparation.
with torch.inference_mode(), B4Inference(
    model, adapter, inputs, question_option_positions
):
    output_ids = model.generate(
        **inputs, use_cache=True, do_sample=False, num_beams=1,
        max_new_tokens=128,
    )
```

每个上下文仅处理一个完整、无 padding 的 prompt（batch=1），使用新的生成 cache。`max_new_tokens` 由调用端根据任务设置；示例不定义数据集的评估口径。解码器层与适配器应在同一设备上。本接入不处理 beam expansion、分段 prefill 或外部历史 KV；这些输入条件由调用端负责。

参数名：`W_Q`、`W_K`、`E_layer`、`W_C`、`b_C` 和 `output_adapters.{0..3}.{U,V}`。完整训练 checkpoint 的适配器 state_dict 位于 `checkpoint['model']`；将这个字典作为权重载入即可。新建适配器的输出矩阵 `V` 初始为零，必须加载已训练的 B4 权重才能获得其适配效果。

## 权重

`b4_weights.pt` 为 B4 适配器权重，格式是可由 `torch.load(..., weights_only=True)` 加载的纯 `state_dict`。文件包含 13 个 FP32 参数张量，共 19,405,440 个参数，约 **74.03 MiB**；不包含骨干、优化器、RNG 或训练数据。结构和格式信息见 `WEIGHTS.json`。

`.gitignore` 已为根目录的 `b4_weights.pt` 单独放行，其他训练 checkpoint 和输出目录仍忽略。

## 训练

训练入口默认从新初始化的 B4 开始。`--resume` 用于恢复本训练目标保存的完整 checkpoint；纯适配器权重用于推理加载。

| 项目 | 默认值 |
|---|---|
| 训练集 | VideoKR 训练集，保留输入清单顺序；步数按实际题数计算 |
| 帧数 | 最多 128 帧；提供帧索引时以清单为准 |
| 训练参数 | 仅 B4 新增参数；原生视觉编码器和语言骨干冻结 |
| Epoch | 1 |
| 全局 batch | **8**，每卡 microbatch=1，多卡梯度求和后除以实际题数 |
| 更新次数 | 21,751 题时为 **2,719**，最后一批 **7** 题，不补题也不丢弃 |
| 学习率 | 峰值 **1e-4**，余弦衰减到 0 |
| Warmup | 总步数的 **3%** 向上取整；2,719 步时为 **82** 步 |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，clip=1 |
| 损失 | 全程使用原始全词表 CE + 0.1×hinge，margin=0.5 |
| 精度 | BF16 骨干、FP32 B4 和梯度；LM head 使用原生 BF16 前向与分块 FP32 反向 |
| 保存 | 每 100 步和最终步，保存适配器、优化器、日程位置、数据进度、各卡 RNG |

令 `z` 为首答案预测位置的原始全词表 logits，`y` 为正确选项字母的 token ID：

```text
CE     = logsumexp(z) - z[y]
margin = z[y] - max(z[j] for j != y)
loss   = CE + 0.1 * max(0, 0.5 - margin)
```

所有更新使用同一个损失，直接在未归一化的全词表 logits 上计算；hinge 的竞争项包含除正确答案 token 外的全部词表。模型内部 Query、Key 和条件编码的 RMS 属于架构计算。答案只参与首答案位置的损失，不拼进输入 prompt。

### 数据

输入 JSONL 每行一个 VideoKR 选择题，字段如下；`options` 可以带 `A.` 等前缀。当前训练读取器针对这批视频加文本问答，不处理题内独立图片。

```json
{"id":"sample_001","video":"/path/to/video.mp4","question":"What happens next?","options":["A. First choice","B. Second choice","C. Third choice","D. Fourth choice"],"answer":"B","frame_indices":[0,4,8,12]}
```

例子的 4 帧仅展示字段格式；训练时可传入 **128 帧版本**的 `FRAME_MANIFEST.json`。提供 `frame_indices` 时直接使用这些索引，`--max-frames` 不覆盖已有清单。只有未给出索引的新数据，才按 2 FPS、最多 128 帧均匀采样。

将训练清单转换为本包的格式：

```sh
python prepare_data.py \
  --manifest /path/to/allqa128_run/manifests/train.json \
  --labels /path/to/allqa128_run/private/train \
  --frames /path/to/allqa128_run/FRAME_MANIFEST.json \
  --output data/train_21k.jsonl
```

标签文件以 question ID 的 SHA256 命名，转换脚本用它定位文件。视频搬到其他机器后可增加 `--video-root /path/to/videos`，按 `meta.file_name` 重写路径。转换器只读取提供的训练清单；请仅传入训练集，验证集和测试集不参与梯度训练。训练数据和运行输出已加入 `.gitignore`，根目录的 `b4_weights.pt` 单独放行。

### 启动与恢复

默认语言注意力为 SDPA、视觉注意力为 FlashAttention2；FlashAttention2 需在目标 CUDA 环境单独安装。若使用 `--vision-attention sdpa`，则明确切换视觉注意力实现。

双卡启动示例（显卡编号由调用者选择）：

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --data data/train_21k.jsonl \
  --output outputs/b4_train
```

单卡将上述命令换为 `CUDA_VISIBLE_DEVICES=0 python train.py ...`。全局 batch 始终为 8，不会乘以卡数。每个优化批次内按视频分组分配给 GPU，同视频题逐题前向、反向；所有卡完成后仅同步 B4 梯度，再统一裁剪与更新。

恢复时使用相同卡数和同一训练数据清单，添加 `--resume outputs/b4_train/checkpoint_0100.pt`。恢复已保存的模型、AdamW、步数、各卡 RNG、视觉注意力实现与训练配置，从下一批继续。入口检查保存的损失标识与卡数；不兼容或缺少损失标识的 checkpoint 会在初始化 CUDA 前报错。输出目录中的 `b4_weights.pt` 是最新保存节点的纯适配器权重，可用于上面的推理示例。

```sh
python watch.py outputs/b4_train
```

状态包括加载、等待输入、前向/反向、优化器更新、保存、完成，以及 PID、每卡进度、loss、原始 CE、LR、batch 耗时、CPU 等待和 ETA。心跳超过预期未更新时应检查终端错误，不能仅凭状态文件的 RUNNING 判断存活；具体异常由 Python 直接输出，不自动重试或跳过样本。

### 实际接入的加速

- 每卡默认一个 spawn CPU 预处理进程、最多两个在途视频任务，跨视频预加载。
- 一个视频组只解码和进行 CPU 视觉预处理一次；同组题共享 pixels/grid。
- 同视频组复用冻结视觉编码器的 merger 和 DeepStack 输出；各题仍重建自身语言状态、R/C 和反向图。
- 每卡一次计算一题；通过梯度累积实现 batch8，不将长视频问题 padding 到同一个大 microbatch。
- 非 reentrant 选择性激活检查点保留融合注意力输出，重算其他解码器运算；冻结骨干仍保留传向前层 B4 的梯度路径。
- 仅对 prompt 最后位置计算全词表 logits；每次更新只同步一份扁平 B4 梯度。

CPU 缓存和视觉特征缓存均限当前视频组，不建立全数据集磁盘缓存。训练采用每批次内的视频分组与各卡有界预加载队列，所有卡在梯度更新前同步。本包不附带数据集评估入口，训练期间也不自动评价。

## CPU 检查

```sh
python -m unittest discover -s tests -v
```

检查损失数值与梯度、等值 logits 的有限性、缩放敏感性及断点恢复的损失配置兼容性。完整 GPU 训练和 benchmark 需在目标环境中另行运行。

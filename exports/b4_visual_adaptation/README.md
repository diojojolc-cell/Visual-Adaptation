# B4 Visual Adaptation

Qwen3-VL-8B 的 B4 主实现：共享四头视觉读取与条件编码，在四个解码器层的原生 FFN 输出上添加条件增量。仅用于 prefill，后续 decode 保持原生计算。

本目录包含独立的 B4 主实现、训练入口，以及已完成三数据集全量测试的 B4 u200 适配器权重。不依赖原实验仓库，没有附带评测、对比、验收、审计、资源门禁或服务器任务切换代码；Qwen3-VL-8B 原生权重与数据需自行准备。

## 文件

| 文件 | 内容 |
|---|---|
| `b4/model.py` | 状态 Query、视觉 Key/Value、多头读取、共享条件编码、四层输出适配器 |
| `b4/memory.py` | 原生视觉 memory、因果可见性、文本余弦先验、位置编码 |
| `b4/inference.py` | Qwen3-VL prefill 接入与 FFN 输出注入 |
| `train.py` | 单卡/torchrun 多卡训练、梯度累积、保存与恢复 |
| `b4/training.py` | 原损失、余弦日程、选择性激活检查点与视觉特征复用 |
| `b4/data.py` | MCQ 输入、整视频 CPU 预处理和有界预加载 |
| `prepare_data.py` | 将原全题训练清单及训练标签转换为 JSONL |
| `watch.py` | 读取训练状态，显示实时进度 |
| `b4_weights.pt` | 原 B4 u200 的 FP32 适配器权重，纯 state_dict |
| `WEIGHTS.json` | 权重来源、训练节点、结构与文件大小 |

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

运行环境参考原实现：Python 3.10、PyTorch 2.8.0、Transformers 4.57.6。进入本目录后安装依赖：

```sh
pip install -r requirements.txt
```

下面展示模型接入方式。`inputs` 是 Qwen3-VL 原生 processor 已准备好并放到模型设备上的输入，包含 `input_ids`、`attention_mask`、视觉 pixels 与对应 grid；视频解码和采样由调用端处理。

`question_option_positions` 是**视觉 token 展开后的最终 input_ids** 中题干、选项文本 token 的位置列表。它不包括系统文本、视觉占位 token、模板特殊 token 或答案；如有题内图片，位置列表跳过图片段。代码另外加入 `len(input_ids[0])-1`，即生成首答案所用的 prompt 最后位置。此位置表由调用端沿用原 prompt 分词结果提供，本包不包含数据集专用题目解析和评分逻辑。

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

参数名沿用 B4：`W_Q`、`W_K`、`E_layer`、`W_C`、`b_C` 和 `output_adapters.{0..3}.{U,V}`。已有研究 checkpoint 的适配器 state_dict 位于 `checkpoint['model']`；将这个字典作为权重载入即可。新建适配器的输出矩阵 `V` 初始为零，必须加载已训练的 B4 权重才能获得其适配效果。

## 随包权重

`b4_weights.pt` 提取自原 **8K 训练的 B4 u200**，是 `visual_ffn_qwen3vl8b_ablation_b_u200_full_v1` 完成 3,625 条正式生成时使用的权重。它不是下述新 21K 训练配置的训练产物。只保存 13 个 FP32 参数张量，共 19,405,440 个参数；不包含 8B 骨干、AdamW、RNG 或训练数据，也没有降精度。

原全量评估的三域宏平均为 53.76%；这属于原实验入口的已保存结果，不表示精简包已重新跑过该评估。权重可直接按上面的 `torch.load(..., weights_only=True)` 示例加载。

该文件约 **74.03 MiB**，可通过普通 **Git push** 上传；GitHub 对超过 50 MiB 的文件会提示，超过 100 MiB 才阻止普通 Git 上传。网页向仓库上传文件的限制是 25 MiB，因此不能直接在网页上传这个权重。此大小不强制使用 Git LFS。[GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

`.gitignore` 已为根目录的 `b4_weights.pt` 单独放行，其他训练 checkpoint 和输出目录仍忽略。

## 21K 全题训练

最终默认配置如下。它是在 B4 架构上使用全题训练配置，不是原 8K/u200 训练的位级重放。训练入口默认从新初始化的 B4 开始，不自动加载随包 u200 权重；`--resume` 用于完整训练 checkpoint 的断点恢复。

| 项目 | 默认值 |
|---|---|
| 训练集 | 原 8,000 个训练视频的全部 21,751 个问答，保留清单顺序 |
| 帧数 | 原 128 帧清单；短视频沿用原实际帧数 |
| 训练参数 | 仅 B4 新增参数；原生视觉编码器和语言骨干冻结 |
| Epoch | 1 |
| 全局 batch | **8**，每卡 microbatch=1，多卡梯度求和后除以实际题数 |
| 更新次数 | **2,719**，最后一批 **7** 题，不补题也不丢弃 |
| 学习率 | 峰值 **1e-4**，余弦衰减到 0 |
| Warmup | 总步数的 **3%** 向上取整，即 **82** 步 |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，clip=1 |
| 损失 | 前 **50** 次更新为原始全词表 CE + 0.1×hinge，之后使用原去尺度复合损失 |
| 精度 | BF16 骨干、FP32 B4 和梯度；LM head 使用原生 BF16 前向与分块 FP32 反向 |
| 保存 | 每 100 步和最终步，保存适配器、优化器、日程位置、数据进度、各卡 RNG |

`raw_steps=50` 沿用此前确认的全题设置，按 optimizer 更新计数；改变 batch 不会自动改变这个值。去尺度阶段只对**合法选项 logits**中心化并除以可微 RMS，损失为选项 CE + 0.1×归一化 margin hinge + 合法选项概率质量损失。它不是纯 CE SFT。答案只参与首答案位置的损失，不拼进输入 prompt。

### 数据

输入 JSONL 每行一个 VideoKR 选择题，字段如下；`options` 可以带 `A.` 等前缀。当前训练读取器针对这批视频加文本问答，不处理题内独立图片。

```json
{"id":"sample_001","video":"/path/to/video.mp4","question":"What happens next?","options":["A. First choice","B. Second choice","C. Third choice","D. Fourth choice"],"answer":"B","frame_indices":[0,4,8,12]}
```

例子的 4 帧仅展示字段格式；正式全题训练请传入原 **128 帧版本**的 `FRAME_MANIFEST.json`。提供 `frame_indices` 时直接使用这些索引，`--max-frames` 不覆盖已有清单。只有未给出索引的新数据，才按 2 FPS、最多 128 帧均匀采样。

将原训练清单转换为本包的格式：

```sh
python prepare_data.py \
  --manifest /path/to/allqa128_run/manifests/train.json \
  --labels /path/to/allqa128_run/private/train \
  --frames /path/to/allqa128_run/FRAME_MANIFEST.json \
  --output data/train_21k.jsonl
```

原标签文件以 question ID 的 SHA256 命名，转换脚本仅用它定位文件，不执行内容哈希、身份对比或验收。视频搬到其他机器后可增加 `--video-root /path/to/videos`，按原 `meta.file_name` 重写路径。转换器只读取提供的训练清单，其他集合不会加入。训练数据和运行输出已加入 `.gitignore`；仅上述随包 B4 u200 权重允许上传。

### 启动与恢复

默认语言注意力为 SDPA、视觉注意力为 FlashAttention2，沿用原 Qwen 配置；FlashAttention2 需在目标 CUDA 环境单独安装。若使用 `--vision-attention sdpa`，则明确切换视觉注意力实现。

双卡启动示例（显卡编号由调用者选择）：

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --data data/train_21k.jsonl \
  --output outputs/b4_allqa128_b8_cos1em4
```

单卡将上述命令换为 `CUDA_VISIBLE_DEVICES=0 python train.py ...`。全局 batch 始终为 8，不会乘以卡数。每个优化批次内按视频分组分配给 GPU，同视频题逐题前向、反向；所有卡完成后仅同步 B4 梯度，再统一裁剪与更新。

恢复时使用相同卡数，添加 `--resume outputs/b4_allqa128_b8_cos1em4/checkpoint_0100.pt`。恢复已保存的模型、AdamW、步数、RNG 与科学配置，从下一批继续，不重复已保存批次。`b4_weights.pt` 是最新保存节点的纯适配器权重，可直接用于上面的推理示例。

```sh
python watch.py outputs/b4_allqa128_b8_cos1em4
```

状态包括加载、等待输入、前向/反向、优化器更新、保存、完成，以及 PID、每卡进度、loss、原始 CE、LR、batch 耗时、CPU 等待和 ETA。心跳超过预期未更新时应检查终端错误，不能仅凭状态文件的 RUNNING 判断存活；具体异常由 Python 直接输出，不自动重试或跳过样本。

### 实际接入的加速

- 每卡默认一个 spawn CPU 预处理进程、最多两个在途视频任务，跨视频预加载。
- 一个视频组只解码和进行 CPU 视觉预处理一次；同组题共享 pixels/grid。
- 同视频组复用冻结视觉编码器的 merger 和 DeepStack 输出；各题仍重建自身语言状态、R/C 和反向图。
- 每卡一次计算一题；通过梯度累积实现 batch8，不将长视频问题 padding 到同一个大 microbatch。
- 非 reentrant 选择性激活检查点保留融合注意力输出，重算其他解码器运算；冻结骨干仍保留传向前层 B4 的梯度路径。
- 仅对 prompt 最后位置计算全词表 logits；每次更新只同步一份扁平 B4 梯度。

CPU 缓存和视觉特征缓存均限当前视频组，不建立全数据集磁盘缓存；没有复制研究工程的 GPU 占位、自动切换和 CPU 激活 offload 控制器。本包不附带 Dev/全集评价，训练期间也不自动运行这些评价。

## 实现来源

此版本提取原 B4 的 `learned_output` 分支，移除其他架构选项和实验管理依赖：

- `visual_ffn_core_ablations_v1/model.py`：B 的状态读取、条件编码与输出适配器。
- `visual_ffn_core_ablations_v1/hooks.py`：原生 FFN 输出之后的 prefill 注入。
- `visual_ffn_v1.py`、`visual_ffn_v1_speed_v1.py`、`visual_ffn_v1_steady_v5.py`：RMS、视觉 memory、余弦先验、位置编码、FP32 分块读取与缓存。
- `visual_ffn_ablation_b_full_v1/runtime.py`：从实际 Qwen3-VL prefill embedding 建立 memory 的接入位置。

保留一次 prompt 内的 FP32 视觉缓冲、RMS 缓冲、因果掩码和 Key 复用，以及 16 Query 分块读取。训练部分参考 `visual_ffn_qwen8b_allqa128_cost_r5`、`visual_ffn_qwen8b_allqa32_v1`、`visual_ffn_core_ablations_8k_4gpu_v2` 的实际路径，以及原 `visual_ffn_scalefree_v1` 损失。没有附带服务器运行记录。该精简包尚未执行完整 GPU 训练或重跑正式 benchmark，不将代码提取宣称为新的精度或速度结果。

"""Train only B4 on the ordered 21K multiple-choice question manifest."""

import argparse
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist

from b4 import B4Adapter, B4Inference
from b4.data import Prefetch, read_rows, video_jobs
from b4.training import FrozenHead, FrozenVideo, answer_loss, learning_rate, mean_gradients, selective_context


class Progress:
    def __init__(self, output, rank, device):
        self.path = output / f'status_rank{rank}.json'
        self.info = dict(pid=os.getpid(), rank=rank, device=str(device), stage='loading',
                         update=0, time=time.time(), state='RUNNING')
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.info.update(values, time=time.time())
            self.write()

    def write(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(self.info, heartbeat=time.time()), indent=2) + '\n')
        tmp.replace(self.path)

    def heartbeat(self):
        while not self.stop.wait(5):
            with self.lock:
                self.write()

    def close(self):
        self.stop.set()
        self.thread.join()


def save(output, adapter, optimizer, config, update, scheduler, rank, world):
    rng = dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
    rngs = [None] * world
    if world > 1:
        dist.all_gather_object(rngs, rng)
    else:
        rngs[0] = rng
    if rank == 0:
        weights = {name: value.detach().cpu() for name, value in adapter.state_dict().items()}
        per_epoch = math.ceil(config['question_count'] / config['batch_size'])
        completed = ((update // per_epoch) * config['question_count']
                     + min((update % per_epoch) * config['batch_size'], config['question_count']))
        state = dict(model=weights, optimizer=optimizer.state_dict(), config=config,
                     update=update, scheduler=scheduler, rng=rngs,
                     completed_questions=completed)
        path = output / f'checkpoint_{update:04d}.pt'
        tmp = path.with_suffix('.tmp')
        torch.save(state, tmp)
        tmp.replace(path)
        torch.save(weights, output / 'b4_weights.tmp')
        (output / 'b4_weights.tmp').replace(output / 'b4_weights.pt')
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--data', required=True, help='Ordered training-only JSONL, with answers and frame indices')
    parser.add_argument('--output', default='outputs/b4_allqa128_b8_cos1em4')
    parser.add_argument('--batch-size', type=int, default=8, help='Global question count per optimizer update')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-ratio', type=float, default=0.03)
    parser.add_argument('--raw-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-workers', type=int, default=1, help='Spawned CPU preprocessors per GPU')
    parser.add_argument('--prefetch', type=int, default=2, help='Maximum outstanding video jobs per GPU')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--vision-attention', default='flash_attention_2', choices=['flash_attention_2', 'sdpa'])
    parser.add_argument('--resume', help='Full training checkpoint; resume with the same GPU count')
    args = parser.parse_args()
    rank, local, world = (int(os.environ.get(k, '0' if k != 'WORLD_SIZE' else '1'))
                          for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status = Progress(output, rank, device)
    resume = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    config = dict(vars(args))
    if resume:
        # Continue the saved scientific configuration; paths may move between machines.
        for name in ('batch_size', 'epochs', 'max_frames', 'lr', 'warmup_ratio', 'raw_steps', 'seed'):
            config[name] = resume['config'][name]
    rows = read_rows(args.data)
    batch = config['batch_size']
    per_epoch = math.ceil(len(rows) / batch)
    total = per_epoch * config['epochs']
    warmup = math.ceil(total * config['warmup_ratio'])
    start = resume['update'] if resume else 0
    config.update(question_count=len(rows), total_steps=total, warmup_steps=warmup, world_size=world,
                  microbatch_per_gpu=1, training_scope='B4_only_prefill', loss='raw_then_centered_RMS')
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        print(f'{len(rows)} QA | batch={batch} | {total} updates | warmup={warmup} | frames<={config["max_frames"]}', flush=True)
    # Video jobs stay together on one GPU. Only gradients cross GPU boundaries.
    schedule = []
    for u in range(start + 1, total + 1):
        offset = ((u - 1) % per_epoch) * batch
        current = rows[offset:offset + batch]
        schedule.append((u, len(current), video_jobs(current)[rank::world]))
    loader = Prefetch((job for _, _, jobs in schedule for job in jobs), args.model,
                      config['max_frames'], args.cpu_workers, args.prefetch)
    from transformers import Qwen3VLForConditionalGeneration
    adapter = B4Adapter().to(device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation={'text_config': 'sdpa', 'vision_config': args.vision_attention},
    ).to(device).eval().requires_grad_(False)
    lm = model.model.language_model
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        'use_reentrant': False, 'context_fn': selective_context})
    lm.train()  # Activate layer checkpointing; Qwen's attention dropout is zero.
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['lr'], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0)
    if resume:
        adapter.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        rng = resume['rng'][rank]
        random.setstate(rng['python'])
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'])
        del resume
    status.update(stage='training', update=start, total=total, samples=0)
    for update, count, jobs in schedule:
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros(2, device=device)
        cpu_seconds = wait_seconds = 0.0
        completed = 0
        for _ in jobs:
            status.update(stage='waiting_for_input', update=update - 1, current_update=update)
            prepared = loader.take()
            cpu_seconds += prepared['cpu_seconds']
            wait_seconds += prepared['wait_seconds']
            questions = prepared['questions']
            media = {k: questions[0]['inputs'][k].to(device)
                     for k in ('pixel_values_videos', 'video_grid_thw')}
            with FrozenVideo(model):
                for question in questions:
                    status.update(stage='forward_backward', question=question['id'], samples=completed)
                    inputs = {k: (media[k] if k in media else v.to(device))
                              for k, v in question['inputs'].items()}
                    # Keep the context alive through backward/checkpoint recomputation.
                    with B4Inference(model, adapter, inputs, question['positions']):
                        hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
                        logits = FrozenHead.apply(hidden[0, -1].float(), model.lm_head.weight)
                        loss, raw_ce = answer_loss(logits, question['gold'], question['option_ids'],
                                                   update, config['raw_steps'])
                        loss.backward()
                        loss_sum += torch.stack((loss.detach(), raw_ce.detach()))
                    completed += 1
                    del hidden, logits, loss, raw_ce, inputs
            del media, prepared, questions
        status.update(stage='optimizer', samples=completed)
        mean_gradients(adapter, count, world > 1)
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        lr = learning_rate(update, total, warmup, config['lr'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if world > 1:
            dist.all_reduce(loss_sum)
        values = (loss_sum / count).tolist()
        torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        remaining = (total - update) * seconds
        status.update(stage='training', update=update, total=total, lr=lr, loss=values[0], raw_ce=values[1],
                      batch_seconds=seconds, cpu_seconds=cpu_seconds, cpu_wait_seconds=wait_seconds,
                      eta_seconds=remaining, gradient_norm=float(norm), global_batch=count)
        if rank == 0:
            print(f'u{update}/{total} | loss {values[0]:.4f} | CE {values[1]:.4f} | LR {lr:.3e} | '
                  f'{seconds:.1f}s/batch | CPU wait {wait_seconds:.1f}s | ETA {remaining/3600:.2f}h', flush=True)
            with (output / 'training.jsonl').open('a') as log:
                log.write(json.dumps(status.info) + '\n')
        if update % args.save_every == 0 or update == total:
            status.update(stage='saving')
            save(output, adapter, optimizer, config, update,
                 dict(update=update, total=total, warmup=warmup, peak=config['lr']), rank, world)
    loader.close()
    status.update(state='COMPLETE', stage='complete', update=total, total=total)
    status.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

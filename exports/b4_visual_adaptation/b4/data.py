"""Ordered MCQ data, whole-video decoding and bounded CPU prefetch."""

import json
import multiprocessing as mp
import os
import re
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly and only give the best option."


def read_rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def video_jobs(rows):
    groups = {}
    for row in rows:
        key = (row['video'], tuple(row.get('frame_indices', [])))
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def worker_init(model_path):
    global PROCESSOR
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(2)
    from transformers import AutoProcessor
    PROCESSOR = AutoProcessor.from_pretrained(model_path, padding_side='right')


def decode_video(row, max_frames):
    import decord
    from qwen_vl_utils.vision_process import smart_nframes, smart_resize
    from torchvision.transforms import InterpolationMode, functional as transforms

    reader = decord.VideoReader(row['video'], ctx=decord.cpu(0), num_threads=1)
    total, fps = len(reader), float(reader.get_avg_fps())
    indices = row.get('frame_indices')
    if indices is None:
        count = smart_nframes(dict(fps=2.0, max_frames=max_frames), total_frames=total, video_fps=fps)
        indices = torch.linspace(0, total - 1, count).round().long().tolist()
    frames = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
    # Same qwen-vl-utils resize budget as the original Qwen input path.
    min_pixels = 16 * 32 * 32
    max_pixels = max(min(768 * 32 * 32, 128000 * 32 * 32 / len(indices) * 2), int(min_pixels * 1.05))
    height, width = smart_resize(frames.shape[2], frames.shape[3], factor=32,
                                 min_pixels=min_pixels, max_pixels=max_pixels)
    frames = transforms.resize(frames, [height, width],
                               interpolation=InterpolationMode.BICUBIC, antialias=True).float()
    metadata = dict(fps=fps, frames_indices=indices, total_num_frames=total, video_backend='decord')
    return frames, metadata


def question_prompt(row):
    choices = [re.sub(r'^\s*[A-Z]\.\s*', '', text.strip()) for text in row['options']]
    options = '\n'.join(f'{chr(65 + i)}. {text}' for i, text in enumerate(choices))
    body = row['question'] + '\nOptions:\n' + options
    return body + '\n' + ANSWER_INSTRUCTION, (row['question'], options)


class VideoProcessorCache:
    """Reuse the CPU pixel preparation of this video, including timestamp metadata."""

    def __init__(self, processor):
        self.processor, self.value = processor, None

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        if self.value is None:
            self.value = dict(self.processor(*args, **kwargs))
        # The Qwen processor pops metadata from the returned mapping.
        return dict(self.value)


def prepare_video(rows, max_frames):
    started = time.monotonic()
    frames, metadata = decode_video(rows[0], max_frames)
    questions, media = [], None
    tokenizer = PROCESSOR.tokenizer
    video_processor = PROCESSOR.video_processor
    PROCESSOR.video_processor = VideoProcessorCache(video_processor)
    for row in rows:
        prompt, segments = question_prompt(row)
        messages = [dict(role='system', content='You are a helpful assistant.'),
                    dict(role='user', content=[dict(type='video', video=row['video']),
                                              dict(type='text', text=prompt)])]
        text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = dict(PROCESSOR(text=text, videos=[frames], video_metadata=[dict(metadata)],
                                do_sample_frames=False, do_resize=False,
                                return_tensors='pt', padding=True, padding_side='right'))
        # Qwen3 interleaves a timestamp and visual group for each temporal pair.
        temporal, height, width = inputs['video_grid_thw'][0].tolist()
        token_types = inputs.get('mm_token_type_ids')
        if token_types is not None:
            video = token_types[0] == 2
            groups = (video & ~torch.cat((video.new_zeros(1), video[:-1]))).sum().item()
            if groups == temporal:
                inputs['video_grid_thw'] = inputs['video_grid_thw'].new_tensor([[1, height, width]] * temporal)
        # The video tensors are identical across siblings; keep one shared copy.
        if media is None:
            media = {k: inputs[k] for k in ('pixel_values_videos', 'video_grid_thw')}
        inputs.update(media)
        ids = inputs['input_ids'][0].tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        offsets = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        positions, lower = set(), 0
        for segment in segments:
            start = decoded.index(segment, lower)
            stop = start + len(segment)
            positions.update(i for i, (a, b) in enumerate(offsets)
                             if a < stop and b > start and ids[i] not in tokenizer.all_special_ids)
            lower = stop
        option_ids = [tokenizer.encode(chr(65 + i), add_special_tokens=False)[0]
                      for i in range(len(row['options']))]
        questions.append(dict(id=row['id'], inputs=inputs, positions=sorted(positions),
                              option_ids=option_ids, gold=option_ids[ord(row['answer']) - 65]))
    PROCESSOR.video_processor = video_processor
    return dict(questions=questions, cpu_seconds=time.monotonic() - started)


class Prefetch:
    def __init__(self, jobs, model_path, max_frames=128, workers=1, depth=2):
        self.jobs, self.depth, self.frames = iter(jobs), depth, max_frames
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                        initializer=worker_init, initargs=(model_path,))
        self.pending = deque()
        self.fill()

    def fill(self):
        while len(self.pending) < self.depth:
            rows = next(self.jobs, None)
            if rows is None:
                break
            self.pending.append(self.pool.submit(prepare_video, rows, self.frames))

    def take(self):
        started = time.monotonic()
        result = self.pending.popleft().result()
        result['wait_seconds'] = time.monotonic() - started
        self.fill()
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

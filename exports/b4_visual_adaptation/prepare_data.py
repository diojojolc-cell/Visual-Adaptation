"""Convert an existing all-QA training manifest to the portable B4 JSONL format."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-root', type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    frames = json.loads(args.frames.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as output:
        for row in rows:
            qid = row['qa_id']
            # Existing label filenames are keyed by the question ID; this is a lookup, not a content audit.
            label_file = args.labels / (hashlib.sha256(qid.encode()).hexdigest() + '.json')
            answer = json.loads(label_file.read_text())['answer']
            video = str(args.video_root / row['meta']['file_name']) if args.video_root else row['video_path']
            sample = dict(id=qid, video=video, question=row['question'], options=row['options'],
                          answer=answer, frame_indices=frames[qid])
            output.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f'Wrote {len(rows)} training questions to {args.output}')


if __name__ == '__main__':
    main()

import argparse
from pathlib import Path

from tqdm import tqdm

from extensions import VideoRealFakeDetector


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('video_dir', type=Path)

    return parser.parse_args()


def main(args):
    video_dir: Path = args.video_dir
    true_cnt, false_cnt = 0, 0

    for video_path in tqdm(sorted(video_dir.rglob("*.webm"))):
        result = VideoRealFakeDetector()(str(video_path))
        if result:
            true_cnt += 1
        else:
            false_cnt += 1
    print(true_cnt, false_cnt)
    return 0


if __name__ == '__main__':
    main(parse_args())

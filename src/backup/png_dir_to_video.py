"""Combine a directory of PNG files into an MP4 video sorted by filename."""

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MP4 video from PNG frames sorted by filename."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing PNG frames",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the MP4 file to create",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the output video (default: 30)",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec to use (default: mp4v)",
    )
    return parser.parse_args()


def collect_frames(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        print(f"ERROR: No PNG files found in {input_dir}")
        sys.exit(1)
    return frames


def read_frame_size(frame_path: Path) -> tuple[int, int]:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"ERROR: Unable to read {frame_path}")
        sys.exit(1)
    height, width = frame.shape[:2]
    return width, height


def write_video(frames: list[Path], output: Path, fps: float, codec: str) -> None:
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        print(f"ERROR: Unable to read {frames[0]}")
        sys.exit(1)

    height, width = first_frame.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("ERROR: Failed to initialize video writer")
        sys.exit(1)

    for idx, frame_path in enumerate(frames, 1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            writer.release()
            print(f"ERROR: Unable to read frame {frame_path}")
            sys.exit(1)
        if frame.shape[:2] != (height, width):
            writer.release()
            print(
                "ERROR: Frame size mismatch. All frames must share the same resolution."
            )
            sys.exit(1)
        writer.write(frame)
        if idx % 50 == 0 or idx == len(frames):
            print(f"Written {idx}/{len(frames)} frames", end="\r", flush=True)

    writer.release()
    print()


def main() -> None:
    args = parse_args()
    frames = collect_frames(args.input_dir)
    print(f"Found {len(frames)} PNG frames in {args.input_dir}")
    print(f"Writing video to {args.output}")
    write_video(frames, args.output, args.fps, args.codec)
    print("Video creation complete.")


if __name__ == "__main__":
    main()

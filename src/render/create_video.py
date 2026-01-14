"""
Combine PNG sequence into video using ffmpeg.

Usage:
    python src/render/create_video.py -i output/high_water/images -o output/rigid_complex/raw.mp4 --fps 60
"""

import subprocess
import sys
import argparse
from pathlib import Path


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def count_frames(input_dir: Path) -> int:
    """Count PNG frames in input directory."""
    frames = list(input_dir.glob("frame_*.png"))
    return len(frames)


def create_video(input_dir: Path, output_file: Path, fps: int) -> bool:
    """Create video from PNG sequence."""
    input_pattern = input_dir / "frame_%04d.png"
    
    cmd = [
        "ffmpeg",
        "-y",                                 # Overwrite output
        "-framerate", str(fps),
        "-i", str(input_pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        str(output_file)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Create video from PNG sequence')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing PNG frames')
    parser.add_argument('-o', '--output', required=True, help='Output video file path')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_file = Path(args.output)
    fps = args.fps
    
    print("=" * 60)
    print("Create Video from PNG Sequence")
    print("=" * 60)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("ERROR: ffmpeg is not installed!")
        print("Install with: sudo apt install ffmpeg")
        sys.exit(1)
    
    # Check input
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    num_frames = count_frames(input_dir)
    if num_frames == 0:
        print(f"ERROR: No PNG frames found in {input_dir}")
        sys.exit(1)
    
    print(f"Input directory: {input_dir}")
    print(f"Found {num_frames} frames")
    print(f"FPS: {fps}")
    print(f"Output: {output_file}")
    print("-" * 60)
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video
    if create_video(input_dir, output_file, fps):
        print("-" * 60)
        print(f"Video created successfully!")
        print(f"Output: {output_file}")
        duration = num_frames / fps
        print(f"Duration: {duration:.2f} seconds ({num_frames} frames @ {fps} fps)")
        print("=" * 60)
    else:
        print("ERROR: Failed to create video")
        sys.exit(1)


if __name__ == "__main__":
    main()

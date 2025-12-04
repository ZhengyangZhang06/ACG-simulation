"""
Combine PNG sequence into video using ffmpeg.

Usage:
    python create_video.py

Requirements:
    ffmpeg must be installed (sudo apt install ffmpeg)
"""

import subprocess
import sys
from pathlib import Path

# --- Configuration ---------------------------------------------------------
INPUT_DIR = Path("/home/cxy/Desktop/ACG/ACG-simulation/output/render")
INPUT_PATTERN = "frame_%04d.png"              # ffmpeg pattern for numbered frames
OUTPUT_FILE = Path("/home/cxy/Desktop/ACG/ACG-simulation/output/fluid_animation.mp4")

# Video settings
FPS = 60
CODEC = "libx264"                             # H.264 codec
PIXEL_FORMAT = "yuv420p"                      # Compatible with most players
CRF = 18                                      # Quality (0-51, lower = better, 18-23 recommended)
PRESET = "medium"                             # Encoding speed: ultrafast, fast, medium, slow, veryslow
# ---------------------------------------------------------------------------


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


def count_frames() -> int:
    """Count PNG frames in input directory."""
    frames = list(INPUT_DIR.glob("frame_*.png"))
    return len(frames)


def create_video() -> bool:
    """Create video from PNG sequence."""
    input_path = INPUT_DIR / INPUT_PATTERN
    
    cmd = [
        "ffmpeg",
        "-y",                                 # Overwrite output
        "-framerate", str(FPS),
        "-i", str(input_path),
        "-c:v", CODEC,
        "-pix_fmt", PIXEL_FORMAT,
        "-crf", str(CRF),
        "-preset", PRESET,
        str(OUTPUT_FILE)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Create Video from PNG Sequence")
    print("=" * 60)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("ERROR: ffmpeg is not installed!")
        print("Install with: sudo apt install ffmpeg")
        sys.exit(1)
    
    # Check input
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    num_frames = count_frames()
    if num_frames == 0:
        print(f"ERROR: No PNG frames found in {INPUT_DIR}")
        sys.exit(1)
    
    print(f"Input directory: {INPUT_DIR}")
    print(f"Found {num_frames} frames")
    print(f"FPS: {FPS}")
    print(f"Output: {OUTPUT_FILE}")
    print("-" * 60)
    
    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video
    if create_video():
        print("-" * 60)
        print(f"Video created successfully!")
        print(f"Output: {OUTPUT_FILE}")
        duration = num_frames / FPS
        print(f"Duration: {duration:.2f} seconds ({num_frames} frames @ {FPS} fps)")
        print("=" * 60)
    else:
        print("ERROR: Failed to create video")
        sys.exit(1)


if __name__ == "__main__":
    main()

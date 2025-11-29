"""
Surface reconstruction from SPH particle PLY files using splashsurf CLI.

This script converts particle point clouds to water surface meshes.

Usage:
    python reconstruct_surface.py

Requirements:
    cargo install splashsurf
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---------------------------------------------------------
INPUT_DIR = Path("/home/cxy/Desktop/ACG/ACG-simulation/src/materials/fluid/ply_output")
OUTPUT_DIR = Path("/home/cxy/Desktop/ACG/ACG-simulation/src/materials/fluid/mesh_output")

# Particle parameters (must match your simulation)
PARTICLE_RADIUS = 0.01

# Reconstruction parameters
SMOOTHING_LENGTH = 2.0        # Smoothing kernel radius multiplier
CUBE_SIZE = 1.5               # Grid cell size multiplier (smaller = finer mesh)
SURFACE_THRESHOLD = 0.6       # Density threshold for surface detection

# Processing
NUM_WORKERS = 4               # Number of parallel workers (set to 1 for debugging)
# ---------------------------------------------------------------------------


def reconstruct_single_frame(ply_path: Path, output_path: Path) -> str:
    """Reconstruct surface mesh from a single PLY particle file using splashsurf CLI."""
    try:
        cmd = [
            "splashsurf", "reconstruct",
            str(ply_path),
            "-o", str(output_path),
            "-q",
            f"--particle-radius={PARTICLE_RADIUS}",
            f"--smoothing-length={SMOOTHING_LENGTH}",
            f"--cube-size={CUBE_SIZE}",
            f"--surface-threshold={SURFACE_THRESHOLD}",
            "--subdomain-grid=on",
            "--mesh-cleanup=on",
            "--mesh-smoothing-weights=on",
            "--mesh-smoothing-iters=25",
            "--normals=on",
            "--normals-smoothing-iters=10",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return f"ERROR: {ply_path.name} - {result.stderr.strip()}"
        
        if output_path.exists():
            return f"OK: {ply_path.name} -> {output_path.name}"
        else:
            return f"ERROR: {ply_path.name} - Output file not created"
    
    except subprocess.TimeoutExpired:
        return f"ERROR: {ply_path.name} - Timeout"
    except Exception as e:
        return f"ERROR: {ply_path.name} - {str(e)}"


def reconstruct_single_frame_wrapper(args):
    """Wrapper for parallel processing."""
    return reconstruct_single_frame(*args)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Surface Reconstruction using splashsurf CLI")
    print("=" * 60)
    
    # Check splashsurf is installed
    try:
        result = subprocess.run(["splashsurf", "--version"], capture_output=True, text=True)
        print(f"Using: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: splashsurf not found. Install with: cargo install splashsurf")
        return
    
    # Check input directory
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all PLY files
    ply_files = sorted(INPUT_DIR.glob("*.ply"))
    if not ply_files:
        print(f"ERROR: No PLY files found in {INPUT_DIR}")
        return
    
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(ply_files)} PLY files")
    print(f"Particle radius:  {PARTICLE_RADIUS}")
    print(f"Smoothing length: {SMOOTHING_LENGTH}")
    print(f"Cube size:        {CUBE_SIZE}")
    print(f"Surface threshold:{SURFACE_THRESHOLD}")
    print(f"Workers:          {NUM_WORKERS}")
    print("-" * 60)
    
    # Prepare tasks
    tasks = []
    for ply_path in ply_files:
        output_path = OUTPUT_DIR / f"{ply_path.stem}.obj"
        tasks.append((ply_path, output_path))
    
    # Process files
    completed = 0
    errors = 0
    
    if NUM_WORKERS == 1:
        # Sequential processing (for debugging)
        for ply_path, output_path in tasks:
            result = reconstruct_single_frame(ply_path, output_path)
            completed += 1
            if result.startswith("ERROR"):
                errors += 1
            print(f"[{completed}/{len(tasks)}] {result}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(reconstruct_single_frame_wrapper, task): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result.startswith("ERROR"):
                    errors += 1
                print(f"[{completed}/{len(tasks)}] {result}")
    
    print("-" * 60)
    print(f"Completed: {completed - errors}/{len(tasks)} successful, {errors} errors")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
Surface reconstruction from SPH particle PLY files using splashsurf CLI.
This script converts particle point clouds to water surface meshes.

Usage:
    python reconstruct_surface.py --scene src/configs/cat_dynamic.json
"""

import os
import subprocess
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_config(scene_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(scene_file, 'r') as f:
        return json.load(f)


def get_scene_name(scene_file: str) -> str:
    """Extract scene name from config file path."""
    return Path(scene_file).stem


def reconstruct_single_frame(ply_path: Path, output_path: Path, particle_radius: float, 
                           smoothing_length: float, cube_size: float, surface_threshold: float) -> str:
    """Reconstruct surface mesh from a single PLY particle file using splashsurf CLI."""
    try:
        command = "splashsurf reconstruct {} -o {} -q -r={} -l={} -c={} -t={} --subdomain-grid=on --mesh-cleanup=on --mesh-smoothing-weights=on --mesh-smoothing-iters=25 --normals=on --normals-smoothing-iters=10".format(
            str(ply_path), str(output_path), particle_radius, smoothing_length, cube_size, surface_threshold
        )
        
        result = subprocess.run(
            command,
            shell=True,
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
    parser = argparse.ArgumentParser(description='Surface reconstruction using splashsurf CLI')
    parser.add_argument('--scene', required=True, help='Scene configuration file (e.g., src/configs/cat_dynamic.json)')
    parser.add_argument('--smoothing-length', type=float, default=2.0, help='Smoothing kernel radius multiplier')
    parser.add_argument('--cube-size', type=float, default=0.5, help='Grid cell size multiplier')
    parser.add_argument('--surface-threshold', type=float, default=0.6, help='Density threshold for surface detection')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.scene)
    scene_name = get_scene_name(args.scene)
    
    # Extract particle radius from config
    particle_radius = config.get("Configuration", {}).get("particleRadius", 0.01)
    
    # Set up directories
    input_dir = Path(f"output/{scene_name}/ply_output")
    output_dir = Path(f"output/{scene_name}/mesh_output")
    
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
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PLY files
    ply_files = sorted(input_dir.glob("*.ply"))
    if not ply_files:
        print(f"ERROR: No PLY files found in {input_dir}")
        return
    
    print(f"Scene:            {scene_name}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(ply_files)} PLY files")
    print(f"Particle radius:  {particle_radius}")
    print(f"Smoothing length: {args.smoothing_length}")
    print(f"Cube size:        {args.cube_size}")
    print(f"Surface threshold:{args.surface_threshold}")
    print(f"Workers:          {args.workers}")
    print("-" * 60)
    
    # Prepare tasks
    tasks = []
    for ply_path in ply_files:
        output_path = output_dir / f"{ply_path.stem}.obj"
        tasks.append((ply_path, output_path, particle_radius, args.smoothing_length, 
                     args.cube_size, args.surface_threshold))
    
    # Process files
    completed = 0
    errors = 0
    
    if args.workers == 1:
        # Sequential processing (for debugging)
        for task in tasks:
            result = reconstruct_single_frame(*task)
            completed += 1
            if result.startswith("ERROR"):
                errors += 1
            print(f"[{completed}/{len(tasks)}] {result}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(reconstruct_single_frame_wrapper, task): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result.startswith("ERROR"):
                    errors += 1
                print(f"[{completed}/{len(tasks)}] {result}")
    
    print("-" * 60)
    print(f"Completed: {completed - errors}/{len(tasks)} successful, {errors} errors")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
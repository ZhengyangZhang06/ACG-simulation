import os
import shutil
import subprocess
import argparse
from tqdm import tqdm

BLENDER_BIN = r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_EXEC = os.path.join(WORK_DIR, "rendering_script.py")


def perform_rendering(src_dir, src_name, config, final_dir):
    # Output path within the frame folder
    frame_result = os.path.join(src_dir, config.image_name)
    # Output path within the汇总 folder
    summary_result = os.path.join(final_dir, f"{src_name}.png")
    
    execution_cmd = [
        BLENDER_BIN,
        '-b', config.blend_scene,
        '--python', RENDER_EXEC,
        '--',
        config.device,
        '0',
        src_dir,
        frame_result
    ]
    
    if config.hide_output:
        subprocess.run(execution_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(execution_cmd)
    
    if os.path.exists(frame_result):
        shutil.copy2(frame_result, summary_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Blender rendering script')
    parser.add_argument('--blend_scene', type=str, required=True, help='Path to the Blender scene file')
    parser.add_argument('--image_name', type=str, default='render.png', help='Name of the rendered image file')
    parser.add_argument('--frames_dir', type=str, required=True, help='Directory containing frame subfolders')
    parser.add_argument('--device', type=str, default='OPTIX', help='Rendering device type (OPTIX/CUDA/HIP)')
    parser.add_argument('--hide_output', action='store_true', help='Quiet mode, hides Blender output')
    parser.add_argument('--match_file', type=str, default=None,
                        help='Only render frames containing this file')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to collect rendering results')

    settings = parser.parse_args()
    
    settings.frames_dir = os.path.abspath(settings.frames_dir)
    settings.blend_scene = os.path.abspath(settings.blend_scene)
    
    if settings.results_dir:
        collection_dir = os.path.abspath(settings.results_dir)
    else:
        collection_dir = os.path.join(os.path.dirname(settings.frames_dir), "renders")
    
    if os.path.exists(collection_dir):
        shutil.rmtree(collection_dir)
    os.makedirs(collection_dir, exist_ok=True)
    print(f"Results will be collected in: {collection_dir}")
    
    all_frames = os.listdir(settings.frames_dir)
    
    if settings.match_file:
        all_frames = [
            f for f in all_frames 
            if os.path.isdir(os.path.join(settings.frames_dir, f)) and
               os.path.exists(os.path.join(settings.frames_dir, f, settings.match_file))
        ]
        print(f"Filtered to {len(all_frames)} frames containing '{settings.match_file}'")
    
    all_frames.sort(key=lambda x: int(x))
    total_count = len(all_frames)

    print(f"Starting render of {total_count} frames")
    print(f"Device: {settings.device}")
    print("-" * 50)

    for item in tqdm(all_frames, desc="Processing"):
        current_path = os.path.join(settings.frames_dir, item)
        perform_rendering(current_path, item, settings, collection_dir)

    print(f"\nRendering complete! Results located at: {collection_dir}")

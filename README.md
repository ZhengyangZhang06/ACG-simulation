# Rigid Body Toy Simulation

This repo contains a minimal rigid-body simulator written from scratch with only Python's standard library. It now writes a numbered sequence of Wavefront OBJ meshes that Blender can import directly for visualization.

## Requirements

- Python 3.10+

## Running the simulator

```powershell
cd c:\Users\72667\Desktop\2025Fall\ACG\ACG-simulation
python -m src.rigid_body_sim
```

The script produces `output/obj_frames/frame_0001.obj`, `frame_0002.obj`, … where every file encodes the full scene for one simulation step.

## Blender visualization workflow

1. **Run the simulator** so that `output/obj_frames` is populated with `frame_####.obj` files.
2. **Open Blender** → `File` → `Import` → `Wavefront (.obj)`.
3. In the import panel, navigate to `output/obj_frames`, select `frame_0001.obj`, and enable the **Animation** checkbox. Blender will automatically load the numbered sequence as an animated mesh (each file acts as a frame of the animation).
4. Click **Import OBJ** and press the spacebar/Play button on the timeline to preview the rigid bodies moving.

Each OBJ contains named groups (`CrateA`, `CrateB`, `CrateC`, …), so you can assign materials per body after the import if desired.

Feel free to tweak `demo_scene()` or create your own set of bodies to experiment with different scenarios.
# ACG-simulation

## Usage

### Step 1: Run Fluid Simulation
```bash
python src/materials/fluid/run.py
```
**Output:** `src/materials/fluid/ply_output/*.ply` (particle point clouds for each frame)

### Step 2: Surface Reconstruction
```bash
python src/blender/reconstruct_surface.py
```
**Output:** `src/materials/fluid/mesh_output/*.obj` (water surface meshes)

### Step 3: Import to Blender
```bash
~/Downloads/blender-5.0.0-linux-x64/blender --background --python /home/cxy/Desktop/ACG/ACG-simulation/src/blender/import_to_blender.py
```
**Output:** `output/fluid_animation.blend` (Blender project file with animated meshes)

### Step 4: Render Animation
```bash
~/Downloads/blender-5.0.0-linux-x64/blender -b /home/cxy/Desktop/ACG/ACG-simulation/output/fluid_animation.blend --python render_animation.py
```
**Output:** `output/render/frame_XXXX.png` (rendered PNG image sequence)

### Step 5: Create Video
```bash
python /home/cxy/Desktop/ACG/ACG-simulation/src/blender/create_video.py
```
**Output:** `output/fluid_animation.mp4` (final video)

### References
[SPH](https://dl.acm.org/doi/10.5555/846276.846298)

[WCSPH](https://dl.acm.org/doi/10.5555/1272690.1272719)

[PCISPH](https://dl.acm.org/doi/abs/10.1145/1576246.1531346)

[DFSPH](https://dl.acm.org/doi/10.1145/2786784.2786796)

[rigid-fluid Coupling](https://dl.acm.org/doi/10.1145/2185520.2185558)

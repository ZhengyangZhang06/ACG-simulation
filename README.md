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

### Loading custom meshes

- Place a convex OBJ mesh under `assets/meshes` (or point to any other directory) and load it with `load_convex_mesh(Path("path/to/mesh.obj"))`.
- Pass the resulting `MeshShape` to `RigidBody(shape=my_shape, size=None, ...)`. When `size` is omitted, the body derives its inertia from the mesh's bounding box.
- The default `demo_scene()` automatically loads `assets/meshes/convex_gem.obj` so you can see a convex polyhedron colliding with the wooden crates.

## Blender visualization workflow

1. **Run the simulator** so that `output/obj_frames` is populated with `frame_####.obj` files.
2. **Open Blender** → `File` → `Import` → `Wavefront (.obj)`.
3. In the import panel, navigate to `output/obj_frames`, select `frame_0001.obj`, and enable the **Animation** checkbox. Blender will automatically load the numbered sequence as an animated mesh (each file acts as a frame of the animation).
4. Click **Import OBJ** and press the spacebar/Play button on the timeline to preview the rigid bodies moving.

Each OBJ contains named groups (`CrateA`, `CrateB`, `CrateC`, …), so you can assign materials per body after the import if desired.

Feel free to tweak `demo_scene()` or create your own set of bodies to experiment with different scenarios.
# ACG-simulation

## Environment Setup

### Conda Environment
Create and activate a conda environment with the required Python version:
```bash
conda create -n acg python=3.11
conda activate acg
```

### Dependencies
Install Taichi (tested with version 1.7.4):
```bash
pip install taichi==1.7.4
pip install trimesh
pip install scipy
```
Install Rust and splashsurf for surface reconstruction:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
cargo install splashsurf
```

### Blender
Download and install Blender 5.0 from the [official website](https://www.blender.org/download/). Ensure `blender` is in your PATH.

Alternatively, download via command line:
```bash
# Download Blender 5.0.0 (adjust version as needed)
wget https://download.blender.org/release/Blender5.0/blender-5.0.0-linux-x64.tar.xz
tar -xf blender-5.0.0-linux-x64.tar.xz
export PATH=$PWD/blender-5.0.0-linux-x64:$PATH
```

Test the setup:
```bash
python -c "import taichi as ti; print('Taichi version:', ti.__version__)"
blender --version
```

## Usage

### Step 1: Run Fluid Simulation
```bash
python src/run.py --scene src/configs/basic_fluid.json 
```
**Output:** `output/fluid/ply_output/*.ply` (particle point clouds for each frame), `output/fluid/images/*.png` (rendered images)


### Step 2: Surface Reconstruction (no need for rigid simulation)
```bash
python src/render/reconstruct_surface.py
```

### Step 3: Import to Blender and Render Animation
```bash
blender --background --python src/render/render_blender.py -- --scene src/configs/basic_fluid.json
```
To resume rendering from a specific OBJ index (useful for large sequences):
```bash
blender --background --python src/render/render_blender.py -- --scene src/configs/cat_dynamic.json --start-obj 100
```
**Output:** `output/{scene_name}/render/frame_XXXX.png` (rendered PNG image sequence)

### Step 4: Create Video
```bash
python src/render/create_video.py
```
**Output:** `output/{scene_name}/fluid_animation.mp4` (final video)

### References
[SPH](https://dl.acm.org/doi/10.5555/846276.846298)

[WCSPH](https://dl.acm.org/doi/10.5555/1272690.1272719)

[PCISPH](https://dl.acm.org/doi/abs/10.1145/1576246.1531346)

[DFSPH](https://dl.acm.org/doi/10.1145/2786784.2786796)

[rigid-fluid Coupling](https://dl.acm.org/doi/10.1145/2185520.2185558)

[SPH_Taichi](https://github.com/erizmr/SPH_Taichi)
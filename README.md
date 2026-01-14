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

### Step 1: Run Fluid / Rigid Simulation
```bash
python src/run.py --scene src/configs/basic_fluid.json 
```
**Output:** `output/{scene_name}/ply_output/*.ply` (particle point clouds for each frame), `output/{scene_name}/images/*.png` (rendered images)


### Step 2: Surface Reconstruction (no need for rigid simulation)
```bash
python src/render/reconstruct_surface.py --scene src/configs/cat_float.json
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
python src/render/create_video.py -i output/high_water/images -o output/high_water/raw.mp4 --fps 60
```
```bash
python src/render/create_video.py -i output/high_water/render -o output/high_water/animation.mp4 --fps 60
```
**Output:** `output/{scene_name}/animation.mp4` (final video)

## Output Structure
```
output/{scene_name}/
├── mesh_output/
│   ├── fluid_*.obj (only for fluid scene)
│   └── obj_{objectId}/
│       └── obj_{objectId}_*.obj
├── ply_output/ (only for fluid scene)
├── images/ (only for fluid scene)
└── render/
```


### References
[SPH](https://dl.acm.org/doi/10.5555/846276.846298)

[WCSPH](https://dl.acm.org/doi/10.5555/1272690.1272719)

[PCISPH](https://dl.acm.org/doi/abs/10.1145/1576246.1531346)

[DFSPH](https://dl.acm.org/doi/10.1145/2786784.2786796)

[rigid-fluid Coupling](https://dl.acm.org/doi/10.1145/2185520.2185558)

[SPH_Taichi](https://github.com/erizmr/SPH_Taichi)
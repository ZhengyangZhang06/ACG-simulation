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


### References
[SPH](https://dl.acm.org/doi/10.5555/846276.846298)

[WCSPH](https://dl.acm.org/doi/10.5555/1272690.1272719)

[PCISPH](https://dl.acm.org/doi/abs/10.1145/1576246.1531346)

[DFSPH](https://dl.acm.org/doi/10.1145/2786784.2786796)

[rigid-fluid Coupling](https://dl.acm.org/doi/10.1145/2185520.2185558)

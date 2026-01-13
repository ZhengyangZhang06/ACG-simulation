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
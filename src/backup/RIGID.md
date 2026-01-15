# Rigid body simulation 
## Code structure
- `rigid_body_sim.py`: Main simulation code for rigid body dynamics.
- `rendering_script.py`: Blender script for rendering simulation results.
- `render_pic.py`: The script to generate pictures from simulation data.
- `split_obj_frames.py`: Utility script to split OBJ files into individual frames for rendering.
- `png_dir_to_video.py`: Utility script to compile rendered images into a video file.
## Workflow
1. **Simulation Setup**: Define rigid bodies, their properties, and initial conditions in `rigid_body_sim.py`. You are free to modify the mesh paths and other parameters as needed. Some examples are provided in `../../assets/meshes/`.
2. **Run Simulation**: Execute `rigid_body_sim.py` to perform the rigid body simulation. This will generate output data files containing the simulation results. Default in `outputs_24/temp`.
3. **Mesh post processing**: Use `split_obj_frames.py` to split the output OBJ files into individual objects for each frame. This is necessary for proper rendering in Blender.
Do this by running:
```bash
python3 split_obj_frames.py outputs_24/temp output_splited
```
4. **Fully configurable rendering**: Open blender and load the first frame's OBJ files from the `output_splited` directory (simply dragging). Then save the .blend file. 

5. **Run Rendering Script**: Execute `rendering_script.py` 
```shell
python3 ./render_pic.py --frames_dir ./output_splited/  --blend_scene ./Untitled.blend (Path to saved .blend file)
```
    to render images from the simulation data. Adjust camera settings and rendering parameters as needed within the script. It will save the rendered images in `./renders/` by default.

6. **Generate video (optional)**: Execute the code `png_dir_to_video.py` to compile the rendered images into a video file. Be sure to set the same fps as defined in the simulation. 
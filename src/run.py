import taichi as ti
import numpy as np
import os
import argparse
from configs.config_builder import SimConfig
from materials.fluid.particle_system import ParticleSystem
from materials.fluid.WCSPH import WCSPHSolver
from blender.render_2d import Renderer2D

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

def export_ply(ps, frame, output_dir, obj_id=0):
    num_particles = ps.particle_num[None]
    np_object_id = ps.object_id.to_numpy()[:num_particles]
    mask = (np_object_id == obj_id)
    
    np_pos = ps.x.to_numpy()[:num_particles][mask]
    np_color = ps.color.to_numpy()[:num_particles][mask]
    
    num_obj_particles = np_pos.shape[0]
    if num_obj_particles == 0:
        return
    
    writer = ti.tools.PLYWriter(num_vertices=num_obj_particles)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.add_vertex_color(np_color[:, 0], np_color[:, 1], np_color[:, 2])
    
    series_prefix = os.path.join(output_dir, "fluid.ply")
    writer.export_frame_ascii(frame, series_prefix)

def export_obj(ps, frame, output_dir, obj_id):
    if obj_id not in ps.object_collection:
        return
    obj_data = ps.object_collection[obj_id]
    if "mesh" not in obj_data:
        return
    
    obj_dir = os.path.join(output_dir, f"obj_{obj_id}")
    os.makedirs(obj_dir, exist_ok=True)
    
    obj_path = os.path.join(obj_dir, f"obj_{obj_id}_{frame:06d}.obj")
    with open(obj_path, "w") as f:
        e = obj_data["mesh"].export(file_type='obj')
        # Remove material references to avoid MTL file errors
        lines = e.split('\n')
        filtered_lines = [line for line in lines if not line.startswith(('mtllib', 'usemtl'))]
        f.write('\n'.join(filtered_lines))

def setup_lights(scene, lights_config):
    for light in lights_config:
        scene.point_light(tuple(light["position"]), color=tuple(light["color"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    lights_config = config.get_lights()
    render_config = config.get_render()
    scene_name = scene_path.split("/")[-1].split(".")[0]
    ps = ParticleSystem(config)
    solver = WCSPHSolver(ps, config)
    solver.initialize()

    # Calculate export intervals based on target fps
    number_of_steps_per_render = config.get_cfg("numberOfStepsPerRenderUpdate", 1)
    dt = config.get_cfg("dt", 0.0001)
    fps = render_config.get("fps", 60)
    render_fps = 1 / (number_of_steps_per_render * dt)  # Render frames per second
    export_interval = int(render_fps / fps)

    export_ply_enabled = config.get_cfg("exportPly", False)
    ply_output_dir = f"output/fluid/{scene_name}/ply_output"
    
    export_obj_enabled = config.get_cfg("exportObj", False)
    obj_output_dir = f"output/fluid/{scene_name}/mesh_output"
    
    export_images_enabled = config.get_cfg("exportImages", False)
    image_output_dir = f"output/fluid/{scene_name}/images"
    image_interval = export_interval  # Same as ply/obj
    
    export_2d_renders = config.get_cfg("export2DRenders", False)
    render_2d_output_dir = f"output/fluid/{scene_name}/render"
    if solver.is_bad_apple:
        render_2d_interval = number_of_steps_per_render * (fps / 60)
        # For badapple, max_frames is num_frames * badAppleFrameInterval
        bad_apple_frame_interval = config.get_cfg("badAppleFrameInterval", 2)
        max_frames = solver.num_frames * bad_apple_frame_interval
        export_interval = render_2d_interval
    else:
        render_2d_interval = export_interval
        max_frames = render_config.get("totalFrames", 300)
    
    show_window = not export_images_enabled
    
    if export_ply_enabled:
        os.makedirs(ply_output_dir, exist_ok=True)
    
    if export_obj_enabled:
        os.makedirs(obj_output_dir, exist_ok=True)
    
    if export_images_enabled:
        os.makedirs(image_output_dir, exist_ok=True)
    
    if export_2d_renders:
        os.makedirs(render_2d_output_dir, exist_ok=True)
        domain_start = config.get_cfg("domainStart")
        domain_end = config.get_cfg("domainEnd")
        window_size = config.get_cfg("windowSize", [1024, 1024])
        renderer_2d = Renderer2D(window_size[0], window_size[1], domain_start, domain_end, render_2d_output_dir)
    else:
        window_size = config.get_cfg("windowSize", [1024, 1024])
    window = ti.ui.Window('SPH', tuple(window_size), show_window=show_window, vsync=False)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    cam_pos = config.get_cfg("cameraPosition") or [8.0, 2.5, 6.0]
    cam_lookat = config.get_cfg("cameraLookat") or [-1.0, 0.0, -2.2]
    camera.position(*cam_pos)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(*cam_lookat)
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    movement_speed = 0.02

    domain_end = config.get_cfg("domainEnd")
    dim = len(domain_end)
    if dim == 2:
        x_max, y_max = domain_end
        z_max = 0.0
        box_anchors = ti.Vector.field(3, dtype=ti.f32, shape=4)
        box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
        box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
        box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
        box_anchors[3] = ti.Vector([x_max, y_max, 0.0])
        box_lines_indices = ti.field(int, shape=(2 * 4))
        for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3]):
            box_lines_indices[i] = val
    elif dim == 3:
        x_max, y_max, z_max = domain_end
        box_anchors = ti.Vector.field(3, dtype=ti.f32, shape=8)
        box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
        box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
        box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
        box_anchors[3] = ti.Vector([x_max, y_max, 0.0])
        box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
        box_anchors[5] = ti.Vector([0.0, y_max, z_max])
        box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
        box_anchors[7] = ti.Vector([x_max, y_max, z_max])
        box_lines_indices = ti.field(int, shape=(2 * 12))
        for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
            box_lines_indices[i] = val
    else:
        raise ValueError("domainEnd must have 2 or 3 elements")

    step_count = 0
    frame_count = 0
    image_frame_count = 0
    render_2d_frame_count = 0
    # Mouse interaction settings
    mouse_interaction_strength = config.get_cfg("mouseInteractionStrength", 100.0)

    while window.running:
        # Update video frame once per render frame (before physics steps)
        if hasattr(solver, 'update_frame'):
            solver.update_frame()
        
        # Handle mouse interaction
        if show_window and hasattr(solver, 'update_mouse_interaction'):
            mouse_pos = window.get_cursor_pos()
            # For 2D simulation, map screen coordinates to world XY plane
            # Assuming camera looks at XY plane from Z direction
            if dim == 2:
                world_x = mouse_pos[0] * x_max
                world_y = mouse_pos[1] * y_max
            else:
                # For 3D, use XY plane at Z=0
                world_x = mouse_pos[0] * x_max
                world_y = mouse_pos[1] * y_max
            
            # Check mouse buttons (note: camera now uses MMB)
            strength = 0.0
            if window.is_pressed(ti.ui.LMB):  # Left mouse button = attract
                strength = mouse_interaction_strength
            elif window.is_pressed(ti.ui.RMB):  # Right mouse button = repel
                strength = -mouse_interaction_strength
            
            solver.update_mouse_interaction(world_x, world_y, strength)
                   
        ps.copy_to_vis_buffer()

        if show_window:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.MMB)
        scene.set_camera(camera)

        setup_lights(scene, lights_config)
        scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer, per_vertex_radius=ps.radius_vis_buffer)
        scene.lines(box_anchors, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
        canvas.scene(scene)

        if show_window:
            window.show()

        # Export after rendering
        if (export_ply_enabled or export_obj_enabled) and step_count % export_interval == 0:
            if frame_count < max_frames:
                if export_ply_enabled:
                    export_ply(ps, frame_count, ply_output_dir, obj_id=0)
                    print(f"Exported PLY frame {frame_count}")
                if export_obj_enabled:
                    for r_body_id in ps.object_id_rigid_body:
                        export_obj(ps, frame_count, obj_output_dir, r_body_id)
                    if len(ps.object_id_rigid_body) > 0:
                        print(f"Exported OBJ frame {frame_count}")
                frame_count += 1
        
        if export_images_enabled and step_count % image_interval == 0:
            if image_frame_count < max_frames:
                window.save_image(os.path.join(image_output_dir, f"frame_{image_frame_count:04d}.png"))
                print(f"Exported image frame {image_frame_count}")
                image_frame_count += 1
        
        # Export 2D renders
        if export_2d_renders and step_count % render_2d_interval == 0:
            if render_2d_frame_count < max_frames:
                # Gather particle data
                num_particles = ps.particle_num[None]
                positions_np = ps.x.to_numpy()[:num_particles, :2]  # Only XY
                velocities_np = ps.v.to_numpy()[:num_particles, :2]
                
                particles_data = {
                    'positions': positions_np,
                    'velocities': velocities_np,
                    'radius': ps.particle_radius
                }
                
                # Check if Bad Apple mode
                bad_apple_data = None
                render_config_local = None
                if hasattr(solver, 'is_bad_apple') and solver.is_bad_apple:
                    current_frame = solver.current_frame_field[None]
                    bad_apple_data = {
                        'jfa_results': solver._load_frame_jfa(current_frame),
                        'image_data': solver.frames[current_frame],
                        'size': solver.bad_apple_size
                    }
                    render_config_local = {
                        'maxVelocityFor2DRender': config.get_cfg('maxVelocityFor2DRender', 50.0),
                        'distThresholdFor2DRender': config.get_cfg('distThresholdFor2DRender', 10.0)
                    }
                
                renderer_2d.render_frame(particles_data, render_2d_frame_count, bad_apple_data, render_config_local)
                print(f"Exported 2D render frame {render_2d_frame_count}")
                render_2d_frame_count += 1
        
        for i in range(config.get_cfg("numberOfStepsPerRenderUpdate") or 1):
            solver.step()
            step_count += 1
            
        if ((export_ply_enabled or export_obj_enabled) and frame_count >= max_frames) or (export_images_enabled and image_frame_count >= max_frames) or (export_2d_renders and render_2d_frame_count >= max_frames):
            break

    print(f"Simulation Finished")
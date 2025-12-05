import taichi as ti
import numpy as np
import os
from particle_system import ParticleSystem
from WCSPH import WCSPHSolver

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

config = {
    "domainStart": [0.0, 0.0, 0.0],
    "domainEnd": [5.0, 3.0, 2.0],
    "particleRadius": 0.01,
    "numberOfStepsPerRenderUpdate": 1,
    "exportPly": True,
    "plyOutputDir": "output/fluid/ply_output",  # Directory for exported PLY files
    "exportInterval": 42,  # steps per frame at 60 FPS (1/60 / 0.0004 ≈ 42)
    "maxFrames": 300,  # 5 seconds at 60 FPS
    "exportImages": True,  # New option to export images
    "imageOutputDir": "output/fluid/images",  # Directory for exported images
    "imageInterval": 42,  # Same as exportInterval for consistency
    "FluidBlocks": [
        {
            "objectId": 0,
            "start": [0.1, 0.1, 0.5],
            "end": [1.2, 2.9, 1.6],
            "translation": [0.2, 0.0, 0.2],
            "scale": [1, 1, 1],
            "velocity": [0.0, -1.0, 0.0],
            "density": 1000.0,
            "color": [50, 100, 200]
        }
    ]
}

def export_ply(ps, frame, output_dir):
    num_particles = ps.particle_num[None]
    np_pos = ps.x.to_numpy()[:num_particles]
    np_color = ps.color.to_numpy()[:num_particles]
    
    writer = ti.tools.PLYWriter(num_vertices=num_particles)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.add_vertex_color(np_color[:, 0], np_color[:, 1], np_color[:, 2])
    
    series_prefix = os.path.join(output_dir, "fluid.ply")
    writer.export_frame_ascii(frame, series_prefix)

if __name__ == "__main__":
    ps = ParticleSystem(config)
    solver = WCSPHSolver(ps)

    export_ply_enabled = config.get("exportPly", False)
    ply_output_dir = config.get("plyOutputDir", "ply_output")
    export_interval = config.get("exportInterval", 42)
    max_frames = config.get("maxFrames", 300)
    
    export_images_enabled = config.get("exportImages", False)
    image_output_dir = config.get("imageOutputDir", "images")
    image_interval = config.get("imageInterval", 42)
    
    show_window = not export_images_enabled
    
    if export_ply_enabled:
        os.makedirs(ply_output_dir, exist_ok=True)
    
    if export_images_enabled:
        os.makedirs(image_output_dir, exist_ok=True)

    window = ti.ui.Window('SPH', (1024, 1024), show_window=show_window, vsync=False)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(8.0, 2.5, 6.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-1.0, 0.0, -2.2)
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    movement_speed = 0.02

    x_max, y_max, z_max = config["domainEnd"]
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

    step_count = 0
    frame_count = 0
    image_frame_count = 0

    while window.running:
        for i in range(config["numberOfStepsPerRenderUpdate"]):
            solver.step()
            step_count += 1
            
            if export_ply_enabled and step_count % export_interval == 0:
                if frame_count < max_frames:
                    export_ply(ps, frame_count, ply_output_dir)
                    print(f"Exported PLY frame {frame_count}")
                    frame_count += 1
            
        ps.copy_to_vis_buffer()

        if show_window:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)
        scene.lines(box_anchors, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
        canvas.scene(scene)

        if show_window:
            window.show()

        # Export after rendering
        if export_images_enabled and step_count % image_interval == 0:
            if image_frame_count < max_frames:
                window.save_image(os.path.join(image_output_dir, f"frame_{image_frame_count:04d}.png"))
                print(f"Exported image frame {image_frame_count}")
                image_frame_count += 1
        
        if (export_ply_enabled and frame_count >= max_frames) or (export_images_enabled and image_frame_count >= max_frames):
            break

    print(f"Simulation Finished")
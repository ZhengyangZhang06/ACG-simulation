import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from WCSPH import WCSPHSolver

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

config = {
    "domainStart": [0.0, 0.0, 0.0],
    "domainEnd": [5.0, 3.0, 2.0],
    "particleRadius": 0.01,
    "numberOfStepsPerRenderUpdate": 1,
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

if __name__ == "__main__":
    ps = ParticleSystem(config)
    solver = WCSPHSolver(ps)

    window = ti.ui.Window('SPH', (1024, 1024), show_window=True, vsync=False)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5.5, 2.5, 4.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-1.0, 0.0, 0.0)
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

    while window.running:
        for i in range(config["numberOfStepsPerRenderUpdate"]):
            solver.step()
        ps.copy_to_vis_buffer()

        camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)
        scene.lines(box_anchors, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
        canvas.scene(scene)

        window.show()
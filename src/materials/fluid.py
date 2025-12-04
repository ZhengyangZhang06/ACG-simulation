import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# SPH Parameters
num_particles = 8000
dim = 3
screen_res = (800, 800)
boundary = [1.0, 1.0, 1.0]

# Physical parameters
particle_radius = 0.01
support_radius = particle_radius * 4.0
particle_diameter = 2.0 * particle_radius
h = support_radius
m_V = 0.8 * particle_diameter ** dim
rho0 = 1000.0
mass = m_V * rho0
stiffness = 50.0  # Pressure stiffness
gamma = 7.0
viscosity = 0.05
gravity = ti.Vector([0.0, -9.8, 0.0])
dt = 0.0001
boundary_epsilon = 1e-5

# Grid parameters for neighbor search
grid_size = support_radius
grid_num = [int(boundary[i] / grid_size) + 1 for i in range(dim)]
max_neighbors = 100
max_particles_per_cell = 100

# Particle data
positions = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles)
velocities = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles)
accelerations = ti.Vector.field(dim, dtype=ti.f32, shape=num_particles)
densities = ti.field(dtype=ti.f32, shape=num_particles)
pressures = ti.field(dtype=ti.f32, shape=num_particles)

# Grid data for neighbor search
grid_particle_count = ti.field(dtype=ti.i32, shape=tuple(grid_num))
grid_particles = ti.field(dtype=ti.i32, shape=(grid_num[0], grid_num[1], grid_num[2], max_particles_per_cell))


@ti.func
def poly6_kernel(r, h):
    """Poly6 kernel for density computation"""
    result = 0.0
    r_len = r.norm()
    if 0 <= r_len <= h:
        coeff = 315.0 / (64.0 * np.pi * h ** 9)
        result = coeff * (h * h - r_len * r_len) ** 3
    return result


@ti.func
def spiky_gradient(r, h):
    """Spiky kernel gradient for pressure force"""
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len <= h:
        coeff = -45.0 / (np.pi * h ** 6)
        result = coeff * (h - r_len) ** 2 * r / r_len
    return result


@ti.func
def viscosity_laplacian(r, h):
    """Viscosity kernel laplacian"""
    result = 0.0
    r_len = r.norm()
    if 0 < r_len <= h:
        coeff = 45.0 / (np.pi * h ** 6)
        result = coeff * (h - r_len)
    return result


@ti.func
def get_cell(pos):
    """Get grid cell index for a position"""
    return ti.Vector([
        int(pos[0] / grid_size),
        int(pos[1] / grid_size),
        int(pos[2] / grid_size)
    ])


@ti.kernel
def initialize_particles():
    """Initialize particles in a cube region"""
    spacing = particle_diameter
    num_per_dim = int(ti.pow(num_particles, 1.0 / 3.0)) + 1
    start_pos = ti.Vector([0.1, 0.4, 0.1])
    
    for i in range(num_particles):
        ix = i % num_per_dim
        iy = (i // num_per_dim) % num_per_dim
        iz = i // (num_per_dim * num_per_dim)
        
        positions[i] = start_pos + ti.Vector([ix * spacing, iy * spacing, iz * spacing])
        velocities[i] = ti.Vector([0.0, 0.0, 0.0])
        densities[i] = rho0
        pressures[i] = 0.0


@ti.kernel
def build_grid():
    """Build spatial hash grid for neighbor search"""
    # Clear grid
    for i, j, k in grid_particle_count:
        grid_particle_count[i, j, k] = 0
    
    # Insert particles into grid
    for i in range(num_particles):
        cell = get_cell(positions[i])
        if 0 <= cell[0] < grid_num[0] and 0 <= cell[1] < grid_num[1] and 0 <= cell[2] < grid_num[2]:
            idx = ti.atomic_add(grid_particle_count[cell[0], cell[1], cell[2]], 1)
            if idx < max_particles_per_cell:
                grid_particles[cell[0], cell[1], cell[2], idx] = i


@ti.kernel
def compute_density():
    """Compute density for each particle"""
    for i in range(num_particles):
        pos_i = positions[i]
        cell = get_cell(pos_i)
        density = 0.0
        
        # Search neighboring cells
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    ni = cell[0] + di
                    nj = cell[1] + dj
                    nk = cell[2] + dk
                    
                    if 0 <= ni < grid_num[0] and 0 <= nj < grid_num[1] and 0 <= nk < grid_num[2]:
                        count = grid_particle_count[ni, nj, nk]
                        for idx in range(count):
                            j = grid_particles[ni, nj, nk, idx]
                            r = pos_i - positions[j]
                            density += mass * poly6_kernel(r, h)
        
        densities[i] = ti.max(density, rho0)
        # Tait equation of state
        pressures[i] = stiffness * (ti.pow(densities[i] / rho0, gamma) - 1.0)


@ti.kernel
def compute_forces():
    """Compute pressure and viscosity forces"""
    for i in range(num_particles):
        pos_i = positions[i]
        vel_i = velocities[i]
        cell = get_cell(pos_i)
        
        pressure_force = ti.Vector([0.0, 0.0, 0.0])
        viscosity_force = ti.Vector([0.0, 0.0, 0.0])
        
        # Search neighboring cells
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    ni = cell[0] + di
                    nj = cell[1] + dj
                    nk = cell[2] + dk
                    
                    if 0 <= ni < grid_num[0] and 0 <= nj < grid_num[1] and 0 <= nk < grid_num[2]:
                        count = grid_particle_count[ni, nj, nk]
                        for idx in range(count):
                            j = grid_particles[ni, nj, nk, idx]
                            if i != j:
                                r = pos_i - positions[j]
                                r_len = r.norm()
                                
                                if r_len < h:
                                    # Pressure force
                                    pressure_term = -mass * (pressures[i] + pressures[j]) / (2.0 * densities[j])
                                    pressure_force += pressure_term * spiky_gradient(r, h)
                                    
                                    # Viscosity force
                                    viscosity_force += viscosity * mass * (velocities[j] - vel_i) / densities[j] * viscosity_laplacian(r, h)
        
        # Total acceleration
        accelerations[i] = (pressure_force + viscosity_force) / densities[i] + gravity


@ti.kernel
def update_particles():
    """Update particle positions and velocities with boundary handling"""
    for i in range(num_particles):
        # Semi-implicit Euler integration
        velocities[i] += accelerations[i] * dt
        positions[i] += velocities[i] * dt
        
        # Boundary collision handling
        c_f = 0.3
        for d in ti.static(range(dim)):
            # Left boundary (d dimension)
            if positions[i][d] < particle_radius:
                vec = ti.Vector([1.0 if k == d else 0.0 for k in range(dim)])
                dist = particle_radius - positions[i][d]
                positions[i] += vec * dist
                velocities[i] -= (1.0 + c_f) * velocities[i].dot(vec) * vec
            # Right boundary (d dimension)
            if positions[i][d] > boundary[d] - particle_radius:
                vec = ti.Vector([-1.0 if k == d else 0.0 for k in range(dim)])
                dist = positions[i][d] - (boundary[d] - particle_radius)
                positions[i] += vec * dist
                velocities[i] -= (1.0 + c_f) * velocities[i].dot(vec) * vec


def step():
    """Perform one simulation step"""
    build_grid()
    compute_density()
    compute_forces()
    update_particles()


def main():
    """Main function to run the simulation with visualization"""
    initialize_particles()
    
    # Create a window for 3D visualization
    window = ti.ui.Window("SPH Fluid Simulation", screen_res, vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    camera.position(1.5, 1.5, 2.5)
    camera.lookat(0.5, 0.3, 0.5)
    camera.up(0.0, 1.0, 0.0)
    
    while window.running:
        # Perform multiple substeps per frame for stability
        for _ in range(5):
            step()
        
        # Update camera
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        # Lighting
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(1.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        
        # Draw particles
        scene.particles(positions, radius=particle_radius, color=(0.3, 0.5, 0.9))
        
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()


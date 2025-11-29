import taichi as ti
import numpy as np
from functools import reduce

@ti.data_oriented
class ParticleSystem:
    def __init__(self, config):
        self.cfg = config

        self.domain_start = np.array(self.cfg["domainStart"])
        self.domain_end = np.array(self.cfg["domainEnd"])
        self.domain_size = self.domain_end - self.domain_start
        self.dim = len(self.domain_size)
        assert self.dim > 1

        self.material_fluid = 1
        self.material_boundary = 0

        self.particle_radius = self.cfg["particleRadius"]
        self.particle_diameter = 2 * self.particle_radius
        self.h = self.particle_radius * 4.0
        self.m_V = 0.8 * self.particle_diameter ** self.dim
        self.grid_size = self.h
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        self.particle_max_num_per_cell = 100
        self.particle_max_num_neighbor = 100

        fluid_blocks = self.cfg["FluidBlocks"]
        fluid_particle_num = 0
        for fluid in fluid_blocks:
            start = np.array(fluid["start"]) + np.array(fluid["translation"])
            end = np.array(fluid["end"]) + np.array(fluid["translation"])
            particle_num = self.compute_cube_particle_num(start, end)
            fluid_particle_num += particle_num

        self.particle_max_num = fluid_particle_num
        self.particle_num = ti.field(int, shape=())

        self.x = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)

        self.particle_neighbors = ti.field(int, shape=(self.particle_max_num, self.particle_max_num_neighbor))
        self.particle_neighbors_num = ti.field(int, shape=self.particle_max_num)

        self.grid_particles_num = ti.field(int)
        self.grid_particles = ti.field(int)
        if self.dim == 2:
            grid_node = ti.root.dense(ti.ij, self.grid_num)
            grid_node.place(self.grid_particles_num)
            cell_node = grid_node.dense(ti.k, self.particle_max_num_per_cell)
            cell_node.place(self.grid_particles)
        else:
            grid_node = ti.root.dense(ti.ijk, self.grid_num)
            grid_node.place(self.grid_particles_num)
            cell_node = grid_node.dense(ti.l, self.particle_max_num_per_cell)
            cell_node.place(self.grid_particles)

        self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)

        self.initialize_particles()

    def initialize_particles(self):
        offset = 0
        fluid_blocks = self.cfg["FluidBlocks"]
        for fluid in fluid_blocks:
            start = np.array(fluid["start"]) + np.array(fluid["translation"])
            end = np.array(fluid["end"]) + np.array(fluid["translation"])
            velocity = np.array(fluid["velocity"])
            density = fluid["density"]
            color = fluid["color"]
            self.add_cube(offset, start, end, velocity, density, color)
            particle_num = self.compute_cube_particle_num(start, end)
            offset += particle_num
        self.particle_num[None] = offset

    def compute_cube_particle_num(self, start, end):
        num_dim = [np.arange(start[i], end[i], self.particle_diameter) for i in range(self.dim)]
        return reduce(lambda x, y: x * y, [len(n) for n in num_dim])

    def add_cube(self, offset, start, end, velocity, density, color):
        positions = []
        if self.dim == 2:
            x_coords = np.arange(start[0], end[0], self.particle_diameter)
            y_coords = np.arange(start[1], end[1], self.particle_diameter)
            X, Y = np.meshgrid(x_coords, y_coords)
            positions = np.column_stack((X.ravel(), Y.ravel()))
        elif self.dim == 3:
            x_coords = np.arange(start[0], end[0], self.particle_diameter)
            y_coords = np.arange(start[1], end[1], self.particle_diameter)
            z_coords = np.arange(start[2], end[2], self.particle_diameter)
            X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords)
            positions = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        num_particles = len(positions)
        for i in range(num_particles):
            p = offset + i
            self.x[p] = positions[i]
            self.v[p] = velocity
            self.density[p] = density
            self.pressure[p] = 0.0
            self.material[p] = 1
            self.color[p] = ti.Vector([color[0]/255.0, color[1]/255.0, color[2]/255.0])

    @ti.func
    def pos_to_index(self, pos):
        return ((pos - ti.Vector(self.domain_start)) / self.grid_size).cast(int)

    @ti.func
    def is_valid_cell(self, cell):
        is_valid = True
        for d in ti.static(range(self.dim)):
            is_valid = is_valid and (0 <= cell[d] < self.grid_num[d])
        return is_valid

    @ti.kernel
    def allocate_particles_to_grid(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for p in range(self.particle_num[None]):
            cell = self.pos_to_index(self.x[p])
            if self.is_valid_cell(cell):
                offset = ti.atomic_add(self.grid_particles_num[cell], 1)
                if offset < self.particle_max_num_per_cell:
                    self.grid_particles[cell, offset] = p

    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_num[None]):
            self.particle_neighbors_num[p_i] = 0
            center_cell = self.pos_to_index(self.x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                cell = center_cell + offset
                if self.is_valid_cell(cell):
                    for j in range(self.grid_particles_num[cell]):
                        p_j = self.grid_particles[cell, j]
                        if p_i != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.h:
                            nb_idx = ti.atomic_add(self.particle_neighbors_num[p_i], 1)
                            if nb_idx < self.particle_max_num_neighbor:
                                self.particle_neighbors[p_i, nb_idx] = p_j

    def initialize_particle_system(self):
        self.allocate_particles_to_grid()
        self.search_neighbors()

    @ti.kernel
    def copy_to_vis_buffer_kernel(self):
        for p in range(self.particle_num[None]):
            self.x_vis_buffer[p] = self.x[p]
            self.color_vis_buffer[p] = self.color[p]

    def copy_to_vis_buffer(self):
        self.copy_to_vis_buffer_kernel()

    def dump(self):
        positions = self.x.to_numpy()[:self.particle_num[None]]
        return {"position": positions}

    
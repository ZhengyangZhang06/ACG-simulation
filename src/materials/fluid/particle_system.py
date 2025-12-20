import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce

@ti.data_oriented
class ParticleSystem:
    def __init__(self, config):
        self.cfg = config

        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))
        self.domain_end = np.array(self.cfg.get_cfg("domainEnd"))
        self.domain_size = self.domain_end - self.domain_start
        self.dim = len(self.domain_size)

        self.material_fluid = 1
        self.material_solid = 0

        self.particle_radius = self.cfg.get_cfg("particleRadius")
        self.particle_diameter = 2 * self.particle_radius
        self.h = self.particle_radius * 4.0
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim
        self.grid_size = self.h
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        self.particle_max_num_per_cell = 100
        self.particle_max_num_neighbor = 100

        self.object_collection = dict()
        self.object_id_rigid_body = set()

        #========== Compute number of particles ==========#
        fluid_particle_num = 0
        rigid_particle_num = 0

        # Process Fluid Blocks
        fluid_blocks = self.cfg.get_fluid_blocks()
        for fluid in fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num
        
        # Process Fluid Bodies (mesh-based)
        fluid_bodies = self.cfg.get_fluid_bodies()
        for fluid_body in fluid_bodies:
            voxelized_points_np = self.load_body(fluid_body)
            fluid_body["particleNum"] = voxelized_points_np.shape[0]
            fluid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[fluid_body["objectId"]] = fluid_body
            fluid_particle_num += voxelized_points_np.shape[0]
        
        # Process Rigid Bodies (mesh-based)
        rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in rigid_bodies:
            voxelized_points_np = self.load_body(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            self.object_id_rigid_body.add(rigid_body["objectId"])
            rigid_particle_num += voxelized_points_np.shape[0]

        self.particle_max_num = fluid_particle_num + rigid_particle_num
        self.particle_num = ti.field(int, shape=())

        # Particle fields
        self.x = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)
        self.object_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        
        # Rigid body fields
        self.padding = self.particle_radius
        num_rigid_bodies = len(rigid_bodies) + 10
        self.rigid_rest_cm = ti.Vector.field(self.dim, dtype=float, shape=num_rigid_bodies)
        
        # Boundary volume for fluid-solid coupling
        self.boundary_volume = ti.field(dtype=float, shape=self.particle_max_num)

        # Neighbor search
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

        # Visualization buffers
        self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)

        #========== Initialize particles ==========#
        # Fluid blocks
        for fluid in fluid_blocks:
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            self.add_cube(
                start=start,
                end=start + (end - start) * scale,
                velocity=fluid["velocity"],
                density=fluid["density"],
                color=fluid["color"],
                material=self.material_fluid,
                is_dynamic=1,
                object_id=fluid["objectId"]
            )

        # Fluid bodies (mesh-based)
        for fluid_body in fluid_bodies:
            self.add_particles(
                positions=fluid_body["voxelizedPoints"],
                velocity=fluid_body.get("velocity", [0.0, 0.0, 0.0]),
                density=fluid_body["density"],
                color=fluid_body["color"],
                material=self.material_fluid,
                is_dynamic=1,
                object_id=fluid_body["objectId"]
            )

        # Rigid bodies
        for rigid_body in rigid_bodies:
            is_dynamic = rigid_body.get("isDynamic", False)
            velocity = rigid_body.get("velocity", [0.0, 0.0, 0.0]) if is_dynamic else [0.0, 0.0, 0.0]
            self.add_particles(
                positions=rigid_body["voxelizedPoints"],
                velocity=velocity,
                density=rigid_body["density"],
                color=rigid_body["color"],
                material=self.material_solid,
                is_dynamic=int(is_dynamic),
                object_id=rigid_body["objectId"]
            )

    def load_body(self, body):
        mesh = tm.load(body["geometryFile"])
        
        # Center the mesh at origin before scaling
        mesh.vertices -= mesh.centroid
        
        mesh.apply_scale(body.get("scale", 1.0))
        
        angle = body.get("rotationAngle", 0) / 360 * 2 * np.pi
        direction = body.get("rotationAxis", [0, 1, 0])
        if angle != 0:
            rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
            mesh.apply_transform(rot_matrix)
        
        offset = np.array(body.get("translation", [0, 0, 0]))
        mesh.vertices += offset
        
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        body["mesh"] = mesh_backup
        body["restPosition"] = mesh_backup.vertices.copy()
        body["restCenterOfMass"] = mesh_backup.vertices.mean(axis=0)
        
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        return voxelized_mesh.points

    def compute_cube_particle_num(self, start, end):
        num_dim = [np.arange(start[i], end[i], self.particle_diameter) for i in range(self.dim)]
        return reduce(lambda x, y: x * y, [len(n) for n in num_dim])

    def add_cube(self, start, end, velocity, density, color, material, is_dynamic, object_id=0):
        if self.dim == 2:
            x_coords = np.arange(start[0], end[0], self.particle_diameter)
            y_coords = np.arange(start[1], end[1], self.particle_diameter)
            X, Y = np.meshgrid(x_coords, y_coords)
            positions = np.column_stack((X.ravel(), Y.ravel()))
        else:
            x_coords = np.arange(start[0], end[0], self.particle_diameter)
            y_coords = np.arange(start[1], end[1], self.particle_diameter)
            z_coords = np.arange(start[2], end[2], self.particle_diameter)
            X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords)
            positions = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        
        self.add_particles(positions, velocity, density, color, material, is_dynamic, object_id)

    def add_particles(self, positions, velocity, density, color, material, is_dynamic, object_id=0):
        num_new = len(positions)
        offset = self.particle_num[None]
        
        for i in range(num_new):
            p = offset + i
            self.x[p] = positions[i]
            self.x_0[p] = positions[i]
            self.v[p] = velocity
            self.acceleration[p] = ti.Vector([0.0 for _ in range(self.dim)])
            self.density[p] = density
            self.pressure[p] = 0.0
            self.m_V[p] = self.m_V0
            self.m[p] = self.m_V0 * density
            self.material[p] = material
            self.is_dynamic[p] = is_dynamic
            self.object_id[p] = object_id
            self.color[p] = ti.Vector([color[0]/255.0, color[1]/255.0, color[2]/255.0])
        
        self.particle_num[None] += num_new

    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_solid and (self.is_dynamic[p] == 0)

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and (self.is_dynamic[p] == 1)

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

    
import taichi as ti
import numpy as np

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system, config):
        self.ps = particle_system
        self.config = config
        self.g = config.get_cfg("gravity", -9.81)
        self.viscosity = config.get_cfg("viscosity", 0.01)
        self.density_0 = config.get_cfg("density0", 1000.0)
        self.surface_tension = config.get_cfg("surfaceTension", 0.01)
        self.dt = ti.field(float, shape=())
        self.dt[None] = config.get_cfg("dt", 0.0001)
        self.h = self.ps.h

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.h
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.h
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res


    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                delta = self.cubic_kernel(0.0)
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    if self.ps.material[p_j] == self.ps.material_solid:
                        x_ij = self.ps.x[p_i] - self.ps.x[p_j]
                        delta += self.cubic_kernel(x_ij.norm())
                self.ps.V[p_i] = 1.0 / delta

    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_solid:
                    x_ij = self.ps.x[p_i] - self.ps.x[p_j]
                    delta += self.cubic_kernel(x_ij.norm())
            self.ps.V[p_i] = 1.0 / delta

    @ti.kernel
    def compute_rigid_rest_cm_kernel(self, object_id: int):
        sum_m = 0.0
        cm = ti.Vector([0.0 for _ in range(self.ps.dim)])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_solid and self.ps.object_id[p_i] == object_id:
                mass = self.ps.V0 * self.ps.density[p_i]
                cm += mass * self.ps.x_0[p_i]
                sum_m += mass
        if sum_m > 1e-6:
            self.ps.rigid_rest_cm[object_id] = cm / sum_m

    def compute_rigid_rest_cm(self, object_id):
        self.compute_rigid_rest_cm_kernel(object_id)

    def initialize(self):
        self.ps.initialize_particle_system()
        for r_obj_id in self.ps.object_id_rigid_body:
            self.compute_rigid_rest_cm(r_obj_id)
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        c_f = 0.5
        self.ps.v[p_i] -= (1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([1.0, 0.0]))
                if pos[0] < self.ps.padding:
                    self.ps.x[p_i][0] = self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0]))

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([0.0, 1.0]))
                if pos[1] < self.ps.padding:
                    self.ps.x[p_i][1] = self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([0.0, -1.0]))

    @ti.kernel
    def enforce_boundary_3D(self, particle_type: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([1.0, 0.0, 0.0]))
                if pos[0] < self.ps.padding:
                    self.ps.x[p_i][0] = self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0, 0.0]))

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([0.0, 1.0, 0.0]))
                if pos[1] < self.ps.padding:
                    self.ps.x[p_i][1] = self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([0.0, -1.0, 0.0]))

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    self.ps.x[p_i][2] = self.ps.domain_size[2] - self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([0.0, 0.0, 1.0]))
                if pos[2] < self.ps.padding:
                    self.ps.x[p_i][2] = self.ps.padding
                    self.simulate_collisions(p_i, ti.Vector([0.0, 0.0, -1.0]))

    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0 for _ in range(self.ps.dim)])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                mass = self.ps.V0 * self.ps.density[p_i]
                cm += mass * self.ps.x[p_i]
                sum_m += mass
        if sum_m > 1e-6:
            cm /= sum_m
        return cm

    @ti.kernel
    def compute_com_kernel(self, object_id: int) -> ti.types.vector(3, float):
        return self.compute_com(object_id)

    @ti.kernel
    def solve_constraints(self, object_id: int) -> ti.types.matrix(3, 3, float):
        cm = self.compute_com(object_id)
        A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                q = self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id]
                p = self.ps.x[p_i] - cm
                A += self.ps.V0 * self.ps.density[p_i] * p.outer_product(q)

        R, S = ti.polar_decompose(A)

        if all(abs(R) < 1e-6):
            R = ti.Matrix.identity(ti.f32, 3)

        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                goal = cm + R @ (self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id])
                corr = (goal - self.ps.x[p_i]) * 1.0
                self.ps.x[p_i] += corr
        return R

    def solve_rigid_body(self):
        for i in range(1):
            for r_obj_id in self.ps.object_id_rigid_body:
                if self.ps.object_collection[r_obj_id]["isDynamic"]:
                    R = self.solve_constraints(r_obj_id)

                    if self.ps.cfg.get_cfg("exportObj"):
                        # For output obj only: update the mesh
                        cm = self.compute_com_kernel(r_obj_id)
                        ret = R.to_numpy() @ (self.ps.object_collection[r_obj_id]["restPosition"] - self.ps.object_collection[r_obj_id]["restCenterOfMass"]).T
                        self.ps.object_collection[r_obj_id]["mesh"].vertices = cm.to_numpy() + ret.T

                    # self.compute_rigid_collision()
                    if self.ps.dim == 2:
                        self.enforce_boundary_2D(self.ps.material_solid)
                    else:
                        self.enforce_boundary_3D(self.ps.material_solid)

    def step(self):
        self.ps.initialize_particle_system()
        self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()
        if self.ps.dim == 2:
            self.enforce_boundary_2D(self.ps.material_fluid)
        else:
            self.enforce_boundary_3D(self.ps.material_fluid)
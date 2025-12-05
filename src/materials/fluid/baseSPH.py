import taichi as ti
import numpy as np

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = -9.81
        self.viscosity = 0.01
        self.density_0 = 1000.0
        self.surface_tension = 0.01
        self.mass = self.ps.m_V * self.density_0
        self.dt = ti.field(float, shape=())
        self.dt[None] = 0.0004
        self.h = self.ps.h

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.h
        k = 8.0 / np.pi
        k /= h ** 3
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
        k = 8.0 / np.pi
        k = 6.0 * k / h ** 3
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

    @ti.func
    def pressure_force(self, p_i, p_j, r_ij):
        p_i_pressure = self.ps.pressure[p_i]
        p_j_pressure = self.ps.pressure[p_j]
        rho_i = self.ps.density[p_i]
        rho_j = self.ps.density[p_j]
        force = -self.mass * (p_i_pressure / (rho_i * rho_i) + p_j_pressure / (rho_j * rho_j)) * self.cubic_kernel_derivative(r_ij)
        return force

    @ti.func
    def viscosity_force(self, p_i, p_j, r_ij):
        v = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r_ij)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / self.ps.density[p_j]) * v / (r_ij.norm()**2 + 0.01 * self.h**2) * self.cubic_kernel_derivative(r_ij)
        return res

    @ti.func
    def surface_tension_force(self, p_i, p_j, r_ij):
        r_norm = r_ij.norm()
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > self.ps.particle_diameter:
            res -= self.surface_tension * self.ps.density[p_j] / self.ps.density[p_i] * self.cubic_kernel(r_norm) * r_ij
        else:
            res -= self.surface_tension * self.ps.density[p_j] / self.ps.density[p_i] * self.cubic_kernel(self.ps.particle_diameter) * r_ij
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
                if delta > 1e-6:
                    self.ps.boundary_volume[p_i] = 1.0 / delta
                else:
                    self.ps.boundary_volume[p_i] = 0.0

    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i):
                delta = self.cubic_kernel(0.0)
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    if self.ps.material[p_j] == self.ps.material_solid:
                        x_ij = self.ps.x[p_i] - self.ps.x[p_j]
                        delta += self.cubic_kernel(x_ij.norm())
                if delta > 1e-6:
                    self.ps.boundary_volume[p_i] = 1.0 / delta
                else:
                    self.ps.boundary_volume[p_i] = 0.0

    @ti.kernel
    def compute_rigid_rest_cm_kernel(self, object_id: int):
        sum_m = 0.0
        cm = ti.Vector([0.0 for _ in range(self.ps.dim)])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_solid and self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[p_i]
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
    def simulate_collisions(self, p_i, vec, d):
        c_f = 0.5
        self.ps.x[p_i] += vec * d
        self.ps.v[p_i] -= (1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                d = 0.0
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    d = pos[0] - (self.ps.domain_size[0] - self.ps.padding)
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    d = self.ps.padding - pos[0]
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    d = pos[1] - (self.ps.domain_size[1] - self.ps.padding)
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    d = self.ps.padding - pos[1]
                    self.ps.x[p_i][1] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length, d)

    @ti.kernel
    def enforce_boundary_3D(self, particle_type: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                d = 0.0
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    d = pos[0] - (self.ps.domain_size[0] - self.ps.padding)
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    d = self.ps.padding - pos[0]
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    d = pos[1] - (self.ps.domain_size[1] - self.ps.padding)
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    d = self.ps.padding - pos[1]
                    self.ps.x[p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    d = pos[2] - (self.ps.domain_size[2] - self.ps.padding)
                    self.ps.x[p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    d = self.ps.padding - pos[2]
                    self.ps.x[p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length, d)

    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0 for _ in range(self.ps.dim)])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[p_i]
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
                A += self.ps.m_V0 * self.ps.density[p_i] * p.outer_product(q)

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
        for r_obj_id in self.ps.object_id_rigid_body:
            if self.ps.object_collection[r_obj_id].get("isDynamic", False):
                self.solve_constraints(r_obj_id)
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
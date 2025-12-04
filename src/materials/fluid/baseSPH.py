import taichi as ti
import numpy as np

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = -9.81
        self.viscosity = 0.05
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
        res = ti.Vector([0.0, 0.0, 0.0])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def viscosity_laplacian(self, r_norm):
        result = 0.0
        if 0 < r_norm <= self.h:
            coeff = 45.0 / (np.pi * self.h ** 6)
            result = coeff * (self.h - r_norm)
        return result

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
        res = ti.Vector([0.0, 0.0, 0.0])
        if r_norm > 1e-5:
            res = -self.surface_tension * self.mass * self.cubic_kernel(r_norm) * r_ij / r_norm
        return res

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        c_f = 0.3
        self.ps.x[p_i] += vec * d
        self.ps.v[p_i] -= (1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                pos = self.ps.x[p_i]
                if pos[0] < self.ps.domain_start[0] + self.ps.particle_radius:
                    self.simulate_collisions(p_i, ti.Vector([1.0, 0.0, 0.0]), self.ps.domain_start[0] + self.ps.particle_radius - pos[0])
                if pos[0] > self.ps.domain_end[0] - self.ps.particle_radius:
                    self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0, 0.0]), pos[0] - (self.ps.domain_end[0] - self.ps.particle_radius))
                if pos[1] < self.ps.domain_start[1] + self.ps.particle_radius:
                    self.simulate_collisions(p_i, ti.Vector([0.0, 1.0, 0.0]), self.ps.domain_start[1] + self.ps.particle_radius - pos[1])
                if pos[1] > self.ps.domain_end[1] - self.ps.particle_radius:
                    self.simulate_collisions(p_i, ti.Vector([0.0, -1.0, 0.0]), pos[1] - (self.ps.domain_end[1] - self.ps.particle_radius))
                if ti.static(self.ps.dim == 3):
                    if pos[2] < self.ps.domain_start[2] + self.ps.particle_radius:
                        self.simulate_collisions(p_i, ti.Vector([0.0, 0.0, 1.0]), self.ps.domain_start[2] + self.ps.particle_radius - pos[2])
                    if pos[2] > self.ps.domain_end[2] - self.ps.particle_radius:
                        self.simulate_collisions(p_i, ti.Vector([0.0, 0.0, -1.0]), pos[2] - (self.ps.domain_end[2] - self.ps.particle_radius))

    def step(self):
        self.ps.initialize_particle_system()
        self.substep()
        self.enforce_boundary()
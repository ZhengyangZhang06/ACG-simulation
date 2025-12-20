import taichi as ti
from baseSPH import SPHBase

class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.exponent = 7.0
        self.stiffness = 50000.0

    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
            self.ps.density[p_i] *= self.density_0

    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
            self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i):
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                r_ij = x_i - x_j
                dpi = self.ps.pressure[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])
                dpj = self.ps.pressure[p_j] / (self.ps.density[p_j] * self.ps.density[p_j])
                f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) * self.cubic_kernel_derivative(r_ij)
                d_v += f_p
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]
            self.ps.acceleration[p_i] += d_v

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_v[1] = self.g
            self.ps.acceleration[p_i] = d_v
            if self.ps.material[p_i] == self.ps.material_fluid:
                x_i = self.ps.x[p_i]
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]
                    r_ij = x_i - x_j
                    v_ij = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r_ij)
                    f_v = 2 * (self.ps.dim + 2) * self.viscosity * (self.ps.m[p_j] / self.ps.density[p_j]) * v_ij / (r_ij.norm()**2 + 0.01 * self.h**2) * self.cubic_kernel_derivative(r_ij)
                    d_v += f_v
                    if self.ps.is_dynamic_rigid_body(p_j):
                        self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]
                self.ps.acceleration[p_i] = d_v

    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()
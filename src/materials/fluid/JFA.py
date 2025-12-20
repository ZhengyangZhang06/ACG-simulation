import taichi as ti
import numpy as np

@ti.data_oriented
class JumpFloodAlgorithm:
    def __init__(self, width, height, threshold):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.vector_field = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))  # xy: pos, z: dist, w: valid

    @ti.kernel
    def init(self, img: ti.types.ndarray()):
        for i, j in ti.ndrange(self.width, self.height):
            val = img[i, j]
            if val > self.threshold:
                self.vector_field[i, j] = ti.Vector([i, j, 0.0, 1.0])
            else:
                self.vector_field[i, j] = ti.Vector([0, 0, 1e10, 0.0])

    @ti.kernel
    def compute_pass(self, jump_size: ti.i32):
        for i, j in ti.ndrange(self.width, self.height):
            best_dist = self.vector_field[i, j][2]
            best_pos = self.vector_field[i, j][:2]
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    if not (di == 0 and dj == 0):
                        ni = ti.max(ti.min(i + di * jump_size, self.width - 1), 0)
                        nj = ti.max(ti.min(j + dj * jump_size, self.height - 1), 0)
                        n_pos = self.vector_field[ni, nj][:2]
                        if self.vector_field[ni, nj][3] > 0:
                            dist = (ti.Vector([i, j]) - n_pos).norm()
                            if dist < best_dist:
                                best_dist = dist
                                best_pos = n_pos
            self.vector_field[i, j] = ti.Vector([best_pos[0], best_pos[1], best_dist, self.vector_field[i, j][3]])

    def run(self, img_arr):
        self.init(img_arr)
        jump_size = max(self.width, self.height) // 2
        while jump_size > 0:
            self.compute_pass(jump_size)
            jump_size //= 2
import taichi as ti
import numpy as np

@ti.data_oriented
class JumpFloodAlgorithm:
    def __init__(self, width, height, threshold):
        self.width = width
        self.height = height
        self.threshold = threshold
        # Double buffering to avoid read-write race conditions
        self.vector_field = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))  # xy: pos, z: dist, w: valid
        self.temp_field = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))

    @ti.kernel
    def init(self, img: ti.types.ndarray()):
        for i, j in ti.ndrange(self.width, self.height):
            val = img[i, j]
            if val > self.threshold:
                self.vector_field[i, j] = ti.Vector([i, j, 0.0, 1.0])
            else:
                self.vector_field[i, j] = ti.Vector([0, 0, 1e10, 0.0])

    @ti.kernel
    def compute_pass(self, jump_size: ti.i32, source: ti.template(), target: ti.template()):
        for i, j in ti.ndrange(self.width, self.height):
            best_dist = source[i, j][2]
            best_pos = source[i, j][:2]
            
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    if not (di == 0 and dj == 0):
                        ni = ti.max(ti.min(i + di * jump_size, self.width - 1), 0)
                        nj = ti.max(ti.min(j + dj * jump_size, self.height - 1), 0)
                        
                        if source[ni, nj][3] > 0:  # valid seed
                            n_pos = source[ni, nj][:2]
                            dist = (ti.Vector([i, j]) - n_pos).norm()
                            if dist < best_dist:
                                best_dist = dist
                                best_pos = n_pos
            
            target[i, j] = ti.Vector([best_pos[0], best_pos[1], best_dist, source[i, j][3]])

    def run(self, img_arr):
        self.init(img_arr)
        # Improved jump size: round up to power of 2 to ensure coverage of entire image
        max_dim = max(self.width, self.height)
        jump_size = 1 << ((max_dim - 1).bit_length() - 1)
        
        current_source = self.vector_field
        current_target = self.temp_field
        
        while jump_size > 0:
            self.compute_pass(jump_size, current_source, current_target)
            # Swap buffers
            current_source, current_target = current_target, current_source
            jump_size //= 2
        
        # Ensure final result is in vector_field
        if current_source is self.temp_field:
            self.vector_field.copy_from(self.temp_field)
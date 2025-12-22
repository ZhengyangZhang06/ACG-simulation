import taichi as ti
import numpy as np
import os
from baseSPH import SPHBase
from JFA import JumpFloodAlgorithm

class WCSPHSolver(SPHBase):
    def __init__(self, particle_system, config):
        super().__init__(particle_system)
        self.exponent = 7.0
        self.stiffness = 50000.0
        self.config = config
        self.is_bad_apple = config.get_cfg("isBadApple", False)
        
        # Mouse interaction parameters
        self.mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.mouse_pos[None] = [0.0, 0.0]
        self.mouse_strength = ti.field(dtype=ti.f32, shape=())
        self.mouse_strength[None] = 0.0
        self.mouse_radius = config.get_cfg("mouseInteractionRadius", 0.5)
        
        # Initialize bad apple fields
        self.bad_apple_enabled = ti.field(dtype=ti.i32, shape=())
        self.bad_apple_enabled[None] = 1 if self.is_bad_apple else 0
        
        # Current frame as Taichi field so it can be accessed in kernels
        self.current_frame_field = ti.field(dtype=ti.i32, shape=())
        self.current_frame_field[None] = 0
        
        # Frame update counter for controlling video playback speed
        self.frame_update_counter = 0
        self.frame_update_interval = 1  # Update every N render frames (1 = every frame, 2 = every other frame)
        
        if self.is_bad_apple:
            import PIL.Image as Image
            self.bad_apple_size = config.get_cfg("badAppleSize")
            self.num_frames = config.get_cfg("numFrames")
            self.apple_weight = config.get_cfg("appleWeight")
            self.apple_displace_weight = config.get_cfg("appleDisplaceWeight")
            self.apple_test_params = config.get_cfg("appleTestParams")
            self.apple_spread_strength = config.get_cfg("appleSpreadStrength", 2.0)  # Force to spread particles in white regions
            self.wall_force_dst = config.get_cfg("wallForceDst", 0.5)
            self.wall_force_str = config.get_cfg("wallForceStr", 50.0)
            frames_path = config.get_cfg("badAppleFramesPath")
            
            available_frames = []
            for filename in sorted(os.listdir(frames_path)):
                if filename.endswith('.png'):
                    frame_num = int(filename.split('.')[0])
                    available_frames.append(frame_num)
            
            if not available_frames:
                raise FileNotFoundError(f"No PNG frames found in {frames_path}")
            
            available_frames.sort()
            start_frame = available_frames[0]
            num_available = len(available_frames)
            self.num_frames = min(config.get_cfg("numFrames"), num_available)
            
            print(f"Loading {self.num_frames} frames starting from frame {start_frame}")
            
            frames = []
            for i in range(self.num_frames):
                frame_num = available_frames[i]
                frame_path = os.path.join(frames_path, f"{frame_num:04d}.png")
                frame_img = Image.open(frame_path).convert('L')
                frame = np.array(frame_img) / 255.0
                # Flip vertically to match physics coordinate system (Y up)
                frame = np.flipud(frame)
                frames.append(frame.T)
            self.frames = np.array(frames)
            self.jfa_results = []
            jfa = JumpFloodAlgorithm(self.bad_apple_size[0], self.bad_apple_size[1], 0.5)
            for frame in self.frames:
                jfa.run(frame)
                self.jfa_results.append(jfa.vector_field.to_numpy())
            self.jfa_results = np.array(self.jfa_results)
            self.bad_apple_tex = ti.Vector.field(4, dtype=ti.f32, shape=(self.num_frames, self.bad_apple_size[0], self.bad_apple_size[1]))
            self.bad_apple_tex.from_numpy(self.jfa_results)
            
            # Set frame update interval based on config (default 2 for 30fps video at 60fps render)
            self.frame_update_interval = config.get_cfg("badAppleFrameInterval", 2)
            self.loop_video = config.get_cfg("badAppleLoopVideo", True)
        else:
            self.bad_apple_size = [1, 1]
            self.num_frames = 1
            self.apple_weight = 0.0
            self.apple_displace_weight = 0.0
            self.apple_spread_strength = 0.0
            self.apple_test_params = [1.0, 1.0, 0.0, 0.1]
            self.wall_force_dst = 0.5
            self.wall_force_str = 0.0
            self.bad_apple_tex = ti.Vector.field(4, dtype=ti.f32, shape=(1, 1, 1))

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
                if self.ps.material[p_j] == self.ps.material_fluid or self.ps.material[p_j] == self.ps.material_solid:
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
                dpj = self.ps.pressure[p_i] / (self.density_0 * self.density_0)
                if self.ps.material[p_j] == self.ps.material_fluid:
                    dpj = self.ps.pressure[p_j] / (self.ps.density[p_j] * self.ps.density[p_j])
                f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) * self.cubic_kernel_derivative(r_ij)
                d_v += f_p
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]
            
            if self.bad_apple_enabled[None] == 1 and self.ps.material[p_i] == self.ps.material_fluid:
                # HLSL: float dstWallX = boundsSize.x / 2 - abs(pos.x);
                bounds_size = ti.Vector(self.ps.domain_end[:2]) - ti.Vector(self.ps.domain_start[:2])
                domain_center = (ti.Vector(self.ps.domain_end[:2]) + ti.Vector(self.ps.domain_start[:2])) / 2
                pos_centered = x_i[:2] - domain_center
                dst_wall_x = bounds_size[0] / 2 - ti.abs(pos_centered[0])
                dst_wall_y = bounds_size[1] / 2 - ti.abs(pos_centered[1])
                
                # HLSL: float2 wallAccel = saturate(1 - float2(dstWallX, dstWallY) / wallForceDst) * wallForceStr * density * -sign(pos);
                wall_factor_x = ti.max(0.0, ti.min(1.0, 1.0 - dst_wall_x / self.wall_force_dst))
                wall_factor_y = ti.max(0.0, ti.min(1.0, 1.0 - dst_wall_y / self.wall_force_dst))
                wall_accel = ti.Vector([
                    wall_factor_x * self.wall_force_str * self.ps.density[p_i] * (-ti.math.sign(pos_centered[0])),
                    wall_factor_y * self.wall_force_str * self.ps.density[p_i] * (-ti.math.sign(pos_centered[1]))
                ])
                d_v += wall_accel
                
                # HLSL: Velocities[id.x] *= appleTestParams.x;
                self.ps.v[p_i] *= self.apple_test_params[0]
            
            self.ps.acceleration[p_i] += d_v

    @ti.kernel
    def apply_external_forces(self):
        """Apply gravity and bad apple forces (before density calculation)"""
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                continue
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_v[1] = self.g
            
            # Mouse interaction force
            if self.ps.material[p_i] == self.ps.material_fluid and ti.abs(self.mouse_strength[None]) > 0.001:
                x_i = self.ps.x[p_i]
                mouse_offset = self.mouse_pos[None] - x_i[:2]
                sqr_dst = mouse_offset.dot(mouse_offset)
                radius_sq = self.mouse_radius * self.mouse_radius
                
                if sqr_dst < radius_sq:
                    dst = ti.sqrt(sqr_dst)
                    edge_t = dst / self.mouse_radius
                    centre_t = 1.0 - edge_t
                    dir_to_centre = mouse_offset / ti.max(dst, 0.001)
                    
                    # Modify gravity based on mouse interaction
                    gravity_weight = 1.0 - (centre_t * ti.max(0.0, ti.min(1.0, ti.abs(self.mouse_strength[None]) / 10.0)))
                    mouse_accel = dir_to_centre * centre_t * self.mouse_strength[None]
                    velocity_damp = -self.ps.v[p_i][:2] * centre_t
                    
                    d_v[1] *= gravity_weight
                    d_v[:2] += mouse_accel + velocity_damp
            
            # Apply bad apple force if enabled
            if self.bad_apple_enabled[None] == 1 and self.ps.material[p_i] == self.ps.material_fluid:
                x_i = self.ps.x[p_i]
                bounds_size = ti.Vector(self.ps.domain_end[:2]) - ti.Vector(self.ps.domain_start[:2])
                # Convert to center-origin coordinates like HLSL reference
                domain_center = (ti.Vector(self.ps.domain_end[:2]) + ti.Vector(self.ps.domain_start[:2])) / 2
                pos_centered = x_i[:2] - domain_center
                # HLSL: float2 posT = (pos + boundsSize / 2) / boundsSize;
                # This normalizes to [0, 1] range
                pos_t = (pos_centered + bounds_size / 2) / bounds_size
                # HLSL: posT.x = min(max(0, posT.x), boundsSize.x);
                # Note: The reference code clamps to boundsSize.x, but this seems like it should be 1.0
                # since posT is normalized. Keeping exact reference behavior:
                pos_t[0] = ti.min(ti.max(0.0, pos_t[0]), 1.0)
                pos_t[1] = ti.min(ti.max(0.0, pos_t[1]), 1.0)
                # HLSL: int2 pixelCoord = (int2)(posT * (texSize-1));
                tex_size = ti.Vector(self.bad_apple_size)
                pixel_coord = (pos_t * (tex_size - 1)).cast(int)
                data = self.bad_apple_tex[self.current_frame_field[None], pixel_coord[0], pixel_coord[1]]
                
                # HLSL: float2 offset = data.xy - pixelCoord;
                offset = data[:2] - pixel_coord.cast(float)
                has_nearest = data[:2].dot(data[:2]) > 0.01
                offset_len_sq = offset.dot(offset)
                jfa_distance = data[2]  # Distance in pixels to nearest white pixel
                
                # Convert JFA pixel distance to world distance
                pixels_per_world_unit = tex_size[0] / bounds_size[0]  # Assuming uniform scaling
                world_distance = jfa_distance / pixels_per_world_unit
                
                dir_normalized = ti.Vector([0.0, 0.0])
                force_multiplier = 1.0
                
                if has_nearest:
                    # Get local density ratio to determine if area is crowded
                    density_ratio = self.ps.density[p_i] / self.density_0
                    is_crowded = density_ratio > 1.2  # Crowded if density > 120% of rest density
                    
                    if world_distance > 0.5:  # Far from white region (in black region)
                        # Strong attraction toward white region
                        if offset_len_sq > 0.0001:
                            dir_normalized = offset.normalized()
                            force_multiplier = 1.0
                    elif world_distance > 0.15:  # Approaching white region
                        # Moderate attraction, gradually reduce as getting closer
                        if offset_len_sq > 0.0001:
                            dir_normalized = offset.normalized()
                            # Smooth transition: full force at 0.5, zero at 0.15
                            force_multiplier = (world_distance - 0.15) / (0.5 - 0.15)
                    else:  # Inside white region (world_distance < 0.15)
                        # Goal: Move toward boundary (to outline the shape)
                        # BUT: Spread out if too crowded
                        
                        if is_crowded:
                            # Too crowded - use spreading force to reduce density
                            pos_hash = (ti.cast(pos_centered[0] * 1000.0, int) * 73856093) ^ (ti.cast(pos_centered[1] * 1000.0, int) * 19349663)
                            angle = (pos_hash % 628318) / 100000.0
                            spread_dir = ti.Vector([ti.cos(angle), ti.sin(angle)])
                            
                            # Also add component away from boundary
                            if offset_len_sq > 0.0001:
                                boundary_dir = -offset.normalized()
                                spread_dir = (spread_dir * 0.5 + boundary_dir * 0.5).normalized()
                            
                            dir_normalized = spread_dir
                            # Strong spreading when crowded
                            force_multiplier = self.apple_spread_strength * (density_ratio - 1.0)
                        else:
                            # Not crowded - move toward boundary to outline the shape
                            if offset_len_sq > 0.0001:
                                # Move toward nearest boundary
                                dir_normalized = offset.normalized()
                                # Stronger pull when deeper inside (far from boundary)
                                # Weaker near boundary to avoid overshooting
                                distance_factor = world_distance / 0.15  # 0 at boundary, 1 at depth
                                force_multiplier = self.apple_spread_strength * distance_factor * 0.5
                            else:
                                # Exactly on white pixel - very weak random drift
                                pos_hash = (ti.cast(pos_centered[0] * 1000.0, int) * 73856093) ^ (ti.cast(pos_centered[1] * 1000.0, int) * 19349663)
                                angle = (pos_hash % 628318) / 100000.0
                                dir_normalized = ti.Vector([ti.cos(angle), ti.sin(angle)])
                                force_multiplier = self.apple_spread_strength * 0.1
                
                # HLSL: gravityAccel += dir * appleForce;
                apple_force = self.apple_weight * force_multiplier
                d_v += dir_normalized * apple_force
                # HLSL: Positions[index] += dir * deltaTime * appleDisplaceWeight;
                displace = dir_normalized * ti.min(offset.norm(), self.dt[None] * self.apple_displace_weight)
                self.ps.x[p_i][:2] += displace
            
            self.ps.acceleration[p_i] = d_v

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                continue
            
            if self.ps.material[p_i] == self.ps.material_fluid:
                x_i = self.ps.x[p_i]
                d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
                
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]
                    r_ij = x_i - x_j
                    
                    # Surface Tension
                    if self.ps.material[p_j] == self.ps.material_fluid:
                        diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
                        r2 = r_ij.dot(r_ij)
                        if r2 > diameter2:
                            d_v -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r_ij * self.cubic_kernel(r_ij.norm())
                        else:
                            d_v -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r_ij * self.cubic_kernel(self.ps.particle_diameter)
                    
                    # Viscosity Force
                    v_ij = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r_ij)
                    d_factor = 2 * (self.ps.dim + 2)
                    if self.ps.material[p_j] == self.ps.material_fluid:
                        f_v = d_factor * self.viscosity * (self.ps.m[p_j] / self.ps.density[p_j]) * v_ij / (r_ij.norm()**2 + 0.01 * self.h**2) * self.cubic_kernel_derivative(r_ij)
                        d_v += f_v
                        if self.ps.is_dynamic_rigid_body(p_j):
                            self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]
                    elif self.ps.material[p_j] == self.ps.material_solid:
                        boundary_viscosity = 0.0
                        f_v = d_factor * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / self.ps.density[p_i]) * v_ij / (r_ij.norm()**2 + 0.01 * self.h**2) * self.cubic_kernel_derivative(r_ij)
                        d_v += f_v
                        if self.ps.is_dynamic_rigid_body(p_j):
                            self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]
                self.ps.acceleration[p_i] += d_v

    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]
    
    @ti.kernel
    def check_and_respawn_particles(self):
        """Check for invalid particles and respawn them"""
        domain_start = ti.Vector(self.ps.domain_start[:2])
        domain_end = ti.Vector(self.ps.domain_end[:2])
        
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            
            pos = self.ps.x[p_i][:2]
            vel = self.ps.v[p_i][:2]
            
            # Check for invalid position (NaN, Inf, or out of bounds)
            is_invalid = False
            
            # Check NaN/Inf
            if ti.math.isnan(pos[0]) or ti.math.isnan(pos[1]) or ti.math.isinf(pos[0]) or ti.math.isinf(pos[1]):
                is_invalid = True
            if ti.math.isnan(vel[0]) or ti.math.isnan(vel[1]) or ti.math.isinf(vel[0]) or ti.math.isinf(vel[1]):
                is_invalid = True
            
            # Check out of bounds (with large margin)
            margin = 1.0
            if pos[0] < domain_start[0] - margin or pos[0] > domain_end[0] + margin:
                is_invalid = True
            if pos[1] < domain_start[1] - margin or pos[1] > domain_end[1] + margin:
                is_invalid = True
            
            # Check excessive velocity
            # vel_mag_sq = vel.dot(vel)
            # if vel_mag_sq > 250000.0:  # Speed > 500
            #     is_invalid = True
            
            # Respawn if invalid
            if is_invalid:
                # Better pseudo-random number generation using multiple hash stages
                # Use frame count as additional entropy source for temporal variation
                seed = p_i + self.current_frame_field[None] * 10007
                
                # Multiple hash stages for better randomness
                hash1 = (seed * ti.u32(2654435761)) % ti.u32(1000000007)
                hash2 = (hash1 * ti.u32(1103515245) + ti.u32(12345)) % ti.u32(1000000007)
                hash3 = (hash2 * ti.u32(134775813) + ti.u32(1)) % ti.u32(1000000007)
                
                # Convert to [0, 1] range
                rand_x = (hash1 % ti.u32(10000)) / 10000.0
                rand_y = (hash2 % ti.u32(10000)) / 10000.0
                rand_vx = (hash3 % ti.u32(10000)) / 10000.0
                rand_vy = ((hash3 // ti.u32(10000)) % ti.u32(10000)) / 10000.0
                
                # Spawn in center region with some spread
                center = (domain_start + domain_end) / 2
                spread = (domain_end - domain_start) * 0.5
                new_pos = center + (ti.Vector([rand_x, rand_y]) - 0.5) * 2.0 * spread
                
                # Random velocity (small)
                angle = rand_vx * 6.28318  # 2*pi
                speed = rand_vy * 2.0
                new_vel = ti.Vector([ti.cos(angle), ti.sin(angle)]) * speed
                
                self.ps.x[p_i][:2] = new_pos
                if self.ps.dim == 3:
                    self.ps.x[p_i][2] = (domain_start[0] + domain_end[0]) / 2  # Center Z
                self.ps.v[p_i][:2] = new_vel
                if self.ps.dim == 3:
                    self.ps.v[p_i][2] = 0.0

    def update_mouse_interaction(self, mouse_x, mouse_y, strength):
        """Update mouse interaction parameters
        Args:
            mouse_x, mouse_y: Mouse position in world coordinates
            strength: Interaction strength (positive = attract, negative = repel, 0 = none)
        """
        self.mouse_pos[None] = [mouse_x, mouse_y]
        self.mouse_strength[None] = strength
    
    def update_frame(self):
        """Update to next video frame (called once per render frame)"""
        if self.is_bad_apple:
            self.frame_update_counter += 1
            if self.frame_update_counter >= self.frame_update_interval:
                self.frame_update_counter = 0
                current_frame = self.current_frame_field[None]
                
                # Update to next frame
                if self.loop_video:
                    # Loop: wrap around to 0 after last frame
                    next_frame = (current_frame + 1) % self.num_frames
                else:
                    # No loop: clamp to last frame
                    next_frame = min(current_frame + 1, self.num_frames - 1)
                
                self.current_frame_field[None] = next_frame
                
                if next_frame % 1 == 0:  # Print every 1 frames (change 1 to desired interval)
                    print(f"Video frame: {next_frame}/{self.num_frames}")
    
    def substep(self):
        self.apply_external_forces()
        self.compute_densities()
        self.compute_pressure_forces()
        self.compute_non_pressure_forces()
        self.advect()
        if self.is_bad_apple or self.mouse_strength[None] != 0.0:
            self.check_and_respawn_particles()
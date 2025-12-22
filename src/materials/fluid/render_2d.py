import numpy as np
from PIL import Image, ImageDraw
import os

class Renderer2D:
    def __init__(self, width, height, domain_start, domain_end, output_dir):
        """
        Initialize 2D renderer
        Args:
            width, height: Output image resolution
            domain_start, domain_end: Domain boundaries [x, y]
            output_dir: Directory to save rendered images
        """
        self.width = width
        self.height = height
        self.domain_start = np.array(domain_start)
        self.domain_end = np.array(domain_end)
        self.domain_size = self.domain_end - self.domain_start
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def world_to_pixel(self, world_pos):
        """Convert world coordinates to pixel coordinates"""
        # Normalize to [0, 1]
        normalized = (world_pos - self.domain_start) / self.domain_size
        # Convert to pixel coordinates (flip Y axis)
        pixel_x = int(normalized[0] * self.width)
        pixel_y = int((1.0 - normalized[1]) * self.height)  # Flip Y
        return pixel_x, pixel_y
    
    def world_radius_to_pixel(self, world_radius):
        """Convert world space radius to pixel space"""
        # Use X axis scale (assuming uniform aspect ratio)
        return int(world_radius / self.domain_size[0] * self.width)
    
    def render_frame(self, particles_data, frame_num, bad_apple_data=None, config=None):
        """
        Render a single frame
        Args:
            particles_data: dict with keys 'positions', 'velocities', 'radius'
            frame_num: Frame number for filename
            bad_apple_data: dict with 'jfa_results', 'image_data' (optional, for Bad Apple mode)
            config: dict with rendering parameters
        """
        # Create black background
        img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        positions = particles_data['positions']  # Nx2 array
        velocities = particles_data['velocities']  # Nx2 array
        base_radius = particles_data['radius']
        
        if bad_apple_data is not None and config is not None:
            # Bad Apple mode: classify and render particles
            jfa_data = bad_apple_data['jfa_results']
            image_data = bad_apple_data['image_data']
            bad_apple_size = bad_apple_data['size']
            
            max_velocity = config.get('maxVelocityFor2DRender', 50.0)
            dist_threshold = config.get('distThresholdFor2DRender', 10.0)
            
            # Check if image has white pixels
            has_white = np.any(image_data > 0.5)
            
            # Classify particles
            background_particles = []
            normal_particles = []
            
            for i in range(len(positions)):
                pos = positions[i]
                vel = velocities[i]
                
                # Map to image coordinates
                pixel_x = int((pos[0] - self.domain_start[0]) / self.domain_size[0] * (bad_apple_size[0] - 1))
                pixel_y = int((pos[1] - self.domain_start[1]) / self.domain_size[1] * (bad_apple_size[1] - 1))
                pixel_x = max(0, min(pixel_x, bad_apple_size[0] - 1))
                pixel_y = max(0, min(pixel_y, bad_apple_size[1] - 1))
                
                # Get JFA data: [nearest_x, nearest_y, distance, valid]
                jfa_value = jfa_data[pixel_x, pixel_y]
                best_dist = jfa_value[2]
                
                # Get image value (0=black, 1=white)
                img_value = image_data[pixel_x, pixel_y]
                
                # Classification logic:
                # Background particle: black region AND image has white pixels
                # Normal particle: white region OR pure black image (no white pixels)
                if has_white and img_value < 0.5:
                    # Background particle: black region when image has white
                    background_particles.append((pos, vel, best_dist))
                else:
                    # Normal particle: white region OR all particles in pure black image
                    normal_particles.append((pos, vel))
            
            # Render background particles first (behind)
            pixel_base_radius = self.world_radius_to_pixel(base_radius)
            for pos, vel, best_dist in background_particles:
                # Radius decreases linearly with distance
                # if best_dist > dist_threshold:
                #     continue  # Don't render if too far
                
                # radius_ratio = 1.0 - (best_dist / dist_threshold)
                # particle_radius = pixel_base_radius * radius_ratio
                
                # if particle_radius < pixel_base_radius * 0.1:
                #     continue  # Don't render if too small

                particle_radius = pixel_base_radius
                
                px, py = self.world_to_pixel(pos)
                color = (0, 0, 255)  # Fixed blue color
                
                # Draw filled circle
                draw.ellipse(
                    [px - particle_radius, py - particle_radius,
                     px + particle_radius, py + particle_radius],
                    fill=color
                )
            
            # Render normal particles on top
            for pos, vel in normal_particles:
                speed = np.sqrt(vel[0]**2 + vel[1]**2)
                
                # Color interpolation
                base_color = np.array([0, 90, 255])
                white_color = np.array([180, 250, 255])
                
                t = min(speed / max_velocity, 1.0)  # Clamp to [0, 1]
                color_float = base_color * (1 - t) + white_color * t
                color = tuple(color_float.astype(int))
                
                px, py = self.world_to_pixel(pos)
                particle_radius = pixel_base_radius * 3.0
                
                # Draw filled circle
                draw.ellipse(
                    [px - particle_radius, py - particle_radius,
                     px + particle_radius, py + particle_radius],
                    fill=color
                )

        else:
            # Normal mode: render all particles with same color and size
            pixel_base_radius = self.world_radius_to_pixel(base_radius)
            for i in range(len(positions)):
                pos = positions[i]
                px, py = self.world_to_pixel(pos)
                color = (100, 150, 255)  # Default color
                
                draw.ellipse(
                    [px - pixel_base_radius, py - pixel_base_radius,
                     px + pixel_base_radius, py + pixel_base_radius],
                    fill=color
                )
        
        # Save image
        output_path = os.path.join(self.output_dir, f"frame_{frame_num:04d}.png")
        img.save(output_path)
        return output_path

"""Rigid body simulation wrapper for integration with main project.

This module wraps the rigid body simulation functionality from rigid_body_sim.py
and rigid_body_sim_simple.py, making them compatible with the project's
configuration and export systems.
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

# Import vector math utilities
Vector = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (w, x, y, z)
Matrix3 = Tuple[Vector, Vector, Vector]


def vec_add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(v: Vector, s: float) -> Vector:
    return (v[0] * s, v[1] * s, v[2] * s)


def vec_dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_length(v: Vector) -> float:
    return math.sqrt(vec_dot(v, v))


def vec_normalize(v: Vector) -> Vector:
    length = vec_length(v)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / length
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def quat_normalize(q: Quaternion) -> Quaternion:
    length = math.sqrt(sum(component * component for component in q))
    if length == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(component / length for component in q)  # type: ignore


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_from_axis_angle(axis: Vector, angle: float) -> Quaternion:
    axis_length = vec_length(axis)
    if axis_length == 0:
        return (1.0, 0.0, 0.0, 0.0)
    half_angle = 0.5 * angle
    s = math.sin(half_angle) / axis_length
    return (math.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s)


def quat_integrate(orientation: Quaternion, angular_velocity: Vector, dt: float) -> Quaternion:
    """Integrate orientation using first-order approximation."""
    if angular_velocity == (0.0, 0.0, 0.0):
        return orientation
    omega_mag = vec_length(angular_velocity)
    axis = angular_velocity if omega_mag == 0 else vec_scale(angular_velocity, 1.0 / omega_mag)
    delta = quat_from_axis_angle(axis, omega_mag * dt)
    return quat_normalize(quat_multiply(delta, orientation))


def quat_to_matrix(q: Quaternion) -> Matrix3:
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def mat3_mul_vec(m: Matrix3, v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def mat3_mul_vec_transpose(m: Matrix3, v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    )


BOX_TRIANGLES = (
    (1, 2, 3), (1, 3, 4),  # back face
    (5, 6, 7), (5, 7, 8),  # front face
    (1, 5, 8), (1, 8, 4),  # left face
    (2, 6, 7), (2, 7, 3),  # right face
    (1, 2, 6), (1, 6, 5),  # bottom face
    (4, 3, 7), (4, 7, 8),  # top face
)


def box_vertices(size: Vector) -> List[Vector]:
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    return [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]


def _append_unique_axis(axes: List[Vector], axis: Vector, eps: float = 1e-5) -> None:
    axis = vec_normalize(axis)
    if axis == (0.0, 0.0, 0.0):
        return
    for existing in axes:
        if abs(vec_dot(existing, axis)) >= 1.0 - eps:
            return
    axes.append(axis)


def _zero_based_triangles(triangles: Sequence[Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], ...]:
    converted: List[Tuple[int, int, int]] = []
    for tri in triangles:
        converted.append((tri[0] - 1, tri[1] - 1, tri[2] - 1))
    return tuple(converted)


class ConvexShape:
    """Convex collision shape for rigid bodies."""
    
    def __init__(self, vertices: Sequence[Vector], triangles: Sequence[Tuple[int, int, int]]):
        if not vertices:
            raise ValueError("ConvexShape requires at least one vertex")
        if not triangles:
            raise ValueError("ConvexShape requires at least one triangle")
        self._vertices: Tuple[Vector, ...] = tuple(vertices)
        self._triangles: Tuple[Tuple[int, int, int], ...] = tuple(triangles)
        self._face_normals: Tuple[Vector, ...] = self._compute_face_normals()
        self._edge_dirs: Tuple[Vector, ...] = self._compute_edge_dirs()
        self._bounding_radius: float = max(vec_length(v) for v in self._vertices)
        mins = [min(vertex[i] for vertex in self._vertices) for i in range(3)]
        maxs = [max(vertex[i] for vertex in self._vertices) for i in range(3)]
        self._extents = tuple(maxs[i] - mins[i] for i in range(3))

    @property
    def vertices(self) -> Tuple[Vector, ...]:
        return self._vertices

    @property
    def triangles(self) -> Tuple[Tuple[int, int, int], ...]:
        return self._triangles

    @property
    def face_normals(self) -> Tuple[Vector, ...]:
        return self._face_normals

    @property
    def edge_dirs(self) -> Tuple[Vector, ...]:
        return self._edge_dirs

    def bounding_radius(self) -> float:
        return self._bounding_radius

    def min_half_extent(self) -> float:
        return 0.5 * min(self._extents)

    def axis_aligned_extents(self) -> Tuple[float, float, float]:
        return self._extents

    def inertia_tensor_diag(self, mass: float) -> Vector:
        ex, ey, ez = self._extents
        return (
            mass * (ey * ey + ez * ez) / 12.0,
            mass * (ex * ex + ez * ez) / 12.0,
            mass * (ex * ex + ey * ey) / 12.0,
        )

    def support_point(self, orientation: Quaternion, position: Vector, direction: Vector) -> Vector:
        rotation = quat_to_matrix(orientation)
        return self.support_point_with_rotation(rotation, position, direction)

    def support_point_with_rotation(self, rotation: Matrix3, position: Vector, direction: Vector) -> Vector:
        local_dir = mat3_mul_vec_transpose(rotation, direction)
        idx = self._support_index_local(local_dir)
        return vec_add(mat3_mul_vec(rotation, self._vertices[idx]), position)

    def world_vertices(self, rotation: Matrix3, position: Vector) -> List[Vector]:
        return [vec_add(mat3_mul_vec(rotation, v), position) for v in self._vertices]

    def world_face_normals(self, rotation: Matrix3) -> List[Vector]:
        return [mat3_mul_vec(rotation, n) for n in self._face_normals]

    def world_edge_dirs(self, rotation: Matrix3) -> List[Vector]:
        return [mat3_mul_vec(rotation, d) for d in self._edge_dirs]

    def _support_index_local(self, direction: Vector) -> int:
        best_idx = 0
        best_dot = vec_dot(self._vertices[0], direction)
        for i, vertex in enumerate(self._vertices[1:], start=1):
            d = vec_dot(vertex, direction)
            if d > best_dot:
                best_dot = d
                best_idx = i
        return best_idx

    def _compute_face_normals(self) -> Tuple[Vector, ...]:
        normals: List[Vector] = []
        for tri in self._triangles:
            v0, v1, v2 = self._vertices[tri[0]], self._vertices[tri[1]], self._vertices[tri[2]]
            edge1 = vec_sub(v1, v0)
            edge2 = vec_sub(v2, v0)
            normal = vec_normalize(vec_cross(edge1, edge2))
            if normal != (0.0, 0.0, 0.0):
                normals.append(normal)
        return tuple(normals)

    def _compute_edge_dirs(self) -> Tuple[Vector, ...]:
        edges: List[Vector] = []
        seen_pairs = set()
        for tri in self._triangles:
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                pair = tuple(sorted([a, b]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edge_vec = vec_sub(self._vertices[b], self._vertices[a])
                    _append_unique_axis(edges, edge_vec)
        return tuple(edges)


class BoxShape(ConvexShape):
    """Box collision shape."""
    
    def __init__(self, size: Vector):
        vertices = box_vertices(size)
        triangles = _zero_based_triangles(BOX_TRIANGLES)
        super().__init__(vertices, triangles)


class MeshShape(ConvexShape):
    """Custom mesh collision shape loaded from OBJ file."""
    
    def __init__(self, vertices: Sequence[Vector], triangles: Sequence[Tuple[int, int, int]]):
        super().__init__(vertices, triangles)

    @classmethod
    def from_obj(cls, path: Path, *, scale: float = 1.0, recenter: bool = True) -> MeshShape:
        """Load mesh from OBJ file."""
        vertices: List[Vector] = []
        triangles: List[Tuple[int, int, int]] = []
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append((x * scale, y * scale, z * scale))
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    indices = []
                    for part in parts:
                        idx = int(part.split('/')[0])
                        indices.append(idx)
                    if len(indices) >= 3:
                        triangles.append((indices[0], indices[1], indices[2]))
        
        if recenter:
            cx = sum(v[0] for v in vertices) / len(vertices)
            cy = sum(v[1] for v in vertices) / len(vertices)
            cz = sum(v[2] for v in vertices) / len(vertices)
            vertices = [(v[0] - cx, v[1] - cy, v[2] - cz) for v in vertices]
        
        triangles_zero = _zero_based_triangles(triangles)
        return cls(vertices, triangles_zero)


def load_convex_mesh(path: Path, *, scale: float = 1.0, recenter: bool = True) -> MeshShape:
    """Load a convex mesh from an OBJ file for use as a collision shape."""
    return MeshShape.from_obj(path, scale=scale, recenter=recenter)


@dataclass
class RigidBody:
    """Rigid body with physical properties."""
    
    object_id: int
    mass: float
    size: Vector  # box dimensions (width, height, depth)
    position: Vector
    velocity: Vector
    orientation: Quaternion = (1.0, 0.0, 0.0, 0.0)
    angular_velocity: Vector = (0.0, 0.0, 0.0)
    restitution: float = 0.35
    linear_damping: float = 0.01
    angular_damping: float = 0.02
    shape: Optional[ConvexShape] = None
    forces: List[Vector] = field(default_factory=list)
    torques: List[Vector] = field(default_factory=list)

    def add_force(self, force: Vector) -> None:
        self.forces.append(force)

    def add_torque(self, torque: Vector) -> None:
        self.torques.append(torque)

    @property
    def inertia_tensor_diag(self) -> Vector:
        if self.shape:
            return self.shape.inertia_tensor_diag(self.mass)
        else:
            w, h, d = self.size
            return (
                self.mass * (h * h + d * d) / 12.0,
                self.mass * (w * w + d * d) / 12.0,
                self.mass * (w * w + h * h) / 12.0,
            )

    def clear_accumulators(self) -> None:
        self.forces.clear()
        self.torques.clear()


@dataclass
class SimulationConfig:
    """Configuration for rigid body simulation."""
    
    duration: float = 5.0
    dt: float = 1.0 / 120.0
    gravity: Vector = (0.0, -9.81, 0.0)
    ground_height: float = 0.0
    ground_friction: float = 0.4
    enable_body_collisions: bool = False  # Set to True for complex simulation


class RigidBodySimulation:
    """Rigid body physics simulation."""
    
    def __init__(self, bodies: Iterable[RigidBody], config: SimulationConfig):
        self.bodies = list(bodies)
        self.config = config
        self.time = 0.0
        self.history: List[Dict] = []
        self._body_sizes: Dict[int, Vector] = {body.object_id: body.size for body in self.bodies}
        if len(self._body_sizes) != len(self.bodies):
            raise ValueError("Duplicate body objectId detected")

    def step(self) -> None:
        """Perform one simulation step."""
        dt = self.config.dt
        gravity = self.config.gravity

        for body in self.bodies:
            body.add_force(vec_scale(gravity, body.mass))
            
            # Accumulate forces and torques
            net_force = (0.0, 0.0, 0.0)
            net_torque = (0.0, 0.0, 0.0)
            for force in body.forces:
                net_force = vec_add(net_force, force)
            for torque in body.torques:
                net_torque = vec_add(net_torque, torque)
            
            # Linear integration
            acceleration = vec_scale(net_force, 1.0 / body.mass)
            body.velocity = vec_add(body.velocity, vec_scale(acceleration, dt))
            body.velocity = vec_scale(body.velocity, 1.0 - body.linear_damping)
            body.position = vec_add(body.position, vec_scale(body.velocity, dt))
            
            # Angular integration
            angular_accel = self._apply_inverse_inertia(body, net_torque)
            body.angular_velocity = vec_add(body.angular_velocity, vec_scale(angular_accel, dt))
            body.angular_velocity = vec_scale(body.angular_velocity, 1.0 - body.angular_damping)
            body.orientation = quat_integrate(body.orientation, body.angular_velocity, dt)
            
            body.clear_accumulators()
            
            # Ground collision
            self._resolve_ground_contact(body)
        
        # Body-body collisions (if enabled)
        if self.config.enable_body_collisions:
            self._resolve_body_collisions()

        self.time += dt
        self._record_frame()

    def run(self) -> None:
        """Run the complete simulation."""
        steps = int(self.config.duration / self.config.dt)
        for i in range(steps):
            self.step()
            if i % 120 == 0:  # Print every 120 steps (~1 second at 120Hz)
                print(f"Simulation step {i}/{steps} ({i/steps*100:.1f}%)")

    def export_history(self, path: Path) -> None:
        """Export simulation history to JSON."""
        data = {
            "dt": self.config.dt,
            "frames": self.history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def export_obj_sequence(self, directory: Path, prefix: str = "obj") -> None:
        """Export OBJ sequence compatible with Blender import."""
        if not self.history:
            print("Warning: No simulation history to export")
            return

        directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each body
        body_dirs = {}
        for body in self.bodies:
            body_dir = directory / f"obj_{body.object_id}"
            body_dir.mkdir(exist_ok=True)
            body_dirs[body.object_id] = body_dir
        
        # Export each frame
        for frame_idx, frame in enumerate(self.history):
            for body_data in frame["bodies"]:
                object_id = body_data["objectId"]
                if object_id not in body_dirs:
                    continue
                
                body_dir = body_dirs[object_id]
                position = body_data["position"]
                orientation = body_data["orientation"]
                
                # Find body size
                size = self._body_sizes[object_id]
                
                # Get vertices and transform them
                rotation = quat_to_matrix(orientation)
                
                # Find the body to get its shape
                body = next((b for b in self.bodies if b.object_id == object_id), None)
                if body and body.shape:
                    local_vertices = body.shape.vertices
                    triangles = body.shape.triangles
                else:
                    local_vertices = box_vertices(size)
                    triangles = _zero_based_triangles(BOX_TRIANGLES)
                
                world_vertices = [
                    vec_add(mat3_mul_vec(rotation, v), position)
                    for v in local_vertices
                ]
                
                # Write OBJ file
                obj_path = body_dir / f"obj_{object_id}_{frame_idx:06d}.obj"
                with open(obj_path, 'w') as f:
                    f.write(f"# Frame {frame_idx} - obj_{object_id}\n")
                    for v in world_vertices:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    for tri in triangles:
                        f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        
        print(f"Exported {len(self.history)} frames for {len(self.bodies)} bodies to {directory}")

    def _record_frame(self) -> None:
        """Record current frame state."""
        frame_data = {
            "time": self.time,
            "bodies": [
                {
                    "objectId": body.object_id,
                    "position": body.position,
                    "orientation": body.orientation,
                }
                for body in self.bodies
            ],
        }
        self.history.append(frame_data)

    def _resolve_ground_contact(self, body: RigidBody) -> None:
        """Resolve collision with ground plane."""
        if body.shape:
            min_half_extent = body.shape.min_half_extent()
        else:
            min_half_extent = 0.5 * min(body.size)
        
        penetration = (body.position[1] - min_half_extent) - self.config.ground_height
        if penetration >= 0:
            return

        corrected_y = self.config.ground_height + min_half_extent
        body.position = (body.position[0], corrected_y, body.position[2])
        normal = (0.0, 1.0, 0.0)
        contact_offset = self._estimate_contact_offset(body, min_half_extent)

        # Resolve normal impulse
        contact_velocity = vec_add(body.velocity, vec_cross(body.angular_velocity, contact_offset))
        normal_speed = vec_dot(contact_velocity, normal)
        if normal_speed < 0.0:
            impulse_mag = -(1.0 + body.restitution) * normal_speed
            r_cross_n = vec_cross(contact_offset, normal)
            angular_effect = self._apply_inverse_inertia(body, r_cross_n)
            denominator = 1.0 / body.mass + vec_dot(vec_cross(angular_effect, contact_offset), normal)
            if denominator != 0.0:
                impulse_mag /= denominator
            
            impulse = vec_scale(normal, impulse_mag)
            body.velocity = vec_add(body.velocity, vec_scale(impulse, 1.0 / body.mass))
            delta_omega = self._apply_inverse_inertia(body, vec_cross(contact_offset, impulse))
            body.angular_velocity = vec_add(body.angular_velocity, delta_omega)

        # Clamp downward velocity
        body.velocity = (body.velocity[0], max(body.velocity[1], 0.0), body.velocity[2])

        # Apply ground support torque
        support_force_mag = max(0.0, -vec_dot(self.config.gravity, normal)) * body.mass
        support_force = vec_scale(normal, support_force_mag)
        support_torque = vec_cross(contact_offset, support_force)
        angular_delta = self._apply_inverse_inertia(body, vec_scale(support_torque, self.config.dt))
        body.angular_velocity = vec_add(body.angular_velocity, angular_delta)

    def _estimate_contact_offset(self, body: RigidBody, fallback_half_extent: float) -> Vector:
        """Estimate contact point offset from body center."""
        rotation = quat_to_matrix(body.orientation)
        
        if body.shape:
            local_vertices = body.shape.vertices
        else:
            local_vertices = box_vertices(body.size)
        
        world_vertices = [vec_add(mat3_mul_vec(rotation, vertex), body.position) for vertex in local_vertices]
        min_height = min(vertex[1] for vertex in world_vertices)
        threshold = min_height + 1e-4
        accum = (0.0, 0.0, 0.0)
        count = 0
        for vertex in world_vertices:
            if vertex[1] <= threshold:
                accum = vec_add(accum, vertex)
                count += 1
        if count == 0:
            return (0.0, -fallback_half_extent, 0.0)
        contact_point = vec_scale(accum, 1.0 / count)
        return vec_sub(contact_point, body.position)

    def _apply_inverse_inertia(self, body: RigidBody, torque: Vector) -> Vector:
        """Apply inverse inertia tensor to torque."""
        rotation = quat_to_matrix(body.orientation)
        inertia = body.inertia_tensor_diag
        torque_local = mat3_mul_vec_transpose(rotation, torque)
        delta_local = (
            torque_local[0] / inertia[0] if inertia[0] != 0.0 else 0.0,
            torque_local[1] / inertia[1] if inertia[1] != 0.0 else 0.0,
            torque_local[2] / inertia[2] if inertia[2] != 0.0 else 0.0,
        )
        return mat3_mul_vec(rotation, delta_local)

    def _resolve_body_collisions(self) -> None:
        """Resolve collisions between rigid bodies (SAT-based)."""
        # Simplified collision detection for body-body interactions
        # This is a basic implementation; full SAT is more complex
        for i, body_a in enumerate(self.bodies):
            for body_b in self.bodies[i + 1:]:
                # Simple bounding sphere check
                dist = vec_length(vec_sub(body_a.position, body_b.position))
                if body_a.shape and body_b.shape:
                    sum_radii = body_a.shape.bounding_radius() + body_b.shape.bounding_radius()
                else:
                    sum_radii = 0.5 * (max(body_a.size) + max(body_b.size))
                
                if dist < sum_radii:
                    # Simple separation response
                    normal = vec_normalize(vec_sub(body_b.position, body_a.position))
                    if normal == (0.0, 0.0, 0.0):
                        normal = (0.0, 1.0, 0.0)
                    
                    penetration = sum_radii - dist
                    correction = vec_scale(normal, penetration * 0.5)
                    body_a.position = vec_sub(body_a.position, correction)
                    body_b.position = vec_add(body_b.position, correction)
                    
                    # Apply impulse
                    rel_vel = vec_sub(body_b.velocity, body_a.velocity)
                    vel_along_normal = vec_dot(rel_vel, normal)
                    if vel_along_normal < 0:
                        continue
                    
                    restitution = min(body_a.restitution, body_b.restitution)
                    impulse_mag = -(1.0 + restitution) * vel_along_normal
                    impulse_mag /= (1.0 / body_a.mass + 1.0 / body_b.mass)
                    
                    impulse = vec_scale(normal, impulse_mag)
                    body_a.velocity = vec_sub(body_a.velocity, vec_scale(impulse, 1.0 / body_a.mass))
                    body_b.velocity = vec_add(body_b.velocity, vec_scale(impulse, 1.0 / body_b.mass))


def create_simulation_from_config(config) -> RigidBodySimulation:
    """Create rigid body simulation from project config."""
    # Extract configuration
    sim_cfg = config.get_cfg("RigidBodySimulation", {})
    
    duration = sim_cfg.get("duration", 5.0)
    dt = sim_cfg.get("dt", 1.0 / 120.0)
    gravity = tuple(sim_cfg.get("gravity", [0.0, -9.81, 0.0]))
    ground_height = sim_cfg.get("groundHeight", 0.0)
    ground_friction = sim_cfg.get("groundFriction", 0.4)
    enable_collisions = sim_cfg.get("enableBodyCollisions", False)
    
    sim_config = SimulationConfig(
        duration=duration,
        dt=dt,
        gravity=gravity,
        ground_height=ground_height,
        ground_friction=ground_friction,
        enable_body_collisions=enable_collisions,
    )
    
    # Create rigid bodies from config
    bodies = []
    rigid_bodies_cfg = config.get_rigid_bodies()
    
    for rb_cfg in rigid_bodies_cfg:
        object_id = rb_cfg.get("objectId", 1)
        mass = rb_cfg.get("mass", 1.0)
        size = tuple(rb_cfg.get("size", [1.0, 1.0, 1.0]))
        position = tuple(rb_cfg.get("position", [0.0, 2.0, 0.0]))
        velocity = tuple(rb_cfg.get("velocity", [0.0, 0.0, 0.0]))
        angular_velocity = tuple(rb_cfg.get("angularVelocity", [0.0, 0.0, 0.0]))
        
        # Handle orientation
        orientation_cfg = rb_cfg.get("orientation", None)
        if orientation_cfg:
            if "quaternion" in orientation_cfg:
                orientation = tuple(orientation_cfg["quaternion"])
            elif "eulerDegrees" in orientation_cfg:
                euler = orientation_cfg["eulerDegrees"]
                qx = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(euler[0]))
                qy = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(euler[1]))
                qz = quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(euler[2]))
                orientation = quat_normalize(quat_multiply(qz, quat_multiply(qy, qx)))
            else:
                orientation = (1.0, 0.0, 0.0, 0.0)
        else:
            orientation = (1.0, 0.0, 0.0, 0.0)
        
        restitution = rb_cfg.get("restitution", 0.35)
        linear_damping = rb_cfg.get("linearDamping", 0.01)
        angular_damping = rb_cfg.get("angularDamping", 0.02)
        
        # Handle custom mesh shape
        shape = None
        if "meshFile" in rb_cfg:
            mesh_path = Path(rb_cfg["meshFile"])
            if mesh_path.exists():
                scale = rb_cfg.get("meshScale", 1.0)
                recenter = rb_cfg.get("meshRecenter", True)
                shape = load_convex_mesh(mesh_path, scale=scale, recenter=recenter)
        
        body = RigidBody(
            object_id=object_id,
            mass=mass,
            size=size,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            restitution=restitution,
            linear_damping=linear_damping,
            angular_damping=angular_damping,
            shape=shape,
        )
        bodies.append(body)
    
    return RigidBodySimulation(bodies, sim_config)

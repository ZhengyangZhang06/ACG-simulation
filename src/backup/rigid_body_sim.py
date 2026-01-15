"""Simple rigid body simulation core.

This module implements a tiny rigid body simulator tailored for exporting
animation data that Blender can play back.  It avoids any high-level
physics libraries and instead relies on explicit integration and a couple of
vector helpers implemented from scratch.
"""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, cast
import heapq
import json
import math
import os
import time
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


def vec_normalize(v: Vector) -> Vector:
    length = vec_length(v)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / length
    return (v[0] * inv, v[1] * inv, v[2] * inv)


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
    def __init__(self, vertices: Sequence[Vector], triangles: Sequence[Tuple[int, int, int]]):
        if not vertices:
            raise ValueError("ConvexShape requires at least one vertex")
        if not triangles:
            raise ValueError("ConvexShape requires face indices for SAT")
        self._vertices: Tuple[Vector, ...] = tuple(vertices)
        self._triangles: Tuple[Tuple[int, int, int], ...] = tuple(triangles)
        self._face_normals: Tuple[Vector, ...] = self._compute_face_normals()
        self._edge_dirs: Tuple[Vector, ...] = self._compute_edge_dirs()
        self._bounding_radius: float = max(vec_length(v) for v in self._vertices)
        mins = [min(vertex[i] for vertex in self._vertices) for i in range(3)]
        maxs = [max(vertex[i] for vertex in self._vertices) for i in range(3)]
        self._extents: Tuple[float, float, float] = tuple(max(maxs[i] - mins[i], 1e-4) for i in range(3))
        self._min_half_extent = 0.5 * min(self._extents)

    @property
    def triangles(self) -> Tuple[Tuple[int, int, int], ...]:
        return self._triangles

    @property
    def local_vertices(self) -> Tuple[Vector, ...]:
        return self._vertices

    @property
    def local_face_normals(self) -> Tuple[Vector, ...]:
        return self._face_normals

    @property
    def local_edge_dirs(self) -> Tuple[Vector, ...]:
        return self._edge_dirs

    def bounding_radius(self) -> float:
        return self._bounding_radius

    def min_half_extent(self) -> float:
        return self._min_half_extent

    def axis_aligned_extents(self) -> Tuple[float, float, float]:
        return self._extents

    def inertia_tensor_diag(self, mass: float) -> Vector:
        # Approximate inertia using the axis-aligned bounding box dimensions.
        w, h, d = self._extents
        i_x = (1.0 / 12.0) * mass * (h * h + d * d)
        i_y = (1.0 / 12.0) * mass * (w * w + d * d)
        i_z = (1.0 / 12.0) * mass * (w * w + h * h)
        return (i_x, i_y, i_z)

    def support_point(self, orientation: Quaternion, position: Vector, direction: Vector) -> Vector:
        rotation = quat_to_matrix(orientation)
        return self.support_point_with_rotation(rotation, position, direction)

    def support_point_with_rotation(self, rotation: Matrix3, position: Vector, direction: Vector) -> Vector:
        direction_local = mat3_mul_vec_transpose(rotation, direction)
        best_index = self._support_index_local(direction_local)
        local_point = self._vertices[best_index]
        return vec_add(mat3_mul_vec(rotation, local_point), position)

    def world_vertices(self, rotation: Matrix3, position: Vector) -> List[Vector]:
        return [vec_add(mat3_mul_vec(rotation, vertex), position) for vertex in self._vertices]

    def world_face_normals(self, rotation: Matrix3) -> List[Vector]:
        return [mat3_mul_vec(rotation, normal) for normal in self._face_normals]

    def world_edge_dirs(self, rotation: Matrix3) -> List[Vector]:
        return [mat3_mul_vec(rotation, edge) for edge in self._edge_dirs]

    def _support_index_local(self, direction: Vector) -> int:
        best_index = 0
        best_dot = -math.inf
        for idx, vertex in enumerate(self._vertices):
            score = vec_dot(vertex, direction)
            if score > best_dot:
                best_dot = score
                best_index = idx
        return best_index

    def _compute_face_normals(self) -> Tuple[Vector, ...]:
        normals: List[Vector] = []
        for tri in self._triangles:
            a, b, c = tri
            edge1 = vec_sub(self._vertices[b], self._vertices[a])
            edge2 = vec_sub(self._vertices[c], self._vertices[a])
            normal = vec_cross(edge1, edge2)
            _append_unique_axis(normals, normal)
        if not normals:
            normals.extend([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        return tuple(normals)

    def _compute_edge_dirs(self) -> Tuple[Vector, ...]:
        edges: List[Vector] = []
        for tri in self._triangles:
            a, b, c = tri
            _append_unique_axis(edges, vec_sub(self._vertices[b], self._vertices[a]))
            _append_unique_axis(edges, vec_sub(self._vertices[c], self._vertices[b]))
            _append_unique_axis(edges, vec_sub(self._vertices[a], self._vertices[c]))
        if not edges:
            edges.extend([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        return tuple(edges)


class BoxShape(ConvexShape):
    def __init__(self, size: Vector):
        self.size = size
        vertices = box_vertices(size)
        zero_based = _zero_based_triangles(BOX_TRIANGLES)
        super().__init__(vertices, zero_based)


class MeshShape(ConvexShape):
    def __init__(self, vertices: Sequence[Vector], triangles: Sequence[Tuple[int, int, int]]):
        super().__init__(vertices, triangles)

    @classmethod
    def from_obj(
        cls,
        path: Path,
        *,
        scale: float = 1.0,
        recenter: bool = True,
    ) -> "MeshShape":
        text = path.read_text().splitlines()
        vertices: List[Vector] = []
        triangles: List[Tuple[int, int, int]] = []
        for line in text:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                vertices.append((x * scale, y * scale, z * scale))
            elif parts[0] == "f" and len(parts) >= 4:
                indices: List[int] = []
                for token in parts[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    idx = int(raw)
                    if idx > 0:
                        indices.append(idx - 1)
                    else:
                        indices.append(len(vertices) + idx)
                if len(indices) < 3:
                    continue
                for i in range(1, len(indices) - 1):
                    triangles.append((indices[0], indices[i], indices[i + 1]))
        if not vertices:
            raise ValueError(f"Mesh '{path}' does not contain vertices")
        if not triangles:
            raise ValueError(f"Mesh '{path}' does not contain faces")

        if recenter:
            count = float(len(vertices))
            centroid = (
                sum(v[0] for v in vertices) / count,
                sum(v[1] for v in vertices) / count,
                sum(v[2] for v in vertices) / count,
            )
            vertices = [vec_sub(v, centroid) for v in vertices]
        return cls(vertices, triangles)


def load_convex_mesh(path: Path, *, scale: float = 1.0, recenter: bool = True) -> MeshShape:
    """Load a convex mesh from an OBJ file for use as a collision shape."""
    return MeshShape.from_obj(path, scale=scale, recenter=recenter)


@dataclass
class RigidBody:
    name: str
    mass: float
    size: Vector | None  # Optional box dimensions
    position: Vector
    velocity: Vector
    shape: ConvexShape | None = None
    orientation: Quaternion = (1.0, 0.0, 0.0, 0.0)
    angular_velocity: Vector = (0.0, 0.0, 0.0)
    restitution: float = 0.01
    linear_damping: float = 0.01
    angular_damping: float = 0.02
    forces: List[Vector] = field(default_factory=list)
    torques: List[Vector] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.shape is None:
            if self.size is None:
                raise ValueError("RigidBody requires either a size or an explicit shape")
            self.shape = BoxShape(self.size)
        elif self.size is None:
            self.size = self.shape.axis_aligned_extents()

    def add_force(self, force: Vector) -> None:
        self.forces.append(force)

    def add_torque(self, torque: Vector) -> None:
        self.torques.append(torque)

    @property
    def inertia_tensor_diag(self) -> Vector:
        assert self.shape is not None
        return self.shape.inertia_tensor_diag(self.mass)

    def clear_accumulators(self) -> None:
        self.forces.clear()
        self.torques.clear()

    @property
    def bounding_radius(self) -> float:
        assert self.shape is not None
        return self.shape.bounding_radius()


@dataclass
class SimulationConfig:
    duration: float = 5.0
    dt: float = 1.0 / 30.0
    gravity: Vector = (0.0, -9.81, 0.0)
    ground_height: float = 0.0
    ground_friction: float = 0.0
    resting_speed_epsilon: float = 0.3
    parallel_integration: bool = False
    parallel_collisions: bool = False
    max_workers: int | None = None
    broad_phase_margin: float = 0.05


@dataclass(frozen=True)
class ImpulseEvent:
    time: float
    body_name: str
    impulse: Vector
    local_offset: Vector | None = None


class RigidBodySimulation:
    def __init__(
        self,
        bodies: Iterable[RigidBody],
        config: SimulationConfig | None = None,
        impulses: Iterable[ImpulseEvent] | None = None,
    ) -> None:
        self.bodies = list(bodies)
        self.config = config or SimulationConfig()
        self.time = 0.0
        self.history: List[Dict[str, object]] = []
        self._body_shapes: Dict[str, ConvexShape] = {}
        self._body_lookup: Dict[str, RigidBody] = {}
        self._worker_target = self._compute_worker_target()
        self._executor: ProcessPoolExecutor | None = None
        for body in self.bodies:
            if body.shape is None:
                raise ValueError(f"Body '{body.name}' is missing a collision shape")
            if body.name in self._body_lookup:
                raise ValueError("Rigid body names must be unique")
            self._body_lookup[body.name] = body
            self._body_shapes[body.name] = body.shape
        self._pending_impulses: List[Tuple[float, int, ImpulseEvent]] = []
        self._impulse_seq = 0
        if impulses:
            for event in impulses:
                self._schedule_impulse(event)

    def step(self) -> None:
        dt = self.config.dt
        gravity = self.config.gravity

        self._apply_scheduled_impulses(self.time + dt)

        executor = self._get_executor() if self.config.parallel_integration else None
        if executor is not None and len(self.bodies) > 1:
            futures: List[Tuple[int, Future[Tuple[Vector, Vector, Quaternion, Vector]]]] = []
            for idx, body in enumerate(self.bodies):
                future = executor.submit(
                    _integrate_body_worker,
                    body.position,
                    body.velocity,
                    body.orientation,
                    body.angular_velocity,
                    body.mass,
                    body.inertia_tensor_diag,
                    body.linear_damping,
                    body.angular_damping,
                    dt,
                    gravity,
                )
                futures.append((idx, future))
            for idx, future in futures:
                position, velocity, orientation, angular_velocity = future.result()
                body = self.bodies[idx]
                body.position = position
                body.velocity = velocity
                body.orientation = orientation
                body.angular_velocity = angular_velocity
                self._resolve_ground_contact(body)
                body.clear_accumulators()
        else:
            for body in self.bodies:
                self._integrate_body(body, dt, gravity)

        self._resolve_body_collisions()

        self.time += dt
        self._record_frame()

    def run(self) -> None:
        steps = int(self.config.duration / self.config.dt)
        for _ in range(steps):
            self.step()

    def _integrate_body(self, body: RigidBody, dt: float, gravity: Vector) -> None:
        # Accumulate forces
        body.add_force(vec_scale(gravity, body.mass))
        total_force = (0.0, 0.0, 0.0)
        for force in body.forces:
            total_force = vec_add(total_force, force)
        acceleration = vec_scale(total_force, 1.0 / body.mass)

        # Linear integration (semi-implicit Euler)
        body.velocity = vec_add(
            vec_scale(body.velocity, 1.0 - body.linear_damping),
            vec_scale(acceleration, dt),
        )
        body.position = vec_add(body.position, vec_scale(body.velocity, dt))

        # Angular integration (torque ignored unless user adds one)
        total_torque = (0.0, 0.0, 0.0)
        for torque in body.torques:
            total_torque = vec_add(total_torque, torque)
        inertia = body.inertia_tensor_diag
        angular_acc = (
            total_torque[0] / inertia[0] if inertia[0] != 0.0 else 0.0,
            total_torque[1] / inertia[1] if inertia[1] != 0.0 else 0.0,
            total_torque[2] / inertia[2] if inertia[2] != 0.0 else 0.0,
        )
        body.angular_velocity = vec_add(
            vec_scale(body.angular_velocity, 1.0 - body.angular_damping),
            vec_scale(angular_acc, dt),
        )
        body.orientation = quat_integrate(body.orientation, body.angular_velocity, dt)

        self._resolve_ground_contact(body)
        body.clear_accumulators()

    def export_history(self, path: Path) -> None:
        data = {
            "dt": self.config.dt,
            "frames": self.history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def export_obj_sequence(self, directory: Path, prefix: str = "frame") -> None:
        if not self.history:
            raise ValueError("No frames recorded, run the simulation first")

        directory.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in enumerate(self.history, start=1):
            lines = [
                "# OBJ sequence generated by rigid_body_sim.py",
                f"o {prefix}_{frame_idx:04d}",
            ]
            vertex_offset = 0
            for body_state in frame["bodies"]:
                name = body_state["name"]  # type: ignore[index]
                shape = self._body_shapes.get(name)
                if shape is None:
                    continue
                orientation = cast(Quaternion, tuple(body_state["orientation"]))  # type: ignore[arg-type]
                position = cast(Vector, tuple(body_state["position"]))  # type: ignore[arg-type]
                rotation = quat_to_matrix(orientation)
                lines.append(f"g {name}")
                transformed = shape.world_vertices(rotation, position)
                for v in transformed:
                    lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
                for tri in shape.triangles:
                    indices = [vertex_offset + idx + 1 for idx in tri]
                    lines.append("f " + " ".join(str(i) for i in indices))
                vertex_offset += len(transformed)

            file_path = directory / f"{prefix}_{frame_idx:04d}.obj"
            file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _record_frame(self) -> None:
        frame_data = {
            "time": self.time,
            "bodies": [
                {
                    "name": body.name,
                    "position": body.position,
                    "orientation": body.orientation,
                }
                for body in self.bodies
            ],
        }
        self.history.append(frame_data)

    def add_impulse_event(self, event: ImpulseEvent) -> None:
        self._schedule_impulse(event)

    def _schedule_impulse(self, event: ImpulseEvent) -> None:
        heapq.heappush(self._pending_impulses, (event.time, self._impulse_seq, event))
        self._impulse_seq += 1

    def _apply_scheduled_impulses(self, window_end: float) -> None:
        eps = 1e-9
        while self._pending_impulses and self._pending_impulses[0][0] <= window_end + eps:
            _, _, event = heapq.heappop(self._pending_impulses)
            body = self._body_lookup.get(event.body_name)
            if body is None or body.mass <= 0.0:
                continue
            contact_offset: Vector | None = None
            if event.local_offset is not None:
                rotation = quat_to_matrix(body.orientation)
                contact_offset = mat3_mul_vec(rotation, event.local_offset)
            self._apply_impulse(body, event.impulse, contact_offset)

    def _apply_impulse(self, body: RigidBody, impulse: Vector, contact_offset: Vector | None) -> None:
        if body.mass <= 0.0:
            return
        body.velocity = vec_add(body.velocity, vec_scale(impulse, 1.0 / body.mass))
        if contact_offset is not None:
            angular_delta = self._apply_inverse_inertia(body, vec_cross(contact_offset, impulse))
            body.angular_velocity = vec_add(body.angular_velocity, angular_delta)

    def _resolve_ground_contact(self, body: RigidBody) -> None:
        assert body.shape is not None
        # Coarse bounding sphere check using the smallest half-extent.
        min_half_extent = body.shape.min_half_extent()
        penetration = (body.position[1] - min_half_extent) - self.config.ground_height
        if penetration >= 0:
            return

        corrected_y = self.config.ground_height + min_half_extent
        body.position = (body.position[0], corrected_y, body.position[2])
        normal = (0.0, 1.0, 0.0)
        contact_offset = self._estimate_contact_offset(body, min_half_extent)

        # Check for stable resting contact: multiple points close to ground with low velocities
        rotation = quat_to_matrix(body.orientation)
        world_vertices = body.shape.world_vertices(rotation, body.position)
        contact_threshold = self.config.ground_height + 1e-3
        contact_points = []
        
        for vertex in world_vertices:
            if vertex[1] <= contact_threshold:
                # Compute velocity at this vertex
                offset_from_com = vec_sub(vertex, body.position)
                vertex_velocity = vec_add(body.velocity, vec_cross(body.angular_velocity, offset_from_com))
                vertex_normal_speed = vec_dot(vertex_velocity, normal)
                
                if abs(vertex_normal_speed) < self.config.resting_speed_epsilon:
                    contact_points.append((vertex, vertex_normal_speed))
        
        # Determine if this is a resting contact (3+ stable contact points)
        is_resting = len(contact_points) >= 3
        
        # Compute contact point velocity including angular contribution
        contact_velocity = vec_add(body.velocity, vec_cross(body.angular_velocity, contact_offset))
        normal_speed = vec_dot(contact_velocity, normal)
        
        normal_impulse_mag = 0.0
        
        if is_resting:
            
            # For resting contacts: kill vertical velocity and apply strong damping
            body.velocity = (body.velocity[0], 0.2 * body.velocity[1], body.velocity[2])
            
            # # Apply enhanced damping to horizontal motion and rotation when resting
            # resting_linear_damping = 0.1
            # resting_angular_damping = 0.1
            # normalize the velocity and then scale it down according to the ground's friction
            body.velocity = vec_scale(body.velocity, 1.0)
            body.angular_velocity = vec_scale(body.angular_velocity, 0.7)
        else:
            # For collisions: apply impulse with restitution
            if normal_speed < 0:
                r_cross_n = vec_cross(contact_offset, normal)
                angular_term = vec_dot(vec_cross(self._apply_inverse_inertia(body, r_cross_n), contact_offset), normal)
                impulse_denom = (1.0 / body.mass) + angular_term
                if impulse_denom > 1e-6:
                    # Only use restitution for significant impacts
                    restitution = body.restitution if abs(normal_speed) > 1.0 else 0.0
                    jn = -(1.0 + restitution) * normal_speed / impulse_denom
                    normal_impulse_mag = jn
                    impulse_normal = vec_scale(normal, jn)
                    body.velocity = vec_add(body.velocity, vec_scale(impulse_normal, 1.0 / body.mass))
                    body.angular_velocity = vec_add(
                        body.angular_velocity,
                        self._apply_inverse_inertia(body, vec_cross(contact_offset, impulse_normal)),
                    )
            
            # Ensure no penetration velocity remains
            body.velocity = (body.velocity[0], max(body.velocity[1], 0.0), body.velocity[2])
        
        # Apply friction impulse only if we had a significant normal impulse
        if normal_impulse_mag > 1e-1:
            contact_velocity = vec_add(body.velocity, vec_cross(body.angular_velocity, contact_offset))
            tangent_velocity = vec_sub(contact_velocity, vec_scale(normal, vec_dot(contact_velocity, normal)))
            tangent_speed = vec_length(tangent_velocity)
            
            if tangent_speed > 1e-6:
                tangent_dir = vec_scale(tangent_velocity, -1.0 / tangent_speed)
                r_cross_t = vec_cross(contact_offset, tangent_dir)
                angular_term_t = vec_dot(vec_cross(self._apply_inverse_inertia(body, r_cross_t), contact_offset), tangent_dir)
                friction_denom = (1.0 / body.mass) + angular_term_t
                
                if friction_denom > 1e-6:
                    # Coulomb friction: friction impulse is limited by normal impulse
                    max_friction_impulse = self.config.ground_friction * normal_impulse_mag
                    friction_impulse_mag = min(tangent_speed / friction_denom, max_friction_impulse)
                    friction_impulse = vec_scale(tangent_dir, friction_impulse_mag)
                    
                    body.velocity = vec_add(body.velocity, vec_scale(friction_impulse, 1.0 / body.mass))
                    body.angular_velocity = vec_add(
                        body.angular_velocity,
                        self._apply_inverse_inertia(body, vec_cross(contact_offset, friction_impulse)),
                    )

    def _estimate_contact_offset(self, body: RigidBody, fallback_half_extent: float) -> Vector:
        assert body.shape is not None
        rotation = quat_to_matrix(body.orientation)
        world_vertices = body.shape.world_vertices(rotation, body.position)
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
        body_count = len(self.bodies)
        if body_count < 2:
            return

        rotation_cache: Dict[str, Matrix3] = {body.name: quat_to_matrix(body.orientation) for body in self.bodies}
        tasks: List[Tuple[RigidBody, RigidBody, Matrix3, Matrix3]] = []
        for i in range(body_count):
            body_a = self.bodies[i]
            for j in range(i + 1, body_count):
                body_b = self.bodies[j]
                if not self._broad_phase_overlap(body_a, body_b):
                    continue
                tasks.append((body_a, body_b, rotation_cache[body_a.name], rotation_cache[body_b.name]))

        if not tasks:
            return

        use_parallel = self.config.parallel_collisions and len(tasks) > 1
        executor = self._get_executor() if use_parallel else None

        if executor is None:
            for body_a, body_b, rotation_a, rotation_b in tasks:
                collision = self._sat_test_convex(body_a, body_b, rotation_a, rotation_b)
                if collision is None:
                    continue
                self._resolve_narrow_phase(body_a, body_b, rotation_a, rotation_b, collision)
        else:
            futures: List[Tuple[Tuple[RigidBody, RigidBody, Matrix3, Matrix3], Future[Tuple[Vector, float] | None]]] = []
            for body_a, body_b, rotation_a, rotation_b in tasks:
                assert body_a.shape is not None and body_b.shape is not None
                vertices_a = body_a.shape.world_vertices(rotation_a, body_a.position)
                vertices_b = body_b.shape.world_vertices(rotation_b, body_b.position)
                face_normals_a = body_a.shape.world_face_normals(rotation_a)
                face_normals_b = body_b.shape.world_face_normals(rotation_b)
                edge_dirs_a = body_a.shape.world_edge_dirs(rotation_a)
                edge_dirs_b = body_b.shape.world_edge_dirs(rotation_b)
                
                future = executor.submit(
                    _sat_test_worker,
                    vertices_a,
                    vertices_b,
                    face_normals_a,
                    face_normals_b,
                    edge_dirs_a,
                    edge_dirs_b,
                    body_a.position,
                    body_b.position,
                )
                futures.append(((body_a, body_b, rotation_a, rotation_b), future))
            for (body_a, body_b, rotation_a, rotation_b), future in futures:
                collision = future.result()
                if collision is None:
                    continue
                self._resolve_narrow_phase(body_a, body_b, rotation_a, rotation_b, collision)

    def _resolve_narrow_phase(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        rotation_a: Matrix3,
        rotation_b: Matrix3,
        collision: Tuple[Vector, float],
    ) -> None:
        normal, penetration = collision
        assert body_a.shape is not None and body_b.shape is not None
        contact_point_a = body_a.shape.support_point_with_rotation(rotation_a, body_a.position, normal)
        contact_point_b = body_b.shape.support_point_with_rotation(rotation_b, body_b.position, vec_scale(normal, -1.0))
        contact_offset_a = vec_sub(contact_point_a, body_a.position)
        contact_offset_b = vec_sub(contact_point_b, body_b.position)

        self._apply_positional_correction(body_a, body_b, normal, penetration)
        self._apply_collision_impulse(body_a, body_b, normal, contact_offset_a, contact_offset_b)

    def _broad_phase_overlap(self, body_a: RigidBody, body_b: RigidBody) -> bool:
        if body_a.mass <= 0.0 and body_b.mass <= 0.0:
            return False
        radius = body_a.bounding_radius + body_b.bounding_radius + self.config.broad_phase_margin
        delta = vec_sub(body_b.position, body_a.position)
        return vec_dot(delta, delta) <= radius * radius

    def _apply_collision_impulse(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        normal: Vector,
        contact_offset_a: Vector,
        contact_offset_b: Vector,
    ) -> None:

        vel_a = vec_add(body_a.velocity, vec_cross(body_a.angular_velocity, contact_offset_a))
        vel_b = vec_add(body_b.velocity, vec_cross(body_b.angular_velocity, contact_offset_b))
        relative_velocity = vec_sub(vel_b, vel_a)
        normal_speed = vec_dot(relative_velocity, normal)
        if normal_speed >= 0.0:
            return

        inv_mass_a = 1.0 / body_a.mass if body_a.mass > 0.0 else 0.0
        inv_mass_b = 1.0 / body_b.mass if body_b.mass > 0.0 else 0.0
        raxn = vec_cross(contact_offset_a, normal)
        rbxn = vec_cross(contact_offset_b, normal)
        angular_term = vec_dot(
            vec_cross(self._apply_inverse_inertia(body_a, raxn), contact_offset_a),
            normal,
        )
        angular_term += vec_dot(
            vec_cross(self._apply_inverse_inertia(body_b, rbxn), contact_offset_b),
            normal,
        )
        denom = inv_mass_a + inv_mass_b + angular_term
        if denom < 1e-6:
            return

        restitution = max(0.0, min(body_a.restitution, body_b.restitution))
        jn = -(1.0 + restitution) * normal_speed / denom
        impulse = vec_scale(normal, jn)

        body_a.velocity = vec_sub(body_a.velocity, vec_scale(impulse, inv_mass_a))
        body_b.velocity = vec_add(body_b.velocity, vec_scale(impulse, inv_mass_b))
        body_a.angular_velocity = vec_sub(
            body_a.angular_velocity,
            self._apply_inverse_inertia(body_a, vec_cross(contact_offset_a, impulse)),
        )
        body_b.angular_velocity = vec_add(
            body_b.angular_velocity,
            self._apply_inverse_inertia(body_b, vec_cross(contact_offset_b, impulse)),
        )

    def _compute_worker_target(self) -> int:
        if self.config.max_workers is not None:
            return max(1, self.config.max_workers)
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count)

    def _get_executor(self) -> ProcessPoolExecutor | None:
        worker_cap = max(1, min(self._worker_target, len(self.bodies))) if self.bodies else self._worker_target
        print(worker_cap)
        worker_cap = 8
        if worker_cap <= 1:
            return None
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=worker_cap)
        return self._executor

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.shutdown()
        except Exception:
            pass

    def _apply_positional_correction(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        normal: Vector,
        penetration: float,
    ) -> None:
        inv_mass_a = 1.0 / body_a.mass if body_a.mass > 0.0 else 0.0
        inv_mass_b = 1.0 / body_b.mass if body_b.mass > 0.0 else 0.0
        inv_mass_sum = inv_mass_a + inv_mass_b
        if inv_mass_sum == 0.0:
            return

        correction_percent = 0.8
        slop = 1e-3
        correction_mag = max(penetration - slop, 0.0) * correction_percent / inv_mass_sum
        if correction_mag <= 0.0:
            return

        correction = vec_scale(normal, correction_mag)
        body_a.position = vec_sub(body_a.position, vec_scale(correction, inv_mass_a))
        body_b.position = vec_add(body_b.position, vec_scale(correction, inv_mass_b))

    def _sat_test_convex(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        rotation_a: Matrix3,
        rotation_b: Matrix3,
    ) -> Tuple[Vector, float] | None:
        if body_a.shape is None or body_b.shape is None:
            return None

        vertices_a = body_a.shape.world_vertices(rotation_a, body_a.position)
        vertices_b = body_b.shape.world_vertices(rotation_b, body_b.position)

        axes: List[Vector] = []
        axes.extend(body_a.shape.world_face_normals(rotation_a))
        axes.extend(body_b.shape.world_face_normals(rotation_b))
        for edge_a in body_a.shape.world_edge_dirs(rotation_a):
            for edge_b in body_b.shape.world_edge_dirs(rotation_b):
                axis = vec_cross(edge_a, edge_b)
                if vec_length(axis) < 1e-6:
                    continue
                axes.append(vec_normalize(axis))

        translation = vec_sub(body_b.position, body_a.position)
        best_axis: Vector | None = None
        min_overlap = math.inf

        for axis in axes:
            norm_axis = vec_normalize(axis)
            if norm_axis == (0.0, 0.0, 0.0):
                continue
            min_a, max_a = _project_vertices(vertices_a, norm_axis)
            min_b, max_b = _project_vertices(vertices_b, norm_axis)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap <= 0.0:
                return None
            if overlap < min_overlap:
                direction_hint = vec_dot(norm_axis, translation)
                best_axis = norm_axis if direction_hint >= 0.0 else vec_scale(norm_axis, -1.0)
                min_overlap = overlap

        if best_axis is None or min_overlap is math.inf:
            return None

        return best_axis, min_overlap

def _project_vertices(vertices: Sequence[Vector], axis: Vector) -> Tuple[float, float]:
    projections = [vec_dot(vertex, axis) for vertex in vertices]
    return min(projections), max(projections)


def _integrate_body_worker(
    position: Vector,
    velocity: Vector,
    orientation: Quaternion,
    angular_velocity: Vector,
    mass: float,
    inertia: Vector,
    linear_damping: float,
    angular_damping: float,
    dt: float,
    gravity: Vector,
) -> Tuple[Vector, Vector, Quaternion, Vector]:
    """Worker function for parallel body integration."""
    # Apply gravity
    total_force = vec_scale(gravity, mass)
    acceleration = vec_scale(total_force, 1.0 / mass)
    
    # Linear integration
    velocity = vec_add(
        vec_scale(velocity, 1.0 - linear_damping),
        vec_scale(acceleration, dt),
    )
    position = vec_add(position, vec_scale(velocity, dt))
    
    # Angular integration (no external torques in this version)
    angular_velocity = vec_scale(angular_velocity, 1.0 - angular_damping)
    orientation = quat_integrate(orientation, angular_velocity, dt)
    
    return position, velocity, orientation, angular_velocity


def _sat_test_worker(
    vertices_a: List[Vector],
    vertices_b: List[Vector],
    face_normals_a: List[Vector],
    face_normals_b: List[Vector],
    edge_dirs_a: List[Vector],
    edge_dirs_b: List[Vector],
    position_a: Vector,
    position_b: Vector,
) -> Tuple[Vector, float] | None:
    """Worker function for parallel SAT collision detection."""
    axes: List[Vector] = []
    axes.extend(face_normals_a)
    axes.extend(face_normals_b)
    for edge_a in edge_dirs_a:
        for edge_b in edge_dirs_b:
            axis = vec_cross(edge_a, edge_b)
            if vec_length(axis) < 1e-6:
                continue
            axes.append(vec_normalize(axis))
    
    translation = vec_sub(position_b, position_a)
    best_axis: Vector | None = None
    min_overlap = math.inf
    
    for axis in axes:
        norm_axis = vec_normalize(axis)
        if norm_axis == (0.0, 0.0, 0.0):
            continue
        min_a, max_a = _project_vertices(vertices_a, norm_axis)
        min_b, max_b = _project_vertices(vertices_b, norm_axis)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap <= 0.0:
            return None
        if overlap < min_overlap:
            direction_hint = vec_dot(norm_axis, translation)
            best_axis = norm_axis if direction_hint >= 0.0 else vec_scale(norm_axis, -1.0)
            min_overlap = overlap
    
    if best_axis is None or min_overlap is math.inf:
        return None
    
    return best_axis, min_overlap


def demo_scene() -> List[RigidBody]:
    bodies = []
    mesh_path = Path("../../assets/meshes/convex_gem.obj")
    if mesh_path.exists():
        gem_shape = load_convex_mesh(mesh_path, scale=2.00)
        bodies.append(
            RigidBody(
                name="Gem_1",
                mass=1.2,
                size=None,
                shape=gem_shape,
                position=(-0.2, 5.0, -0.5),
                velocity=(0.4, -0.2, 0.5),
                angular_velocity=(0.8, 2.5, -0.3),
                restitution=0.1,
            )
        )
    
    mesh_path = Path("../../assets/meshes/bunny.obj")
    if mesh_path.exists():
        bodies.append(
            RigidBody(
                name="CrateA",
                mass=3.0,
                size=None,
                shape=load_convex_mesh(mesh_path, scale=0.5),
                position=(0.0, 8.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                restitution=0.05,
            )
        )
    
    return bodies


def main() -> None:
    start_time = time.perf_counter()
    config = SimulationConfig(duration=5.0, dt=1.0 / 48)
    impulses = [
        ImpulseEvent(time=0.2, body_name="CrateA", impulse=(5.0, 0.0, 3.0)),
        # ImpulseEvent(time=0.5, body_name="CrateB", impulse=(2.0, 1.0, 1.5), local_offset=(0.0, 0.3, 0.0)),
    ]
    sim = RigidBodySimulation(demo_scene(), config, impulses=impulses)
    try:
        sim.run()
        output_dir = Path("outputs_24/temp")
        sim.export_obj_sequence(output_dir)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(f"Stored {len(sim.history)} OBJ frames under {output_dir} in {elapsed_ms:.2f} ms")
    finally:
        sim.shutdown()


if __name__ == "__main__":
    main()

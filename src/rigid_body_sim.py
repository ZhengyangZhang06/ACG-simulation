"""Simple rigid body simulation core.

This module implements a tiny rigid body simulator tailored for exporting
animation data that Blender can play back.  It avoids any high-level
physics libraries and instead relies on explicit integration and a couple of
vector helpers implemented from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, cast
import json
import math

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


@dataclass
class RigidBody:
    name: str
    mass: float
    size: Vector  # box dimensions (width, height, depth)
    position: Vector
    velocity: Vector
    orientation: Quaternion = (1.0, 0.0, 0.0, 0.0)
    angular_velocity: Vector = (0.0, 0.0, 0.0)
    restitution: float = 0.35
    linear_damping: float = 0.01
    angular_damping: float = 0.02
    forces: List[Vector] = field(default_factory=list)
    torques: List[Vector] = field(default_factory=list)

    def add_force(self, force: Vector) -> None:
        self.forces.append(force)

    def add_torque(self, torque: Vector) -> None:
        self.torques.append(torque)

    @property
    def inertia_tensor_diag(self) -> Vector:
        w, h, d = self.size
        i_x = (1.0 / 12.0) * self.mass * (h * h + d * d)
        i_y = (1.0 / 12.0) * self.mass * (w * w + d * d)
        i_z = (1.0 / 12.0) * self.mass * (w * w + h * h)
        return (i_x, i_y, i_z)

    def clear_accumulators(self) -> None:
        self.forces.clear()
        self.torques.clear()


@dataclass
class SimulationConfig:
    duration: float = 5.0
    dt: float = 1.0 / 120.0
    gravity: Vector = (0.0, -9.81, 0.0)
    ground_height: float = 0.0
    ground_friction: float = 0.4


class RigidBodySimulation:
    def __init__(self, bodies: Iterable[RigidBody], config: SimulationConfig | None = None) -> None:
        self.bodies = list(bodies)
        self.config = config or SimulationConfig()
        self.time = 0.0
        self.history: List[Dict[str, object]] = []
        self._body_sizes: Dict[str, Vector] = {body.name: body.size for body in self.bodies}
        if len(self._body_sizes) != len(self.bodies):
            raise ValueError("Rigid body names must be unique")

    def step(self) -> None:
        dt = self.config.dt
        gravity = self.config.gravity

        for body in self.bodies:
            # Accumulate forces
            body.add_force(vec_scale(gravity, body.mass))

            # Linear integration (semi-implicit Euler)
            total_force = (0.0, 0.0, 0.0)
            for force in body.forces:
                total_force = vec_add(total_force, force)
            acceleration = vec_scale(total_force, 1.0 / body.mass)

            body.velocity = vec_add(vec_scale(body.velocity, 1.0 - body.linear_damping), vec_scale(acceleration, dt))
            body.position = vec_add(body.position, vec_scale(body.velocity, dt))

            # Angular integration (torque ignored unless user adds one)
            total_torque = (0.0, 0.0, 0.0)
            for torque in body.torques:
                total_torque = vec_add(total_torque, torque)
            inertia = body.inertia_tensor_diag
            angular_acc = (
                total_torque[0] / inertia[0],
                total_torque[1] / inertia[1],
                total_torque[2] / inertia[2],
            )
            body.angular_velocity = vec_add(vec_scale(body.angular_velocity, 1.0 - body.angular_damping), vec_scale(angular_acc, dt))
            body.orientation = quat_integrate(body.orientation, body.angular_velocity, dt)

            self._resolve_ground_contact(body)
            body.clear_accumulators()

        self.time += dt
        self._record_frame()

    def run(self) -> None:
        steps = int(self.config.duration / self.config.dt)
        for _ in range(steps):
            self.step()

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
                size = self._body_sizes.get(name)
                if size is None:
                    continue
                orientation = cast(Quaternion, tuple(body_state["orientation"]))  # type: ignore[arg-type]
                position = cast(Vector, tuple(body_state["position"]))  # type: ignore[arg-type]
                rotation = quat_to_matrix(orientation)
                local_vertices = box_vertices(size)
                lines.append(f"g {name}")
                transformed = [vec_add(mat3_mul_vec(rotation, vertex), position) for vertex in local_vertices]
                for v in transformed:
                    lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
                for tri in BOX_TRIANGLES:
                    indices = [vertex_offset + idx for idx in tri]
                    lines.append("f " + " ".join(str(i) for i in indices))
                vertex_offset += len(local_vertices)

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

    @staticmethod
    def _inverse_inertia_apply(body: RigidBody, torque: Vector, rotation: Matrix3) -> Vector:
        inertia = body.inertia_tensor_diag
        inv_inertia = (
            0.0 if inertia[0] == 0.0 else 1.0 / inertia[0],
            0.0 if inertia[1] == 0.0 else 1.0 / inertia[1],
            0.0 if inertia[2] == 0.0 else 1.0 / inertia[2],
        )
        torque_local = mat3_mul_vec_transpose(rotation, torque)
        local_response = (
            torque_local[0] * inv_inertia[0],
            torque_local[1] * inv_inertia[1],
            torque_local[2] * inv_inertia[2],
        )
        return mat3_mul_vec(rotation, local_response)

    @staticmethod
    def _apply_impulse(body: RigidBody, impulse: Vector, contact_offset: Vector, rotation: Matrix3) -> None:
        inv_mass = 1.0 / body.mass
        body.velocity = vec_add(body.velocity, vec_scale(impulse, inv_mass))
        angular_impulse = vec_cross(contact_offset, impulse)
        angular_delta = RigidBodySimulation._inverse_inertia_apply(body, angular_impulse, rotation)
        body.angular_velocity = vec_add(body.angular_velocity, angular_delta)

    def _resolve_ground_contact(self, body: RigidBody) -> None:
        normal = (0.0, 1.0, 0.0)
        rotation = quat_to_matrix(body.orientation)
        local_vertices = box_vertices(body.size)
        contact_offset: Vector | None = None
        lowest_world_y = float("inf")

        for vertex in local_vertices:
            world_offset = mat3_mul_vec(rotation, vertex)
            world_y = body.position[1] + world_offset[1]
            if world_y < lowest_world_y:
                lowest_world_y = world_y
                contact_offset = world_offset

        if contact_offset is None:
            return

        penetration = self.config.ground_height - lowest_world_y
        if penetration <= 0.0:
            return

        body.position = vec_add(body.position, (0.0, penetration, 0.0))

        contact_velocity = vec_add(body.velocity, vec_cross(body.angular_velocity, contact_offset))
        v_rel_n = vec_dot(contact_velocity, normal)
        inv_mass = 1.0 / body.mass

        j_n = 0.0
        if v_rel_n < 0.0:
            r_cross_n = vec_cross(contact_offset, normal)
            inv_inertia_term = self._inverse_inertia_apply(body, r_cross_n, rotation)
            denom = inv_mass + vec_dot(normal, vec_cross(inv_inertia_term, contact_offset))
            if denom > 1e-8:
                j_n = -(1.0 + body.restitution) * v_rel_n / denom
                normal_impulse = vec_scale(normal, j_n)
                self._apply_impulse(body, normal_impulse, contact_offset, rotation)

        if j_n > 0.0:
            updated_contact_velocity = vec_add(body.velocity, vec_cross(body.angular_velocity, contact_offset))
            normal_component = vec_scale(normal, vec_dot(updated_contact_velocity, normal))
            tangential_velocity = vec_sub(updated_contact_velocity, normal_component)
            tangential_speed = vec_length(tangential_velocity)
            if tangential_speed > 1e-5:
                tangent_dir = vec_scale(tangential_velocity, 1.0 / tangential_speed)
                r_cross_t = vec_cross(contact_offset, tangent_dir)
                inv_inertia_t = self._inverse_inertia_apply(body, r_cross_t, rotation)
                denom_t = inv_mass + vec_dot(tangent_dir, vec_cross(inv_inertia_t, contact_offset))
                if denom_t > 1e-8:
                    jt = -vec_dot(updated_contact_velocity, tangent_dir) / denom_t
                    max_friction = self.config.ground_friction * j_n
                    jt = max(-max_friction, min(jt, max_friction))
                    friction_impulse = vec_scale(tangent_dir, jt)
                    self._apply_impulse(body, friction_impulse, contact_offset, rotation)


def demo_scene() -> List[RigidBody]:
    bodies = []
    bodies.append(
        RigidBody(
            name="CrateA",
            mass=2.0,
            size=(0.5, 0.5, 0.5),
            position=(0.0, 2.0, 0.0),
            velocity=(1.5, 0.0, 0.0),
            angular_velocity=(0.0, 3.0, 0.0),
        )
    )
    bodies.append(
        RigidBody(
            name="CrateB",
            mass=1.5,
            size=(0.4, 0.6, 0.4),
            position=(-0.5, 3.5, 0.5),
            velocity=(0.5, 0.0, -0.2),
            angular_velocity=(1.0, 0.0, 2.0),
        )
    )
    bodies.append(
        RigidBody(
            name="CrateC",
            mass=0.8,
            size=(0.3, 0.3, 0.7),
            position=(0.8, 4.0, -0.3),
            velocity=(-0.2, 0.0, 0.6),
            angular_velocity=(0.0, 1.8, 0.0),
        )
    )
    return bodies


def main() -> None:
    config = SimulationConfig(duration=6.0, dt=1.0 / 120.0)
    sim = RigidBodySimulation(demo_scene(), config)
    sim.run()
    output_dir = Path("output/obj_frames")
    sim.export_obj_sequence(output_dir)
    print(f"Stored {len(sim.history)} OBJ frames under {output_dir}")


if __name__ == "__main__":
    main()

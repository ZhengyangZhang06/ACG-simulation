#!/usr/bin/env python3
"""Split multi-object OBJ frames into per-object files per frame directory."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Face = List[Tuple[int, Optional[int], Optional[int]]]


@dataclass
class ObjectData:
    name: str
    faces: List[Face] = field(default_factory=list)
    meta_lines: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform a directory of frame OBJ files (each containing multiple "
            "objects) into a directory tree where each frame gets its own "
            "subdirectory of per-object OBJ files."
        )
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing frame_XXXX.obj files"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory where split OBJ files will be written"
    )
    parser.add_argument(
        "--pattern",
        default="frame_*.obj",
        help="Glob pattern used to find frame files (default: frame_*.obj)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing per-object OBJ files",
    )
    return parser.parse_args()


def resolve_index(raw_idx: int, total: int) -> int:
    if raw_idx > 0:
        return raw_idx
    # Negative OBJ indices are relative to the most recently defined element.
    resolved = total + 1 + raw_idx
    if resolved <= 0:
        raise ValueError(f"Invalid OBJ index {raw_idx} for total {total}")
    return resolved


def parse_face_token(
    token: str, vertex_count: int, texcoord_count: int, normal_count: int
) -> Tuple[int, Optional[int], Optional[int]]:
    parts = token.split("/")
    v_idx = int(parts[0]) if parts[0] else 0
    vt_idx: Optional[int] = None
    vn_idx: Optional[int] = None

    if len(parts) > 1 and parts[1]:
        vt = int(parts[1])
        vt_idx = resolve_index(vt, texcoord_count)
    if len(parts) > 2 and parts[2]:
        vn = int(parts[2])
        vn_idx = resolve_index(vn, normal_count)

    if v_idx == 0:
        raise ValueError(f"Malformed face token '{token}'")

    v_idx = resolve_index(v_idx, vertex_count)
    return v_idx, vt_idx, vn_idx


def sanitize_object_name(name: str, fallback_index: int, used: Dict[str, int]) -> str:
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip()) or f"object_{fallback_index:03d}"
    count = used.get(slug, 0)
    used[slug] = count + 1
    if count:
        return f"{slug}_{count+1}"
    return slug


def extract_frame_name(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return match.group(1)
    return path.stem


def read_frame(path: Path) -> Tuple[List[str], List[str], List[str], Dict[str, ObjectData], List[str]]:
    vertices: List[str] = []
    texcoords: List[str] = []
    normals: List[str] = []
    objects: Dict[str, ObjectData] = {}
    current_obj: Optional[str] = None
    mtllib_lines: List[str] = []

    def ensure_object(name: str) -> ObjectData:
        nonlocal current_obj
        if name not in objects:
            objects[name] = ObjectData(name=name)
        current_obj = name
        return objects[name]

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("mtllib"):
            mtllib_lines.append(line)
            continue
        if line.startswith("v "):
            vertices.append(line)
            continue
        if line.startswith("vt "):
            texcoords.append(line)
            continue
        if line.startswith("vn "):
            normals.append(line)
            continue
        if line.startswith("g "):
            parts = line.split()
            name = parts[1] if len(parts) > 1 else "default"
            ensure_object(name)
            continue
        if line.startswith("o "):
            # Frame-level object tag; keep but do not treat as per-object identifier.
            continue
        if line.startswith("usemtl") or line.startswith("s "):
            obj = ensure_object(current_obj or "default")
            obj.meta_lines.append(line)
            continue
        if line.startswith("f "):
            obj = ensure_object(current_obj or "default")
            tokens = line.split()[1:]
            face = [
                parse_face_token(token, len(vertices), len(texcoords), len(normals))
                for token in tokens
            ]
            obj.faces.append(face)
            continue

    return vertices, texcoords, normals, objects, mtllib_lines


def reindex_faces(
    faces: Sequence[Face],
    vertices: Sequence[str],
    texcoords: Sequence[str],
    normals: Sequence[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    vertex_map: Dict[int, int] = {}
    texcoord_map: Dict[int, int] = {}
    normal_map: Dict[int, int] = {}
    vertex_lines: List[str] = []
    texcoord_lines: List[str] = []
    normal_lines: List[str] = []
    face_lines: List[str] = []

    def map_index(old_idx: int, index_map: Dict[int, int]) -> int:
        if old_idx in index_map:
            return index_map[old_idx]
        index_map[old_idx] = len(index_map) + 1
        return index_map[old_idx]

    for face in faces:
        new_tokens: List[str] = []
        for v_idx, vt_idx, vn_idx in face:
            new_v = map_index(v_idx, vertex_map)
            if new_v > len(vertex_lines):
                vertex_lines.append(vertices[v_idx - 1])

            new_vt: Optional[int] = None
            new_vn: Optional[int] = None
            if vt_idx is not None:
                new_vt = map_index(vt_idx, texcoord_map)
                if new_vt > len(texcoord_lines):
                    texcoord_lines.append(texcoords[vt_idx - 1])
            if vn_idx is not None:
                new_vn = map_index(vn_idx, normal_map)
                if new_vn > len(normal_lines):
                    normal_lines.append(normals[vn_idx - 1])

            if new_vt is None and new_vn is None:
                new_tokens.append(str(new_v))
            elif new_vn is None:
                new_tokens.append(f"{new_v}/{new_vt}")
            elif new_vt is None:
                new_tokens.append(f"{new_v}//{new_vn}")
            else:
                new_tokens.append(f"{new_v}/{new_vt}/{new_vn}")
        face_lines.append("f " + " ".join(new_tokens))

    return vertex_lines, texcoord_lines, normal_lines, face_lines


def write_object_file(
    output_path: Path,
    object_name: str,
    mtllib_lines: Sequence[str],
    meta_lines: Sequence[str],
    vertex_lines: Sequence[str],
    texcoord_lines: Sequence[str],
    normal_lines: Sequence[str],
    face_lines: Sequence[str],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    lines: List[str] = [f"# Extracted object {object_name}"]
    lines.extend(mtllib_lines)
    lines.append(f"o {object_name}")
    lines.append(f"g {object_name}")
    lines.extend(meta_lines)
    lines.extend(vertex_lines)
    lines.extend(texcoord_lines)
    lines.extend(normal_lines)
    lines.extend(face_lines)
    output_path.write_text("\n".join(lines) + "\n")


def process_frame(
    frame_path: Path, output_root: Path, overwrite: bool
) -> Tuple[str, int]:
    vertices, texcoords, normals, objects, mtllib_lines = read_frame(frame_path)
    frame_name = extract_frame_name(frame_path)
    frame_dir = output_root / frame_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    if not objects:
        return frame_name, 0

    used_names: Dict[str, int] = {}
    object_count = 0

    for idx, data in enumerate(objects.values(), start=1):
        if not data.faces:
            continue
        clean_name = sanitize_object_name(data.name, idx, used_names)
        vertex_lines, tex_lines, normal_lines, face_lines = reindex_faces(
            data.faces, vertices, texcoords, normals
        )
        if not face_lines:
            continue
        out_path = frame_dir / f"{clean_name}.obj"
        write_object_file(
            out_path,
            clean_name,
            mtllib_lines,
            data.meta_lines,
            vertex_lines,
            tex_lines,
            normal_lines,
            face_lines,
            overwrite,
        )
        object_count += 1
    return frame_name, object_count


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(input_dir.glob(args.pattern))
    if not frame_files:
        raise FileNotFoundError(
            f"No OBJ frames matching pattern '{args.pattern}' in {input_dir}"
        )

    for frame_path in frame_files:
        frame_name, count = process_frame(frame_path, output_dir, args.overwrite)
        print(f"Frame {frame_name}: wrote {count} object files")


if __name__ == "__main__":
    main()

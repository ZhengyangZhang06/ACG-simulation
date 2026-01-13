"""Import an OBJ frame sequence as shape-key animation inside Blender.

Usage:
1. Open this file inside Blender's Scripting workspace.
2. Update OBJ_DIRECTORY so it points at the folder that contains the exported
   OBJ frames (relative paths can use the "//" prefix to resolve from the
   .blend file location).
3. Run the script. It will create a single mesh object with one shape key per
   frame and insert keyframes so the object plays back through the sequence.
"""
from __future__ import annotations

import bpy
from pathlib import Path
from typing import Iterable, List

# --- Configuration ---------------------------------------------------------
OBJ_DIRECTORY = Path("C://Users//72667//Desktop//2025Fall//ACG//ACG-simulation//output//obj_frames")  # Path to folder with frame_XXXX.obj
FILE_PATTERN = "frame_*.obj"                  # Glob that matches every frame file
OBJECT_NAME = "RigidBodySequence"             # Name of the animated mesh
FRAME_START = 1                               # Timeline frame to start playback
FRAME_STEP = 1                                # Advance this many frames per OBJ
AXIS_FORWARD = "-Z"                           # Match settings used during export
AXIS_UP = "Y"


def resolve_frame_paths(directory: Path, pattern: str) -> List[Path]:
    resolved_dir = (directory if directory.is_absolute() else Path(bpy.path.abspath(str(directory)))).resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"OBJ directory not found: {resolved_dir}")
    frame_paths = sorted(resolved_dir.glob(pattern))
    if not frame_paths:
        raise FileNotFoundError(f"No OBJ files that match '{pattern}' in {resolved_dir}")
    return frame_paths


def deselect_all() -> None:
    bpy.ops.object.select_all(action="DESELECT")


def import_obj(filepath: Path) -> bpy.types.Object:
    deselect_all()
    bpy.ops.wm.obj_import(
        filepath=str(filepath),
        # axis_forward=AXIS_FORWARD,
        # axis_up=AXIS_UP,
        # use_split_objects=False,
        # use_split_groups=False,
        # use_groups_as_vgroups=False,
        # global_clight_size=1.0,
    )
    imported = list(bpy.context.selected_objects)
    if not imported:
        raise RuntimeError(f"Nothing imported from {filepath}")
    obj = imported[0]
    obj.name = f"{OBJECT_NAME}_{filepath.stem}"
    return obj


def ensure_clean_target(name: str) -> None:
    existing = bpy.data.objects.get(name)
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)


def copy_vertices(source: bpy.types.Object, target_key: bpy.types.ShapeKey) -> None:
    if len(source.data.vertices) != len(target_key.data):
        raise ValueError("OBJ frame vertex count differs from the basis mesh")
    for src_vert, dst_vert in zip(source.data.vertices, target_key.data):
        dst_vert.co = src_vert.co


def build_shape_keys(frame_paths: Iterable[Path]) -> bpy.types.Object:
    frame_paths = list(frame_paths)
    if not frame_paths:
        raise ValueError("Frame path list is empty")

    ensure_clean_target(OBJECT_NAME)
    base_obj = import_obj(frame_paths[0])
    base_obj.name = OBJECT_NAME
    bpy.context.view_layer.objects.active = base_obj

    # Create Basis + first-frame key.
    first_key = base_obj.shape_key_add(name="Basis", from_mix=False)
    first_key.name = "Basis"  # Blender enforces this but we rename explicitly.
    frame_key = base_obj.shape_key_add(name=frame_paths[0].stem, from_mix=False)

    for path in frame_paths[1:]:
        imported = import_obj(path)
        new_key = base_obj.shape_key_add(name=path.stem, from_mix=False)
        copy_vertices(imported, new_key)
        bpy.data.objects.remove(imported, do_unlink=True)

    return base_obj


def keyframe_shape_sequence(obj: bpy.types.Object, frame_start: int, frame_step: int) -> None:
    shape_keys = obj.data.shape_keys
    if shape_keys is None or len(shape_keys.key_blocks) <= 1:
        raise RuntimeError("Object must have at least one non-basis shape key")

    animatable_keys = [key for key in shape_keys.key_blocks if key.name != "Basis"]
    scene = bpy.context.scene
    frame = frame_start

    for block in animatable_keys:
        scene.frame_set(frame)
        for candidate in animatable_keys:
            candidate.value = 1.0 if candidate == block else 0.0
            candidate.keyframe_insert(data_path="value")
        frame += frame_step

    scene.frame_start = frame_start
    scene.frame_end = frame - frame_step


def main() -> None:
    frame_paths = resolve_frame_paths(OBJ_DIRECTORY, FILE_PATTERN)
    obj = build_shape_keys(frame_paths)
    keyframe_shape_sequence(obj, FRAME_START, FRAME_STEP)
    print(f"Imported {len(frame_paths)} OBJ frames into '{obj.name}' as shape keys")


if __name__ == "__main__":
    main()

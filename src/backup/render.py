
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import bpy 
from bpy.props import StringProperty  
from bpy.types import Operator  


def get_object(path: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """Load OBJ file without any coordinate transformation."""
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                indices = [int(part.split("/")[0]) - 1 for part in line.strip().split()[1:]]
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))
    return verts, faces


def make_mesh(obj_path: Path, mesh_name: str) -> bpy.types.Mesh:
    """Build a Blender mesh from an OBJ file."""
    verts, faces = get_object(obj_path)
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh

def make_frame(frames_dir: Path) -> Tuple[List[str], Dict[str, List[bpy.types.Mesh]]]:

    frame_dirs = sorted(
        [d for d in frames_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.name)
    )

    
    first_frame = frame_dirs[0]
    obj_files = sorted(first_frame.glob("*.obj"))
    
    object_names = [f.stem for f in obj_files]
    
    mesh_sequences: Dict[str, List[bpy.types.Mesh]] = {name: [] for name in object_names}
    
    for frame_idx, frame_dir in enumerate(frame_dirs):
        for obj_name in object_names:
            obj_path = frame_dir / f"{obj_name}.obj"
            if obj_path.exists():
                mesh = make_mesh(obj_path, f"{obj_name}_frame{frame_idx:05d}")
                mesh_sequences[obj_name].append(mesh)
            else:
                if mesh_sequences[obj_name]:
                    mesh_sequences[obj_name].append(mesh_sequences[obj_name][-1])
                else:
                    empty_mesh = bpy.data.meshes.new(f"{obj_name}_empty")
                    mesh_sequences[obj_name].append(empty_mesh)
    
    return object_names, mesh_sequences


def ensure_object(obj_name: str, mesh_data: bpy.types.Mesh) -> bpy.types.Object:
    """Get or create a Blender object with the given name."""
    scene = bpy.context.scene
    
    blender_name = f"SPH_{obj_name}"
    
    obj = bpy.data.objects.get(blender_name)
    if obj is None:
        obj = bpy.data.objects.new(blender_name, mesh_data)
        scene.collection.objects.link(obj)
    else:
        obj.data = mesh_data
    return obj


def register_frame_handler(
    object_names: List[str],
    mesh_sequences: Dict[str, List[bpy.types.Mesh]],
    objects: Dict[str, bpy.types.Object]
) -> None:
    """Register a frame change handler to update meshes on each frame."""
    frame_count = len(next(iter(mesh_sequences.values())))
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count

    def _handler(scene):  
        idx = min(max(scene.frame_current - 1, 0), frame_count - 1)
        for obj_name in object_names:
            if obj_name in objects and obj_name in mesh_sequences:
                objects[obj_name].data = mesh_sequences[obj_name][idx]
    
    _handler.__sph_handler__ = True 

    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre 
        if not getattr(h, "__sph_handler__", False)
    ]
    bpy.app.handlers.frame_change_pre.append(_handler)


class SPH_OT_load_surface_animation(Operator):
    """Load OBJ surface frames and hook them to the timeline."""
    
    bl_idname = "sph.load_surface_animation"
    bl_label = "Load SPH Surface Animation"

    directory: StringProperty(name="Frames Directory", subtype="DIR_PATH")  # type: ignore[assignment]

    def execute(self, context):
        frames_dir = Path(self.directory)
        if not frames_dir.exists():
            self.report({"ERROR"}, f"Directory not found: {frames_dir}")
            return {"CANCELLED"}
        
        try:
            object_names, mesh_sequences = make_frame(frames_dir)
        except Exception as exc:  
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        
        objects: Dict[str, bpy.types.Object] = {}
        for obj_name in object_names:
            if mesh_sequences[obj_name]:
                obj = ensure_object(obj_name, mesh_sequences[obj_name][0])
                objects[obj_name] = obj
        
        register_frame_handler(object_names, mesh_sequences, objects)
        
        frame_count = len(next(iter(mesh_sequences.values())))
        self.report(
            {"INFO"}, 
            f"Loaded {frame_count} frames with {len(object_names)} objects: {', '.join(object_names)}"
        )
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


classes = (SPH_OT_load_surface_animation,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
    bpy.ops.sph.load_surface_animation('INVOKE_DEFAULT')
 
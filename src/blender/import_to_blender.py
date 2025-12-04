"""
Import mesh sequence (OBJ/PLY) into Blender and save as .blend file.

This script imports reconstructed surface meshes as an animation.
Since surface reconstruction produces meshes with varying vertex counts,
this uses per-frame object visibility instead of shape keys.

Usage (command line):
    blender --background --python import_to_blender.py

Usage (in Blender):
    Open in Scripting workspace and run.
"""
from __future__ import annotations

import bpy
import math
from pathlib import Path
from typing import List

# --- Configuration ---------------------------------------------------------
MESH_DIRECTORY = Path("/home/cxy/Desktop/ACG/ACG-simulation/src/materials/fluid/mesh_output")
FILE_PATTERN = "*.obj"                        # Use "*.ply" for PLY files
OBJECT_NAME = "FluidSurface"                  # Base name for animated meshes
FRAME_START = 1                               # Timeline frame to start playback
FPS = 60                                      # Frames per second

# Axis settings
AXIS_FORWARD = "NEGATIVE_Z"
AXIS_UP = "Y"

# Output
OUTPUT_BLEND_FILE = Path("output/fluid/fluid_animation.blend")
OUTPUT_RENDER_DIR = Path("output/fluid/render")

# Render settings
RENDER_RESOLUTION_X = 1920
RENDER_RESOLUTION_Y = 1080
# ---------------------------------------------------------------------------


def resolve_frame_paths(directory: Path, pattern: str) -> List[Path]:
    """Resolve and return sorted list of mesh file paths."""
    resolved_dir = directory.resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Directory not found: {resolved_dir}")
    frame_paths = sorted(resolved_dir.glob(pattern))
    if not frame_paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {resolved_dir}")
    return frame_paths


def deselect_all() -> None:
    """Deselect all objects."""
    bpy.ops.object.select_all(action="DESELECT")


def apply_smooth_shading(obj: bpy.types.Object) -> None:
    """Apply smooth shading to mesh object."""
    if obj.type != 'MESH':
        return
    
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Apply smooth shading
    bpy.ops.object.shade_smooth()
    
    # Try to enable auto smooth (API changed in different Blender versions)
    mesh = obj.data
    try:
        # Blender 4.1+
        if hasattr(mesh, 'use_auto_smooth'):
            mesh.use_auto_smooth = True
            mesh.auto_smooth_angle = math.radians(60)
    except (AttributeError, RuntimeError):
        pass
    
    # Alternative: add smooth modifier for Blender 5.0
    try:
        # Check if smooth by angle modifier exists
        if 'Smooth by Angle' not in [m.name for m in obj.modifiers]:
            pass  # Smooth shading should be enough
    except Exception:
        pass
    
    obj.select_set(False)


def import_mesh(filepath: Path) -> bpy.types.Object:
    """Import a single mesh file (OBJ or PLY)."""
    deselect_all()
    
    ext = filepath.suffix.lower()
    if ext == '.ply':
        bpy.ops.wm.ply_import(
            filepath=str(filepath),
            forward_axis=AXIS_FORWARD,
            up_axis=AXIS_UP,
        )
    elif ext == '.obj':
        bpy.ops.wm.obj_import(
            filepath=str(filepath),
            forward_axis=AXIS_FORWARD,
            up_axis=AXIS_UP,
        )
    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    imported = list(bpy.context.selected_objects)
    if not imported:
        raise RuntimeError(f"Nothing imported from {filepath}")
    
    obj = imported[0]
    
    # Apply smooth shading
    apply_smooth_shading(obj)
    
    return obj


def clear_scene() -> None:
    """Remove all mesh objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_mesh_sequence(frame_paths: List[Path]) -> List[bpy.types.Object]:
    """Import all meshes and set up frame-based visibility animation."""
    objects = []
    total = len(frame_paths)
    
    # Create a collection for fluid meshes
    fluid_collection = bpy.data.collections.new("FluidMeshes")
    bpy.context.scene.collection.children.link(fluid_collection)
    
    for i, path in enumerate(frame_paths):
        frame_num = FRAME_START + i
        print(f"Importing frame {i+1}/{total}: {path.name}")
        
        obj = import_mesh(path)
        obj.name = f"{OBJECT_NAME}_{i:04d}"
        
        # Move to fluid collection
        for col in obj.users_collection:
            col.objects.unlink(obj)
        fluid_collection.objects.link(obj)
        
        # Set up visibility keyframes
        # Hide on all frames except its own frame
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=FRAME_START)
        obj.keyframe_insert(data_path="hide_render", frame=FRAME_START)
        
        # Show on its frame
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
        obj.keyframe_insert(data_path="hide_render", frame=frame_num)
        
        # Hide on next frame
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)
        obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
        
        objects.append(obj)
    
    return objects


def setup_water_material() -> bpy.types.Material:
    """Create a water material with proper settings for smooth rendering."""
    mat = bpy.data.materials.new(name="WaterMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    # Use Principled BSDF for realistic water with transparency
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    bsdf.inputs['Base Color'].default_value = (0.7, 0.85, 0.95, 1.0)  # Light blue
    bsdf.inputs['Roughness'].default_value = 0.0
    bsdf.inputs['IOR'].default_value = 1.33
    bsdf.inputs['Alpha'].default_value = 0.5  # Semi-transparent
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Render settings for transparency
    mat.blend_method = 'BLEND'
    
    return mat


def apply_material_to_all(objects: List[bpy.types.Object], material: bpy.types.Material) -> None:
    """Apply material to all objects."""
    for obj in objects:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def setup_lighting() -> None:
    """Add lighting to the scene."""
    # Sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Area light for fill
    bpy.ops.object.light_add(type='AREA', location=(0, -5, 5))
    area = bpy.context.active_object
    area.name = "AreaLight"
    area.data.energy = 500
    area.data.size = 5
    
    # Point light for additional illumination
    bpy.ops.object.light_add(type='POINT', location=(0, 5, -6))
    point = bpy.context.active_object
    point.name = "PointLight"
    point.data.energy = 2000


def setup_camera() -> None:
    """Setup camera with a wider view of the fluid."""
    bpy.ops.object.camera_add(location=(11.617, -13.144, 9.127))
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.rotation_euler = (1.05, 0, 0.64)
    camera.data.lens = 40
    camera.data.clip_end = 200
    bpy.context.scene.camera = camera


def setup_world_background() -> None:
    """Setup a simple gradient background."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    nodes.clear()
    
    # Background node
    background = nodes.new('ShaderNodeBackground')
    background.location = (200, 0)
    background.inputs['Color'].default_value = (0.05, 0.05, 0.1, 1.0)  # Dark blue
    background.inputs['Strength'].default_value = 1.0
    
    # Optional: Add Environment Texture if HDR available
    try:
        env_texture = nodes.new('ShaderNodeTexEnvironment')
        env_texture.location = (0, 0)
        # Uncomment and set path if you have an HDR file
        # env_texture.image = bpy.data.images.load('path/to/background.hdr')
        # links.new(env_texture.outputs['Color'], background.inputs['Color'])
    except Exception:
        pass
    
    # Output
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (400, 0)
    
    links.new(background.outputs['Background'], output.inputs['Surface'])


def setup_scene_settings(num_frames: int) -> None:
    """Configure scene and render settings."""
    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_START + num_frames - 1
    scene.render.fps = FPS
    
    # Render settings
    scene.render.resolution_x = RENDER_RESOLUTION_X
    scene.render.resolution_y = RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    
    # Output path
    OUTPUT_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(OUTPUT_RENDER_DIR / "frame_")
    
    # Render engine - use Cycles for better glass/water rendering
    scene.render.engine = 'BLENDER_EEVEE'
    
    # EEVEE specific settings for better transparency
    try:
        scene.eevee.use_ssr = True
        scene.eevee.use_ssr_refraction = True
        scene.eevee.ssr_quality = 1.0
        scene.eevee.ssr_thickness = 0.1
    except AttributeError:
        pass


def save_blend_file() -> None:
    """Save the .blend file."""
    OUTPUT_BLEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND_FILE))
    print(f"Saved: {OUTPUT_BLEND_FILE}")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Import Mesh Sequence to Blender")
    print("=" * 60)
    
    # Clear existing scene
    clear_scene()
    
    # Find mesh files
    frame_paths = resolve_frame_paths(MESH_DIRECTORY, FILE_PATTERN)
    print(f"Found {len(frame_paths)} mesh files in {MESH_DIRECTORY}")
    
    # Import meshes with visibility animation
    objects = import_mesh_sequence(frame_paths)
    
    # Apply material
    material = setup_water_material()
    apply_material_to_all(objects, material)
    
    # Setup scene
    setup_lighting()
    setup_camera()
    setup_world_background()
    setup_scene_settings(len(frame_paths))
    
    # Save
    save_blend_file()
    
    print("=" * 60)
    print(f"Successfully imported {len(frame_paths)} frames")
    print(f"Animation range: {FRAME_START} - {bpy.context.scene.frame_end}")
    print(f"FPS: {FPS}")
    print(f"Blend file: {OUTPUT_BLEND_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
import bpy
import sys
import argparse
from pathlib import Path

# --- Configuration ---------------------------------------------------------
MESH_DIRECTORY = Path("output/fluid/mesh_output")
FILE_PATTERN = "*.obj"
OUTPUT_DIR = Path("output/fluid/render")

# Render settings
RESOLUTION_X = 1920
RESOLUTION_Y = 1080
RENDER_ENGINE = 'CYCLES'
FPS = 60

# Background
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_STRENGTH = 2.0

# Camera
CAMERA_LOCATION = (11.617, -13.144, 9.127)
CAMERA_ROTATION = (1.05, 0, 0.64)
CAMERA_LENS = 47

# Water Material
WATER_MATERIAL_NAME = "Sea Water.001"
WATER_MATERIAL_FILE = "assets/materials/water.blend"
# ---------------------------------------------------------------------------


def clear_scene() -> None:
    """Clear the scene completely."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def get_material(material_name, file_name):
    """Load material from .blend file."""
    # Load the .blend file
    blend_file_path = file_name
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        if material_name in data_from.materials:
            data_to.materials = [material_name]
        else:
            raise ValueError(f"Material {material_name} not found in {blend_file_path}")

    # Return the material
    return data_to.materials[0]


def setup_water_material() -> bpy.types.Material:
    """Load water material from blend file."""
    return get_material(WATER_MATERIAL_NAME, WATER_MATERIAL_FILE)


def setup_scene() -> None:
    """Setup lighting, camera, and world."""
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))
    point = bpy.context.active_object
    point.name = "PointLight"
    point.data.energy = 10000

    # Camera
    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.active_object
    camera.rotation_euler = CAMERA_ROTATION
    camera.data.lens = CAMERA_LENS
    bpy.context.scene.camera = camera

    # World background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    nodes = world.node_tree.nodes
    nodes.clear()

    background = nodes.new('ShaderNodeBackground')
    background.location = (0, 0)
    background.inputs['Color'].default_value = BACKGROUND_COLOR
    background.inputs['Strength'].default_value = BACKGROUND_STRENGTH

    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (200, 0)
    world.node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

    # Render settings
    scene = bpy.context.scene
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = RENDER_ENGINE

    # Cycles advanced settings
    if RENDER_ENGINE == 'CYCLES':
        scene.cycles.samples = 128
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.use_denoising = True
        scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True

    if RENDER_ENGINE == 'BLENDER_EEVEE':
        try:
            scene.eevee.use_ssr = True
            scene.eevee.use_ssr_refraction = True
        except AttributeError:
            pass


def import_and_render_obj(obj_path: Path, frame_num: int) -> None:
    """Import OBJ, apply material, and render."""
    # Clear existing meshes before importing new one
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

    # Import OBJ
    bpy.ops.wm.obj_import(filepath=str(obj_path))

    # Get imported object
    imported = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported:
        print(f"Warning: No mesh imported from {obj_path}")
        return

    obj = imported[0]

    # Apply smooth shading
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    # Apply material
    mat = setup_water_material()
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Adjust material properties
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Alpha'].default_value = 0.3
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'HASHED'

    # Render
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(OUTPUT_DIR / f"frame_{frame_num:04d}")
    bpy.ops.render.render(write_still=True)

    print(f"Rendered frame {frame_num}: {obj_path.name}")


def main() -> None:
    """Main entry point."""
    # Extract script arguments from sys.argv
    script_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description='Render OBJ sequence to images')
    parser.add_argument('--start-obj', type=int, default=0, help='Start rendering from this OBJ index (0-based)')
    args = parser.parse_args(script_args)

    print("=" * 60)
    print("Render OBJ Sequence to Images")
    print(f"Starting from OBJ index: {args.start_obj}")
    print("=" * 60)

    # Find OBJ files
    obj_files = sorted(MESH_DIRECTORY.glob(FILE_PATTERN))
    if not obj_files:
        print(f"No OBJ files found in {MESH_DIRECTORY}")
        sys.exit(1)

    print(f"Found {len(obj_files)} OBJ files")

    # Setup scene once (lighting, camera, etc.)
    clear_scene()
    setup_scene()

    # Process each OBJ starting from start_obj
    for i, obj_path in enumerate(obj_files[args.start_obj:], args.start_obj + 1):
        print(f"Processing {i}/{len(obj_files)}: {obj_path.name}")
        import_and_render_obj(obj_path, i)

    print("=" * 60)
    print(f"Images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
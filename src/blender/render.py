import bpy
import sys
import argparse
from pathlib import Path

# --- Configuration ---------------------------------------------------------
# These will be set from command line arguments
MESH_DIRECTORY = None
OUTPUT_DIR = None

# Render settings
RESOLUTION_X = 1920
RESOLUTION_Y = 1080
RENDER_ENGINE = 'CYCLES'
FPS = 60

# Background
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND_STRENGTH = 2.0

# Camera
CAMERA_LOCATION = (11.617, -13.144, 7.2)
CAMERA_ROTATION = (1.2, 0, 0.64)
CAMERA_LENS = 60

# Materials
WATER_MATERIAL_NAME = "Sea Water.001"
WATER_MATERIAL_FILE = "assets/materials/water2.blend"
METAL_MATERIAL_NAME = "Metal"
METAL_MATERIAL_FILE = "assets/materials/metal.blend"
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


def setup_metal_material() -> bpy.types.Material:
    """Load metal material from blend file."""
    return get_material(METAL_MATERIAL_NAME, METAL_MATERIAL_FILE)


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


def import_and_render_objs(fluid_obj_path: Path, rigid_obj_path: Path, frame_num: int) -> None:
    """Import both fluid and rigid body OBJs, apply materials, and render."""
    # Clear existing meshes before importing new ones
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

    imported_objects = []

    # Import fluid OBJ
    if fluid_obj_path.exists():
        bpy.ops.wm.obj_import(filepath=str(fluid_obj_path))
        fluid_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if fluid_objs:
            fluid_obj = fluid_objs[0]
            fluid_obj.name = "Fluid"
            imported_objects.append(("fluid", fluid_obj))

    # Import rigid body OBJ
    if rigid_obj_path.exists():
        bpy.ops.wm.obj_import(filepath=str(rigid_obj_path))
        rigid_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if rigid_objs:
            rigid_obj = rigid_objs[0]
            rigid_obj.name = "RigidBody"
            imported_objects.append(("rigid", rigid_obj))

    if not imported_objects:
        print(f"Warning: No meshes imported from {fluid_obj_path} or {rigid_obj_path}")
        return

    # Apply materials to each object
    for obj_type, obj in imported_objects:
        # Apply smooth shading
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

        # Apply appropriate material
        if obj_type == "fluid":
            mat = setup_water_material()
            # Adjust material properties for water
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Alpha'].default_value = 0.3
                mat.blend_method = 'BLEND'
                mat.shadow_method = 'HASHED'
        else:  # rigid
            mat = setup_metal_material()
            # Metal material should be opaque
            mat.blend_method = 'OPAQUE'

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    # Render
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(OUTPUT_DIR / f"frame_{frame_num:04d}")
    bpy.ops.render.render(write_still=True)

    print(f"Rendered frame {frame_num}: fluid={fluid_obj_path.name}, rigid={rigid_obj_path.name}")


def main() -> None:
    """Main entry point."""
    # Extract script arguments from sys.argv
    script_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description='Render OBJ sequence to images')
    parser.add_argument('--scene', type=str, required=True, help='Scene name (e.g., dragon_bath)')
    parser.add_argument('--start-obj', type=int, default=0, help='Start rendering from this OBJ index (0-based)')
    args = parser.parse_args(script_args)

    # Set global paths based on scene
    global MESH_DIRECTORY, OUTPUT_DIR
    MESH_DIRECTORY = Path(f"output/fluid/{args.scene}/mesh_output")
    OUTPUT_DIR = Path(f"output/fluid/{args.scene}/render")

    print("=" * 60)
    print("Render OBJ Sequence to Images")
    print(f"Scene: {args.scene}")
    print(f"Mesh directory: {MESH_DIRECTORY}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Starting from OBJ index: {args.start_obj}")
    print("=" * 60)

    # Find fluid OBJ files
    fluid_pattern = "fluid_*.obj"
    fluid_files = sorted(MESH_DIRECTORY.glob(fluid_pattern))
    if not fluid_files:
        print(f"No fluid OBJ files found in {MESH_DIRECTORY} with pattern {fluid_pattern}")
        sys.exit(1)

    # Find rigid body OBJ files (in obj_1 subdirectory)
    rigid_dir = MESH_DIRECTORY / "obj_1"
    rigid_pattern = "obj_1_*.obj"
    rigid_files = sorted(rigid_dir.glob(rigid_pattern)) if rigid_dir.exists() else []

    print(f"Found {len(fluid_files)} fluid OBJ files")
    print(f"Found {len(rigid_files)} rigid body OBJ files")

    # Determine the number of frames to process
    max_frames = max(len(fluid_files), len(rigid_files))
    if max_frames == 0:
        print("No OBJ files found")
        sys.exit(1)

    # Setup scene once (lighting, camera, etc.)
    clear_scene()
    setup_scene()

    # Process each frame
    for i in range(args.start_obj, max_frames):
        fluid_path = fluid_files[i] if i < len(fluid_files) else None
        rigid_path = rigid_files[i] if i < len(rigid_files) else None

        print(f"Processing frame {i+1}/{max_frames}: fluid={fluid_path.name if fluid_path else 'None'}, rigid={rigid_path.name if rigid_path else 'None'}")
        import_and_render_objs(fluid_path, rigid_path, i+1)

    print("=" * 60)
    print(f"Images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
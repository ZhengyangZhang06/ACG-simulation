import bpy
import sys
import argparse
import json
from pathlib import Path

# --- Configuration ---------------------------------------------------------
# These will be set from command line arguments or config
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

# Camera defaults
CAMERA_LOCATION = (10, -13.144, 9.127)
CAMERA_ROTATION = (1.05, 0, 0.64)
CAMERA_LENS = 80

# Materials defaults
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


def setup_water_material(material_name=None, material_file=None) -> bpy.types.Material:
    """Load water material from blend file."""
    name = material_name or WATER_MATERIAL_NAME
    file = material_file or WATER_MATERIAL_FILE
    return get_material(name, file)


def setup_metal_material(material_name=None, material_file=None) -> bpy.types.Material:
    """Load metal material from blend file."""
    name = material_name or METAL_MATERIAL_NAME
    file = material_file or METAL_MATERIAL_FILE
    return get_material(name, file)


def setup_scene(camera_location=None, camera_rotation=None, camera_lens=None, fps=None) -> None:
    """Setup lighting, camera, and world."""
    global FPS
    if fps is not None:
        FPS = fps
    
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))
    point = bpy.context.active_object
    point.name = "PointLight"
    point.data.energy = 10000

    # Camera
    loc = camera_location or CAMERA_LOCATION
    rot = camera_rotation or CAMERA_ROTATION
    lens = camera_lens or CAMERA_LENS
    bpy.ops.object.camera_add(location=loc)
    camera = bpy.context.active_object
    camera.rotation_euler = rot
    camera.data.lens = lens
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


def import_and_render_objs(fluid_obj_path: Path, rigid_obj_paths: list, frame_num: int, fluid_material=None, rigid_materials=None) -> None:
    """Import fluid and multiple rigid body OBJs, apply materials, and render."""
    # Clear existing meshes before importing new ones
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

    imported_objects = []

    # Import fluid OBJ
    if fluid_obj_path and fluid_obj_path.exists():
        bpy.ops.wm.obj_import(filepath=str(fluid_obj_path))
        fluid_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if fluid_objs:
            fluid_obj = fluid_objs[0]
            fluid_obj.name = "Fluid"
            imported_objects.append(("fluid", fluid_obj))

    # Import rigid body OBJs
    for i, rigid_obj_path in enumerate(rigid_obj_paths):
        if rigid_obj_path and rigid_obj_path.exists():
            bpy.ops.wm.obj_import(filepath=str(rigid_obj_path))
            rigid_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if rigid_objs:
                rigid_obj = rigid_objs[0]
                rigid_obj.name = f"RigidBody_{i+1}"
                imported_objects.append(("rigid", rigid_obj, i))

    if not imported_objects:
        print(f"Warning: No meshes imported")
        return

    # Apply materials to each object
    for item in imported_objects:
        if len(item) == 2:  # fluid
            obj_type, obj = item
        else:  # rigid
            obj_type, obj, idx = item
        
        # Apply smooth shading
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

        # Apply appropriate material
        if obj_type == "fluid":
            mat_name = fluid_material.get("materialName") if fluid_material else None
            mat_file = fluid_material.get("materialFile") if fluid_material else None
            mat = setup_water_material(mat_name, mat_file)
            # Adjust material properties for water
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Alpha'].default_value = 0.3
                mat.blend_method = 'BLEND'
                mat.shadow_method = 'HASHED'
        else:  # rigid
            mat_name = rigid_materials[idx].get("materialName") if rigid_materials and idx < len(rigid_materials) else None
            mat_file = rigid_materials[idx].get("materialFile") if rigid_materials and idx < len(rigid_materials) else None
            mat = setup_metal_material(mat_name, mat_file)
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

    print(f"Rendered frame {frame_num}")


def main() -> None:
    """Main entry point."""
    # Extract script arguments from sys.argv
    script_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description='Render OBJ sequence to images')
    parser.add_argument('--scene', type=str, required=True, help='Path to config JSON file (e.g., src/configs/basic_fluid.json)')
    parser.add_argument('--start-obj', type=int, default=0, help='Start rendering from this OBJ index (0-based)')
    args = parser.parse_args(script_args)

    # Load config from JSON
    config_path = Path(args.scene)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract render settings
    render_config = config.get("Render", {})
    camera_location = render_config.get("cameraLocation", CAMERA_LOCATION)
    camera_rotation = render_config.get("cameraRotation", CAMERA_ROTATION)
    camera_lens = render_config.get("cameraLens", CAMERA_LENS)
    fps = render_config.get("fps", FPS)

    # Extract materials
    fluid_blocks = config.get("FluidBlocks", [])
    rigid_bodies = config.get("RigidBodies", [])
    
    fluid_material = fluid_blocks[0].get("material", {}) if fluid_blocks else {}
    rigid_materials = [rb.get("material", {}) for rb in rigid_bodies]

    # Set global paths based on config
    global MESH_DIRECTORY, OUTPUT_DIR
    scene_name = config_path.stem  # Use config filename as scene name
    
    MESH_DIRECTORY = Path(f"output/{scene_name}/mesh_output")
    OUTPUT_DIR = Path(f"output/{scene_name}/render")

    print("=" * 60)
    print("Render OBJ Sequence to Images")
    print(f"Config: {config_path}")
    print(f"Scene: {scene_name}")
    print(f"Mesh directory: {MESH_DIRECTORY}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Starting from OBJ index: {args.start_obj}")
    print(f"Camera: location={camera_location}, rotation={camera_rotation}, lens={camera_lens}")
    print(f"FPS: {fps}")
    print("=" * 60)

    # Find fluid OBJ files (if exists)
    fluid_pattern = "fluid_*.obj"
    fluid_files = sorted(MESH_DIRECTORY.glob(fluid_pattern))
    if fluid_files:
        print(f"Found {len(fluid_files)} fluid OBJ files")
    else:
        print("No fluid OBJ files found")

    # Find rigid body OBJ files (in obj_1, obj_2, etc. subdirectories)
    rigid_files_list = []
    for i, rb in enumerate(rigid_bodies):
        object_id = rb.get("objectId", i+1)
        rigid_dir = MESH_DIRECTORY / f"obj_{object_id}"
        rigid_pattern = f"obj_{object_id}_*.obj"
        rigid_files = sorted(rigid_dir.glob(rigid_pattern)) if rigid_dir.exists() else []
        rigid_files_list.append(rigid_files)
        if rigid_files:
            print(f"Found {len(rigid_files)} rigid body OBJ files for obj_{object_id}")

    # Determine the number of frames to process
    all_files = ([len(fluid_files)] if fluid_files else []) + [len(rf) for rf in rigid_files_list]
    max_frames = max(all_files) if all_files else 0
    if max_frames == 0:
        print("No OBJ files found")
        sys.exit(1)

    # Setup scene once (lighting, camera, etc.)
    clear_scene()
    setup_scene(camera_location, camera_rotation, camera_lens, fps)

    # Process each frame
    for i in range(args.start_obj, max_frames):
        fluid_path = fluid_files[i] if fluid_files and i < len(fluid_files) else None
        rigid_paths = [rf[i] if i < len(rf) else None for rf in rigid_files_list]

        print(f"Processing frame {i}/{max_frames}: fluid={fluid_path.name if fluid_path else 'None'}, rigids={[rp.name if rp else 'None' for rp in rigid_paths]}")
        import_and_render_objs(fluid_path, rigid_paths, i, fluid_material, rigid_materials)

    print("=" * 60)
    print(f"Images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
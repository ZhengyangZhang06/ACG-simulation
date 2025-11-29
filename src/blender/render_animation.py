"""
Render animation from Blender file to PNG sequence.

This script allows adjusting water color, background, camera position
without regenerating the .blend file.

Usage (command line):
    blender -b /path/to/file.blend --python render_animation.py

Or render directly without this script:
    blender -b /path/to/file.blend -a
"""
from __future__ import annotations

import bpy
import math
from pathlib import Path

# --- Configuration ---------------------------------------------------------
INPUT_BLEND_FILE = Path("/home/cxy/Desktop/ACG/ACG-simulation/output/fluid_animation.blend")
OUTPUT_DIR = Path("/home/cxy/Desktop/ACG/ACG-simulation/output/render")

# Render settings
RESOLUTION_X = 1920
RESOLUTION_Y = 1080
RESOLUTION_PERCENTAGE = 100

# Frame range (None = use scene settings)
FRAME_START = None
FRAME_END = None

# Render engine: 'BLENDER_EEVEE', 'CYCLES', 'BLENDER_WORKBENCH'
RENDER_ENGINE = 'BLENDER_EEVEE'

# Cycles-specific settings
CYCLES_SAMPLES = 128
CYCLES_DEVICE = 'GPU'

# EEVEE-specific settings
EEVEE_SAMPLES = 64

# =============================================================================
# ADJUSTABLE PARAMETERS (modify these to change appearance without regenerating)
# =============================================================================

# Water material settings (Principled BSDF with transmission for realistic water)
WATER_COLOR = (0.05, 0.25, 0.70, 1.0)     # RGBA: blue water
WATER_ROUGHNESS = 0.03                    # 0 = smooth glass, 1 = rough
WATER_IOR = 1.333                         # Index of refraction (water = 1.333)
WATER_TRANSMISSION = 1.0                  # 1.0 = fully refractive glass/water

# Background color (RGBA)
# BACKGROUND_COLOR = (0.05, 0.05, 0.1, 1.0)  # Dark blue
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)  # White
BACKGROUND_STRENGTH = 3.0

# Camera settings
CAMERA_LOCATION = (11.617, -13.144, 9.127)      # (X, Y, Z)
CAMERA_ROTATION = (1.05, 0, 0.64)         # (X, Y, Z) in radians
CAMERA_LENS = 47                          # Focal length in mm

# Lighting settings
SUN_ENERGY = 5.0
SUN_ROTATION = (0.785, 0, 0.785)         # 45 degrees
AREA_LIGHT_ENERGY = 1000
# =============================================================================


def update_water_material() -> None:
    """Update or create realistic water material using Principled BSDF with transmission."""
    mat_name = "WaterReal"
    
    # Create new material or get existing
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Get or create Principled BSDF
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        nodes.clear()
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Set water properties
    bsdf.inputs['Base Color'].default_value = WATER_COLOR
    bsdf.inputs['Roughness'].default_value = WATER_ROUGHNESS
    bsdf.inputs['IOR'].default_value = WATER_IOR
    
    # Transmission property name changed in Blender 4.0+
    try:
        bsdf.inputs['Transmission Weight'].default_value = WATER_TRANSMISSION
    except KeyError:
        try:
            bsdf.inputs['Transmission'].default_value = WATER_TRANSMISSION
        except KeyError:
            pass
    
    # Blend method for transparency
    try:
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'HASHED'
    except AttributeError:
        pass
    
    # Apply to all fluid objects
    count = 0
    for obj in bpy.data.objects:
        if "FluidSurface" in obj.name and obj.type == 'MESH':
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
            count += 1
    
    print(f"Updated water material: color={WATER_COLOR[:3]}, roughness={WATER_ROUGHNESS}, "
          f"IOR={WATER_IOR}, transmission={WATER_TRANSMISSION}")
    print(f"Applied to {count} fluid objects")


def update_background() -> None:
    """Update world background color."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    
    # Find or create background node
    background = None
    for node in nodes:
        if node.type == 'BACKGROUND':
            background = node
            break
    
    if background is None:
        nodes.clear()
        background = nodes.new('ShaderNodeBackground')
        background.location = (0, 0)
        output = nodes.new('ShaderNodeOutputWorld')
        output.location = (200, 0)
        world.node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])
    
    background.inputs['Color'].default_value = BACKGROUND_COLOR
    background.inputs['Strength'].default_value = BACKGROUND_STRENGTH
    print(f"Updated background: color={BACKGROUND_COLOR[:3]}, strength={BACKGROUND_STRENGTH}")


def update_camera() -> None:
    """Update camera position and settings."""
    camera = bpy.context.scene.camera
    if camera is None:
        print("Warning: No camera found")
        return
    
    camera.location = CAMERA_LOCATION
    camera.rotation_euler = CAMERA_ROTATION
    camera.data.lens = CAMERA_LENS
    print(f"Updated camera: location={CAMERA_LOCATION}, rotation={CAMERA_ROTATION}, lens={CAMERA_LENS}mm")


def update_lighting() -> None:
    """Update lighting settings."""
    # Update sun
    sun = bpy.data.objects.get("Sun")
    if sun and sun.type == 'LIGHT':
        sun.data.energy = SUN_ENERGY
        sun.rotation_euler = SUN_ROTATION
        print(f"Updated sun: energy={SUN_ENERGY}")
    
    # Update area light
    area = bpy.data.objects.get("AreaLight")
    if area and area.type == 'LIGHT':
        area.data.energy = AREA_LIGHT_ENERGY
        print(f"Updated area light: energy={AREA_LIGHT_ENERGY}")


def setup_render_settings() -> None:
    """Configure render settings."""
    scene = bpy.context.scene
    
    # Resolution
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = RESOLUTION_PERCENTAGE
    
    # Frame range
    if FRAME_START is not None:
        scene.frame_start = FRAME_START
    if FRAME_END is not None:
        scene.frame_end = FRAME_END
    
    # Output format
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.compression = 15
    
    # Output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(OUTPUT_DIR / "frame_")
    
    # Render engine
    scene.render.engine = RENDER_ENGINE
    
    if RENDER_ENGINE == 'CYCLES':
        scene.cycles.samples = CYCLES_SAMPLES
        scene.cycles.device = CYCLES_DEVICE
        if CYCLES_DEVICE == 'GPU':
            try:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = 'CUDA'
                for device in prefs.devices:
                    device.use = True
            except Exception:
                pass
    
    elif RENDER_ENGINE == 'BLENDER_EEVEE':
        try:
            scene.eevee.taa_render_samples = EEVEE_SAMPLES
            # Enable screen space reflections and refractions for water
            scene.eevee.use_ssr = True
            scene.eevee.use_ssr_refraction = True
            scene.eevee.ssr_quality = 1.0
            scene.eevee.ssr_thickness = 0.1
        except AttributeError:
            pass


def render_animation() -> None:
    """Render the animation."""
    print("=" * 60)
    print("Rendering Animation")
    print("=" * 60)
    
    scene = bpy.context.scene
    print(f"Frame range: {scene.frame_start} - {scene.frame_end}")
    print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"Engine: {scene.render.engine}")
    print(f"Output: {scene.render.filepath}")
    print("-" * 60)
    
    bpy.ops.render.render(animation=True)
    
    print("-" * 60)
    print(f"Render complete!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Updating scene parameters...")
    print("=" * 60)
    
    # Update all adjustable parameters
    update_water_material()
    update_background()
    update_camera()
    update_lighting()
    
    # Setup render settings
    setup_render_settings()
    
    # Render
    render_animation()


if __name__ == "__main__":
    main()

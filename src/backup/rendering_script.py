import bpy
import sys
import os

shading_denoiser = 'OPTIX'

compute_engine = sys.argv[-4]
target_gpu_index = int(sys.argv[-3])
current_device_ptr = 0

for hardware in bpy.context.preferences.addons['cycles'].preferences.devices:
    if hardware.type == compute_engine:
        if current_device_ptr == target_gpu_index:
            hardware.use = True
            print(f"Assigning hardware: {hardware.name} ID: {target_gpu_index} Type: {hardware.type}")
        else:
            hardware.use = False
        current_device_ptr += 1
    else:
        hardware.use = False

source_folder = sys.argv[-2]
final_rendering_path = sys.argv[-1]

active_frame_number = int(os.path.basename(source_folder))
bpy.context.scene.frame_set(active_frame_number)

print("\n[Initial Camera State]")
viewpoint_camera = bpy.data.objects.get('Camera')
if viewpoint_camera:
    eval_graph = bpy.context.evaluated_depsgraph_get()
    eval_viewpoint = viewpoint_camera.evaluated_get(eval_graph)
    print(f"  Original Location: {viewpoint_camera.location}")
    print(f"  Original Rotation: {viewpoint_camera.rotation_euler}")
    print(f"  Evaluated Matrix Translation: {eval_viewpoint.matrix_world.translation}")
    print(f"  Evaluated Matrix Rotation: {eval_viewpoint.matrix_world.to_euler()}")
    if viewpoint_camera.constraints:
        for link in viewpoint_camera.constraints:
            print(f"  Constraint '{link.name}': Target = {link.target.name if link.target else 'None'}")
else:
    print("  No camera found!")


for element in os.listdir(source_folder):
    if element.endswith(".obj"):
        geometry_path = os.path.join(source_folder, element)
        base_mesh_identity = element.split(".")[0]
        
        bpy.ops.wm.obj_import(
            filepath=geometry_path,
            forward_axis='Z',
            up_axis='Y'
        )
        
        newly_added_mesh = None
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                newly_added_mesh = obj
                break
        
        if newly_added_mesh is None:
            continue
        
        if base_mesh_identity in bpy.data.objects:
            existing_scene_mesh = bpy.data.objects[base_mesh_identity]

            newly_added_mesh.data.materials.clear()
            for surface_mat in existing_scene_mesh.data.materials:
                newly_added_mesh.data.materials.append(surface_mat)
            

            legacy_mesh_ref = existing_scene_mesh
            
            modified_constraints_tally = 0
            for obj in bpy.data.objects:
                if obj.constraints:
                    for link in obj.constraints:
                        if hasattr(link, 'target') and link.target == legacy_mesh_ref:
                            print(f"[Link Adjustment] '{obj.name}'.'{link.name}' source: {link.target.name} -> destination: {newly_added_mesh.name}")
                            link.target = newly_added_mesh
                            modified_constraints_tally += 1
                            print(f"    Updated '{link.name}' on '{obj.name}' to point to '{newly_added_mesh.name}'")
                        
                        if hasattr(link, 'subtarget') and link.subtarget == legacy_mesh_ref.name:
                            link.subtarget = newly_added_mesh.name
                    
            cleanup_queue = []
            for item in bpy.data.objects:
                if item != newly_added_mesh and (item.name == base_mesh_identity or item.name.startswith(base_mesh_identity + ".")):
                    cleanup_queue.append(item)
            
            for stale_obj in cleanup_queue:
                bpy.data.objects.remove(stale_obj, do_unlink=True)
            
            newly_added_mesh.name = base_mesh_identity
            
            if newly_added_mesh.type == 'MESH' and newly_added_mesh.data.vertices:
                mesh_data = newly_added_mesh.data
                local_centroid = sum((v.co for v in mesh_data.vertices), start=bpy.context.scene.cursor.location * 0) / len(mesh_data.vertices)
                world_centroid = newly_added_mesh.matrix_world @ local_centroid
                
                for v in mesh_data.vertices:
                    v.co -= local_centroid
                
                newly_added_mesh.location = world_centroid
                print(f"  [Origin Sync] Mesh '{base_mesh_identity}' center at: {newly_added_mesh.location}")
            else:
                print(f"  [Origin Missing] Mesh '{base_mesh_identity}' has no data")
        else:
            newly_added_mesh.name = base_mesh_identity

scene_graph = bpy.context.evaluated_depsgraph_get()
scene_graph.update()
bpy.context.view_layer.update()

bpy.context.scene.frame_set(bpy.context.scene.frame_current)
viewpoint_camera = bpy.data.objects.get('Camera')
if viewpoint_camera and viewpoint_camera.constraints:
    for link in viewpoint_camera.constraints:
        print(f"  Link '{link.name}': Target = {link.target.name if link.target else 'None'}")
    eval_graph = bpy.context.evaluated_depsgraph_get()
    eval_viewpoint = viewpoint_camera.evaluated_get(eval_graph)
    print(f"  Location: {viewpoint_camera.location}")
    print(f"  Rotation: {viewpoint_camera.rotation_euler}")
    print(f"  Eval Translation: {eval_viewpoint.matrix_world.translation}")
    print(f"  Eval Rotation: {eval_viewpoint.matrix_world.to_euler()}")    
    for link in viewpoint_camera.constraints:
        if hasattr(link, 'target') and link.target:
            ptr = link.target
            print(f"  Point of Interest '{ptr.name}' at: {ptr.location}")
bpy.context.scene.render.filepath = final_rendering_path
bpy.ops.render.render(write_still=True)

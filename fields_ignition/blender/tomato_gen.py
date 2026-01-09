import sys
import bpy
import numpy as np
import re
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--model_dir', dest='model_dir',
                    default='generated/test_tomato')
parser.add_argument('-s', '--seed', dest='seed', type=int, default=np.random.randint(10000))
parser.add_argument('-f', '--format', dest='format', choices=['dae', 'fbx'], default='dae',
                    help='Export format: dae (COLLADA) or fbx')
parser.add_argument('--with-textures', dest='with_textures', action='store_true',
                    help='Keep materials and embed textures in export')
parser.add_argument('--alpha-clip', dest='alpha_clip', type=float, default=None,
                    help='Remove faces where texture alpha is below this threshold (0.0-1.0)')
parser.add_argument('--alpha-clip-mode', dest='alpha_clip_mode', choices=['any', 'all', 'majority'],
                    default='majority',
                    help='How to decide face removal: any=if any vertex transparent, all=if all vertices transparent, majority=if most vertices transparent')
parser.add_argument('--export-format', dest='export_format_ext', choices=['fbx', 'glb', 'dae', 'usd'],
                    default=None, help='Alias for --format, also supports glb (glTF binary) and usd')
parser.add_argument('--remesh-alpha', dest='remesh_alpha', action='store_true',
                    help='Remesh objects so mesh boundaries follow texture alpha contours')


if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
else:
    argv = []
args = parser.parse_known_args(argv)[0]

np.random.seed(args.seed)

bpy.ops.object.select_all(action='DESELECT')

model_dir = Path(args.model_dir)
model_dir.mkdir(parents=True, exist_ok=True)
(model_dir / 'meshes').mkdir(parents=True, exist_ok=True)

# Fix texture paths in the blend file
script_dir = Path(__file__).parent
textures_dir = script_dir / 'textures'

# Mapping from old texture names to new ones
TEXTURE_MAPPING = {
    'AG15blo1': 'blo1.png',
    'AG15blo2': 'blo2.png',
    'AG15blo3': 'blo3.png',
    'AG15brn1': 'brn1.png',
    'AG15frt1': 'frt1.png',
    'AG15frt2': 'frt2.png',
    'AG15frt3': 'frt3.png',
    'AG15frt4': 'frt4.png',
    'AG15lef1': 'lef1.png',
    'AG15lef2': 'lef2.png',
}

def fix_texture_paths():
    """Update texture paths in all images to point to the textures directory."""
    for image in bpy.data.images:
        if image.filepath:
            old_name = Path(image.filepath).stem
            if old_name in TEXTURE_MAPPING:
                new_path = str(textures_dir / TEXTURE_MAPPING[old_name])
                image.filepath = new_path
                image.reload()

if args.with_textures or args.alpha_clip is not None:
    fix_texture_paths()


def get_image_for_material(mat):
    """Get the image texture from a material's node tree.

    Prioritizes images that have data loaded (valid textures).
    """
    if mat is None or mat.node_tree is None:
        return None

    # First pass: find images with data
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image is not None:
            if node.image.has_data and node.image.channels >= 4:
                return node.image

    # Second pass: return any image (will likely fail to sample)
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return node.image
    return None


def sample_texture_alpha(image, uv):
    """Sample alpha value from image at given UV coordinate."""
    if image is None or image.pixels is None or len(image.pixels) == 0:
        return 1.0

    width = image.size[0]
    height = image.size[1]
    channels = image.channels

    if channels < 4:
        return 1.0

    u = uv[0] % 1.0
    v = uv[1] % 1.0

    x = int(u * (width - 1))
    y = int(v * (height - 1))

    pixel_index = (y * width + x) * channels
    if pixel_index + 3 < len(image.pixels):
        return image.pixels[pixel_index + 3]
    return 1.0


def remesh_object_by_alpha(obj, threshold=0.5):
    """Remesh an object so mesh boundaries follow texture alpha contours.

    This replaces the mesh with a new one where edges align with
    the alpha boundary of the texture.

    Parameters
    ----------
    obj : bpy.types.Object
        The mesh object to remesh.
    threshold : float
        Alpha threshold for boundary detection.
    """
    import bmesh
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
    from scipy.spatial import Delaunay

    if obj.type != 'MESH':
        return

    mesh = obj.data
    if not mesh.uv_layers:
        return

    mat = None
    if obj.material_slots:
        mat = obj.material_slots[0].material

    image = get_image_for_material(mat)
    if image is None or not image.has_data:
        return

    width = image.size[0]
    height = image.size[1]
    channels = image.channels

    if channels < 4:
        return

    # Extract alpha channel as 2D array
    pixels = np.array(image.pixels[:]).reshape(height, width, channels)
    alpha = pixels[:, :, 3]

    # Create binary mask
    mask = alpha > threshold

    # Fill holes
    mask = binary_fill_holes(mask)

    # Find boundary pixels (edge detection)
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded

    # Get boundary coordinates
    boundary_coords = np.argwhere(boundary)

    if len(boundary_coords) < 10:
        return

    # Subsample boundary for reasonable vertex count
    step = max(1, len(boundary_coords) // 200)
    boundary_coords = boundary_coords[::step]

    # Also add interior points for better triangulation
    interior_mask = binary_erosion(mask, iterations=5)
    interior_coords = np.argwhere(interior_mask)
    if len(interior_coords) > 0:
        interior_step = max(1, len(interior_coords) // 100)
        interior_coords = interior_coords[::interior_step]
        all_coords = np.vstack([boundary_coords, interior_coords])
    else:
        all_coords = boundary_coords

    # Convert to UV coordinates
    uv_points = np.zeros((len(all_coords), 2))
    uv_points[:, 0] = all_coords[:, 1] / width  # u
    uv_points[:, 1] = all_coords[:, 0] / height  # v

    # Triangulate
    try:
        tri = Delaunay(uv_points)
    except Exception:
        return

    # Filter triangles - keep only those with center inside the mask
    valid_faces = []
    for simplex in tri.simplices:
        # Get UV center of triangle
        center_uv = uv_points[simplex].mean(axis=0)

        # Convert to pixel coordinates
        px = int(center_uv[0] * (width - 1))
        py = int(center_uv[1] * (height - 1))

        # Check if inside mask
        if 0 <= px < width and 0 <= py < height and mask[py, px]:
            valid_faces.append(simplex.tolist())

    if not valid_faces:
        return

    # Create new mesh
    bm = bmesh.new()

    # Create vertices (as flat plane in XY, using UV as position)
    verts = []
    for uv in uv_points:
        # Map UV to 3D position (scale to reasonable size)
        x = (uv[0] - 0.5) * 0.2
        y = (uv[1] - 0.5) * 0.2
        z = 0
        v = bm.verts.new((x, y, z))
        verts.append(v)

    bm.verts.ensure_lookup_table()

    # Create faces
    for face_indices in valid_faces:
        try:
            face_verts = [verts[i] for i in face_indices]
            bm.faces.new(face_verts)
        except Exception:
            pass

    # Create UV layer and assign UVs
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for face in bm.faces:
        for loop in face.loops:
            vert_idx = verts.index(loop.vert)
            loop[uv_layer].uv = uv_points[vert_idx]

    # Update the mesh
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def sample_face_alpha_grid(image, uvs, grid_size=4):
    """Sample alpha at multiple points inside a face.

    Parameters
    ----------
    image : bpy.types.Image
        The texture image.
    uvs : list
        List of UV coordinates for face vertices.
    grid_size : int
        Number of sample points along each axis.

    Returns
    -------
    list
        List of alpha values sampled inside the face.
    """
    if len(uvs) < 3:
        return [1.0]

    samples = []

    # Sample at vertices
    for uv in uvs:
        samples.append(sample_texture_alpha(image, uv))

    # Sample inside the face using barycentric interpolation
    # For triangles and quads
    if len(uvs) >= 3:
        # Use first 3 vertices to define barycentric coords
        uv0, uv1, uv2 = uvs[0], uvs[1], uvs[2]

        for i in range(grid_size):
            for j in range(grid_size - i):
                # Barycentric coordinates
                w0 = (i + 0.5) / grid_size
                w1 = (j + 0.5) / grid_size
                w2 = 1.0 - w0 - w1

                if w2 >= 0:
                    u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                    v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                    samples.append(sample_texture_alpha(image, (u, v)))

    return samples


def setup_material_alpha_clip(mat, threshold=0.5):
    """Set up material for alpha clip transparency.

    This configures the material's blend mode and adds a Math node
    to create sharp alpha cutoff (Alpha Clip behavior).
    """
    if mat is None or mat.node_tree is None:
        return

    # Set blend mode to Alpha Clip (for Blender < 4.2)
    # In 4.2+ this may not exist but we try anyway
    try:
        mat.blend_method = 'CLIP'
        mat.alpha_threshold = threshold
    except AttributeError:
        pass

    # For better compatibility, also modify the shader nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Find Principled BSDF and image texture
    principled = None
    img_texture = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled = node
        elif node.type == 'TEX_IMAGE' and node.image and node.image.has_data:
            img_texture = node

    if principled is None or img_texture is None:
        return

    # Check if alpha is already connected
    alpha_input = principled.inputs.get('Alpha')
    if alpha_input is None:
        return

    # Add Math node for alpha clip (Greater Than threshold)
    math_node = nodes.new('ShaderNodeMath')
    math_node.operation = 'GREATER_THAN'
    math_node.inputs[1].default_value = threshold
    math_node.location = (principled.location.x - 200, principled.location.y - 200)

    # Connect: Image Alpha -> Math -> Principled Alpha
    links.new(img_texture.outputs['Alpha'], math_node.inputs[0])
    links.new(math_node.outputs[0], alpha_input)


def remove_transparent_faces(obj, threshold, mode='majority'):
    """Remove faces where texture alpha is below threshold.

    Parameters
    ----------
    obj : bpy.types.Object
        The mesh object to process.
    threshold : float
        Alpha threshold (0.0-1.0).
    mode : str
        'any' - remove if ANY vertex is transparent
        'all' - remove only if ALL vertices are transparent
        'majority' - remove if more than half of vertices are transparent
    """
    import bmesh

    if obj.type != 'MESH':
        return

    mesh = obj.data
    if not mesh.uv_layers:
        return

    mat = None
    if obj.material_slots:
        mat = obj.material_slots[0].material

    image = get_image_for_material(mat)
    if image is None:
        return

    # Ensure image pixels are loaded
    if not image.has_data:
        image.reload()

    bm = bmesh.new()
    bm.from_mesh(mesh)

    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        bm.free()
        return

    faces_to_remove = []
    for face in bm.faces:
        uvs = [loop[uv_layer].uv for loop in face.loops]

        # Count transparent vertices
        transparent_count = 0
        for uv in uvs:
            alpha = sample_texture_alpha(image, (uv[0], uv[1]))
            if alpha < threshold:
                transparent_count += 1

        # Decide based on mode
        should_remove = False
        if mode == 'any':
            should_remove = transparent_count > 0
        elif mode == 'all':
            should_remove = transparent_count == len(uvs)
        else:  # majority
            should_remove = transparent_count > len(uvs) / 2

        if should_remove:
            faces_to_remove.append(face)

    for face in faces_to_remove:
        bm.faces.remove(face)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# Generate the main structure

class Node:
    def __init__(self, x, y, z, yaw, r):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.r = r


def gen_node_heights(height: float, node_count: int):
    sample_count = 8
    samples = np.array([np.concatenate([np.sort(np.random.uniform(
        0.005, height, node_count)), [height]]) for _ in range(sample_count)])
    max_distances = np.max(samples[:, 1:] - samples[:, :-1], axis=1)
    return samples[np.argmin(max_distances)]


def gen_main_stem(collection):
    verts = []
    faces = []

    DIV = 23
    height = 1.25 + np.random.normal(0, 0.1)
    r_start = .009 + np.random.normal(0, 0.0001)
    r_end = r_start * 0.42
    node_count = 19 + np.random.randint(6)

    # create the nodes from bottom-up
    nodes = [Node(0, 0, 0, 0, r_start)]
    node_heights = gen_node_heights(height, node_count)
    for z in node_heights:
        prev = nodes[-1]
        yaw = prev.yaw + (np.pi * (2.0/3.0) + np.random.normal(0, 0.4))
        step_up = z - prev.z
        y = np.sin(yaw) * (step_up/15.0)
        x = np.cos(yaw) * (step_up/15.0)
        r = r_start - (r_start-r_end) * (z / height)
        nodes.append(Node(x, y, z, yaw, r))

    def create_ring_verts(node):
        verts = []
        for i in range(DIV):
            a = (np.pi*2) * (i/DIV)
            x = node.x + np.cos(a) * node.r
            y = node.y + np.sin(a) * node.r
            verts.append((x, y, node.z))
        return verts

    prev_ring = create_ring_verts(nodes[0])
    verts = verts + prev_ring
    for node in nodes[1:]:
        vert_start = len(verts) - DIV
        ring = create_ring_verts(node)
        verts = verts + ring
        prev_ring = ring

        for i in range(DIV):
            vi = vert_start + i
            vip = vert_start + ((i+1) % DIV)
            faces.append((vi, vip, vip+DIV, vi+DIV))

    #close main stem
    last_ring_start = len(verts) - DIV
    verts.append((nodes[-1].x, nodes[-1].y, nodes[-1].z + 0.01))
    for i in range(DIV):
        vi = last_ring_start + i
        vip = last_ring_start + ((i+1) % DIV)
        faces.append((vi, vip, len(verts)-1))

    #Define mesh and object
    mesh = bpy.data.meshes.new("Branch1")

    #Create mesh
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)

    object = bpy.data.objects.new("Branch1", mesh)

    collection.objects.link(object)

    mat = bpy.data.materials.get("Branch1")
    object.data.materials.append(mat)

    return object, nodes



# Generate leaf, fruit, and flower formations

def filter_prefix(objects, prefix):
    return list(
        filter(lambda ob: ob.name.startswith(prefix), objects))

class PrebuiltMeshes:
    def __init__(self):
        objects = bpy.data.collections['sub_stems'].all_objects
        self.sub_stems = filter_prefix(objects, 'b_')

    def get_end_stem_mesh(self, growth):
        groups = ['b_04', 'b_05', 'b_06', 'b_08' ]
        prefix = groups[np.random.choice(len(groups))]
        stems = filter_prefix(self.sub_stems, prefix)
        growths = np.array(list(map(lambda ob: ob.location[2], stems)))
        growths = growths / np.max(growths)
        idx_options = np.argsort(np.abs(growths - growth))[:5]
        return stems[np.random.choice(idx_options)]
prebuilt_meshes = PrebuiltMeshes()

def gen_end_stem(collection, location: tuple, yaw: float, growth: float):
    coll_sub_stems = bpy.data.collections['sub_stems']
    ob_stem = prebuilt_meshes.get_end_stem_mesh(growth)
    ob_fruits = filter_prefix(ob_stem.children, 'pose_fruit')
    bpy.ops.object.select_all(action='DESELECT')
    with bpy.context.temp_override(
        selected_objects=[ob_stem] + ob_fruits,
        active_object=ob_stem
    ):
        bpy.ops.object.duplicate()
    ob_stem = filter_prefix(bpy.context.selected_objects, 'b_')[0]
    ob_fruits = filter_prefix(ob_stem.children, 'pose_fruit')
    
    ob_stem.select_set(False)
    if location is not None:
        ob_stem.location = location
    if yaw is not None:
        ob_stem.rotation_euler[2] += yaw

    collection.objects.link(ob_stem)
    coll_sub_stems.objects.unlink(ob_stem)
    for ob in ob_fruits:
        collection.objects.link(ob)
        coll_sub_stems.objects.unlink(ob)

    return ob_stem, ob_fruits




def gen_plant(write_results=False, keep_materials=False):
    collection = bpy.data.collections.new('new_plant')
    bpy.context.scene.collection.children.link(collection)

    ob_main_stem, nodes = gen_main_stem(collection)

    ob_meshes = [ob_main_stem]
    ob_fruits = []

    for node in nodes[1:]:
        ob_m, ob_f = gen_end_stem(
            collection, 
            location = (node.x, node.y, node.z), 
            yaw = node.yaw,
            growth = node.z / nodes[-1].z)
        ob_meshes.append(ob_m)
        ob_fruits += ob_f

    markers = []


    def add_marker(location, type='FRUIT'):
        marker = {
            'marker_type': type,
            'translation': [location[0], location[1], location[2]],
        }
        markers.append(marker)


    for ob in ob_fruits:
        with bpy.context.temp_override(selected_objects=[ob]):
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        add_marker(ob.location)
        with bpy.context.temp_override(selected_objects=[ob]):
            bpy.ops.object.delete(use_global=True)

    with bpy.context.temp_override(
        selected_objects=ob_meshes,
        selected_editable_objects=ob_meshes,
        active_object=ob_meshes[0]
    ):
        bpy.ops.object.join()

    ob_merged = collection.objects[0]

    with bpy.context.temp_override(selected_editable_objects=[ob_merged]):
        bpy.ops.mesh.separate(type='MATERIAL')

    bpy.ops.object.select_all(action='DESELECT')
    for ob in collection.all_objects:
        # name the mesh
        ob.name = ob.material_slots[0].material.name
        # remesh by alpha contour if requested
        if args.remesh_alpha:
            remesh_object_by_alpha(ob, threshold=0.5)
        # apply alpha clipping to geometry if requested
        elif args.alpha_clip is not None:
            remove_transparent_faces(ob, args.alpha_clip, args.alpha_clip_mode)
        # setup material alpha clip for proper transparency handling
        if keep_materials and args.alpha_clip is not None:
            mat = ob.material_slots[0].material if ob.material_slots else None
            setup_material_alpha_clip(mat, args.alpha_clip)
        # remove the material (it will be added in the SDF file)
        if not keep_materials:
            ob.data.materials.clear()
        # select for the export
        ob.select_set(True)

    if write_results:
        # Determine export format (--export-format takes precedence over --format)
        export_format = args.export_format_ext or args.format
        with_textures = args.with_textures

        if export_format == 'glb':
            bpy.ops.export_scene.gltf(
                filepath=str(model_dir / 'meshes/tomato.glb'),
                use_selection=True,
                export_format='GLB',
                export_image_format='AUTO',
                export_materials='EXPORT' if with_textures else 'NONE'
            )
        elif export_format == 'fbx':
            bpy.ops.export_scene.fbx(
                filepath=str(model_dir / 'meshes/tomato.fbx'),
                use_selection=True,
                path_mode='COPY' if with_textures else 'AUTO',
                embed_textures=with_textures
            )
        elif export_format == 'usd':
            bpy.ops.wm.usd_export(
                filepath=str(model_dir / 'meshes/tomato.usd'),
                selected_objects_only=True,
                export_textures=with_textures,
                export_materials=with_textures
            )
        else:
            bpy.ops.wm.collada_export(
                filepath=str(model_dir / 'meshes/tomato.dae'),
                check_existing=True,
                selected=True
            )

        with open(model_dir / 'markers.json', 'w') as outfile:
            json.dump(markers, outfile, indent=4)
        bpy.ops.object.delete(use_global=True)

    return collection

if __name__ == '__main__':
    gen_plant(write_results=True, keep_materials=args.with_textures)

# if __name__ == '__main__':
#     if 'test_copy' in bpy.data.collections:
#         collection = bpy.data.collections['test_copy']
#     else:
#         collection = bpy.data.collections.new('test_copy')
#         bpy.context.scene.collection.children.link(collection)

#     print(gen_end_stem(collection, (0, 0, 0), 1.0))

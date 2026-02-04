import trimesh
import numpy as np
from collections.abc import Sequence
from scipy.spatial import cKDTree

def locate_neck(model: trimesh.Trimesh, 
                estimated_neck_height_percentage : float = 0.70,
                scan_range : float = 0.4):
    """Find neck position in a human 3D model, in the height range given. The smallest cross-section z is recognized as the neck.
    
    Parameters
    ----------
    model : trimesh.Trimesh
        The human 3D mesh model.
    estimated_neck_height_percentage : float
        Estimated percentage of total model height, where neck is located. The search will be performed only with in h_e +- range/2.
    scan_range : float
        The range of height to scan within. 
    
    Returns
    -------
    neck_center : nd.array
        neck cross section center coords in 3D.
    neck_area : float
        neck cross section area.

    Example
    -------
    >>> mesh = trimesh.load("your_model.stl")
    >>> neck_center, neck_area = locate_neck(mesh)    
    """

    # calculate scan range
    z_min, z_max = model.vertices[:, 2].min(), model.vertices[:, 2].max()
    z_min_scan = z_min + (z_max - z_min) * (estimated_neck_height_percentage - scan_range / 2)
    z_max_scan = z_min + (z_max - z_min) * (estimated_neck_height_percentage + scan_range / 2)

    # scan cross section area
    plane_normal = [0, 0, 1]
    zs, areas, centroids, to_3ds= [], [], [], []
    for z in np.linspace(z_min_scan, z_max_scan):
        plane_origin = (0, 0, z)
        section = model.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section:
            planars, to_3d = section.to_2D()
            zs.append(z)
            areas.append(planars.area)
            centroids.append(planars.centroid)
            to_3ds.append(to_3d)
    
    # determine z_neck: z neck should be the minimum cross-section area z
    zs = np.array(zs)
    areas = np.array(areas)
    ind_min = areas.argmin()
    z_neck = zs[ind_min]
    print(f"Neck z position: {z_neck:.1f}")

    # compute neck center in 3D and neck area
    neck_center_2d = centroids[ind_min]
    to_3d = to_3ds[ind_min]
    neck_center = trimesh.transform_points(np.array([[neck_center_2d[0], neck_center_2d[1], 0]]), to_3d)[0]
    neck_area = areas[ind_min]
    print(f"Neck center position: {neck_center}\nNeck cross-section area: {neck_area:.1f}")
    
    return neck_center, neck_area

# define a function that maps old model color to new model
def map_color(old_model, new_model):
    """Map the colors of old model to the new model, typically used after boolean operation to preserve color information.
    
    Paramteres
    ----------
    old_model : trimesh.Trimesh
        original model with color.
    new_model : trimesh.Trimesh
        model after boolean operation.

    Returns
    -------
    new_model_colored : trimesh.Trimesh
        new model with visual info mapped.
    
    Example
    -------
    >>> new_model_colored = map_color(old_model, new_model)
    """
    old_vertices = old_model.vertices.copy()
    old_colors = old_model.visual.vertex_colors.copy()

    # 3. 构建 KD-Tree 并将颜色映射回新生成的顶点
    tree = cKDTree(old_vertices)

    _, indices = tree.query(new_model.vertices)

    new_model.visual = trimesh.visual.ColorVisuals(
        mesh=new_model, 
        vertex_colors=old_colors[indices]
    )
    return new_model

def cut_through_neck(model : trimesh.Trimesh,
                     neck_center : Sequence,
                     neck_area : float,
                     tool_height : float = 2.0,
                     tool_size_ratio : float = 1.1):
    """Use a cylinder tool to cut the human 3D model through the neck into head part and body part.
    
    Parameters
    ----------
    model : trimesh.Trimesh
        The human 3D mesh model.
    neck_center : nd.array
        Neck cross section center coords in 3D.
    neck_area : float
        Neck cross section area.
    tool_height : float
        The height of the tool cylinder, which is subtracted from the model by boolean difference. 
    tool_size_ratio : float
        The ratio between tool diameter and neck cross section equivalent diameter. This parameter is to ensure that the cut can actually separate the model into two parts. 
    
    Returns
    -------
    head_part : trimesh.Trimesh
        The head part mesh.
    body_part : trimesh.Trimesh
        The body part mesh.

    Example
    -------
    >>> mesh = trimesh.load("your_model.stl")
    >>> neck_center, neck_area = locate_neck(mesh) 
    >>> head_part, body_part = cut_through_neck(mesh, neck_center, neck_area)
    """

    # create cutter mask
    cutter_mask_radius = (neck_area / np.pi ) ** 0.5 * tool_size_ratio
    cutter_mask = trimesh.creation.cylinder(radius=cutter_mask_radius, height=tool_height)
    cutter_mask.apply_translation(neck_center)

    # apply the boolean difference to cut the model at the neck
    mesh_after_cut = model.difference(cutter_mask, engine="manifold")

    # split the cut model into head and body
    parts = mesh_after_cut.split()
    parts_filtered = filter_tiny_objects(parts)
    if len(parts_filtered) == 2:
        parts_filtered = sorted(parts_filtered, key=lambda x: x.centroid[2], reverse=True) # sort by z coord, so that the first part is head (assuming head is above body)
        head_part = parts_filtered[0]
        body_part = parts_filtered[1]
    else:
        raise ValueError("The model is not splitted into two parts. Adjust `tool_size_ratio` and try again.")
    
    # map color 
    head_part = map_color(model, head_part)
    body_part = map_color(model, body_part)

    return head_part, body_part

def filter_tiny_objects(meshes, threshold=0.05, criterion='volume'):
    """Filter out tiny mesh objects

    Parameters
    ----------
    meshes : list of trimesh 
        list of trimesh
    threshold: float
        volume threshold relative to the largest object
    criterion: str
        'volume' or 'area', depending on the mesh

    Returns
    -------
    meshes_filtered : list of trimesh
        list with only large objects included
    
    Examples
    --------
    >>> meshes = mesh.split()
    >>> meshes_filtered = filter_tiny_objects(meshes)
    """

    if not meshes:
        return []
    
    # 获取所有子对象的属性值
    attrs = [getattr(m, criterion) for m in meshes]
    max_val = max(attrs)
    
    # 仅保留大于最大值一定比例的对象
    return [m for m, val in zip(meshes, attrs) if val > max_val * threshold]


def make_joint(head_part : trimesh.Trimesh, 
               body_part : trimesh.Trimesh,
               neck_center : Sequence,
               neck_area : float,
               tool_height : float = 2.0,
               padding_ratio : float = 1.3,
               male_tool_path : str = "male_tool.stl",
               female_tool_path : str = "female_tool.stl"):
    """Add the assembly joint to the head part and the body part generated by `cut_through_neck()`. The joint structures are made beforehead and loaded from .stl files. The paths to these tool models should be passed as args.
    
    Parameters
    ----------
    head_part : trimesh.Trimesh
        Head part mesh.
    body_part : trimesh.Trimesh
        Body part mesh.
    neck_center : nd.array
        Neck cross section center coords in 3D.
    neck_area : float
        Neck cross section area.
    tool_height : float
        Height of the cutter cylinder tool.
    padding_ratio : float
        The ratio between neck cross section equivalent diameter and the female tool diameter. 
    male_tool_path : str
        Path to the male tool stl model.
    female_tool_path : str
        Path to the female tool stl model.
    
    Returns
    -------
    head_slot : trimesh.Trimesh
        head part with a slot at the neck.
    body_peg : trimesh.Trimesh
        body part with a peg at the neck to assemble with the head. 

    Example
    -------
    >>> mesh = trimesh.load("your_model.stl")
    >>> neck_center, neck_area = locate_neck(mesh) 
    >>> head_part, body_part = cut_through_neck(mesh, neck_center, neck_area)
    >>> head_slot, body_peg = make_joint(head_part, body_part, neck_center, neck_area)
    """

    # load the tools
    male_tool = trimesh.load(male_tool_path)
    female_tool = trimesh.load(female_tool_path)
    
    # determine if rescaling is required
    tool_diameter = female_tool.vertices[:, 0].max() - female_tool.vertices[:, 0].min()
    section_diameter = (neck_area / np.pi) ** 0.5 * 2 
    print(f"Tool diameter is {tool_diameter:.1f}, cut section diameter is {section_diameter:.1f}")

    if tool_diameter * padding_ratio > section_diameter:
        scale = section_diameter / 1.3 / tool_diameter
        female_tool.apply_scale(scale)
        print(f"Tool size is too large, scale tool size down by a factor of {scale:.1f}.")
    else:
        scale = 1.0
        print("Tool size ok.")

    # determine the translation
    # the idea is to make female tool bottom center coincide with neck_center
    x_target, y_target, z_target = neck_center
    z_target += tool_height / 2.0 + female_tool.centroid[2] - female_tool.vertices[:, 2].min()
    translation = np.array([x_target, y_target, z_target]) - female_tool.centroid
    female_tool.apply_translation(translation)
    
    # apply female tool
    head_slot = head_part.difference(female_tool)

    head_slot = map_color(head_part, head_slot)

    # apply scaling to the male tool
    male_tool.apply_scale(scale)

    # determine translation
    # the idea is to make the bottom center of the male tool coincide with the centroid of the top surface of body_part
    # here we assume the centroid of the body top surface shares x, y with neck center
    # top_surface_centroid = neck_center
    # top_surface_centroid[2] = neck_center[2] - tool_height / 2.0
    # male_tool_bottom_center = male_tool.centroid.copy()
    # male_tool_bottom_center[2] = male_tool.vertices[:, 2].min()
    # translation = top_surface_centroid - male_tool_bottom_center

    x_target, y_target, z_target = neck_center
    z_target += - tool_height / 2.0 + male_tool.centroid[2] - male_tool.vertices[:, 2].min()
    translation = np.array([x_target, y_target, z_target]) - male_tool.centroid

    # apply translation to male tool
    male_tool.apply_translation(translation)

    # apply male tool to body part
    body_peg = body_part.union(male_tool)

    body_peg = map_color(body_part, body_peg)

    return head_slot, body_peg
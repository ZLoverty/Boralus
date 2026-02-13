from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _section_samples(
    mesh: trimesh.Trimesh,
    z_start: float,
    z_end: float,
    scan_steps: int,
) -> list[dict]:
    records: list[dict] = []
    for z in np.linspace(z_start, z_end, scan_steps):
        section = mesh.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
        if not section:
            continue
        planar, to_3d = section.to_2D()
        if planar.area <= 0:
            continue
        centroid_2d = planar.centroid
        centroid_3d = trimesh.transform_points(np.array([[centroid_2d[0], centroid_2d[1], 0.0]]), to_3d)[0]
        perimeter = float(getattr(planar, "length", 0.0))
        records.append(
            {
                "z": float(z),
                "area": float(planar.area),
                "centroid": centroid_3d,
                "perimeter": perimeter,
            }
        )
    return records


def _pick_min_area_stable(records: list[dict]) -> tuple[np.ndarray, float]:
    if not records:
        raise ValueError("No valid cross section found.")
    areas = np.array([r["area"] for r in records], dtype=float)
    zs = np.array([r["z"] for r in records], dtype=float)
    centroids = np.array([r["centroid"] for r in records], dtype=float)
    perimeters = np.array([max(r["perimeter"], 1e-6) for r in records], dtype=float)

    area_norm = (areas - areas.min()) / max(areas.max() - areas.min(), 1e-6)
    z_norm = (zs - zs.min()) / max(zs.max() - zs.min(), 1e-6)
    circularity = (4.0 * np.pi * areas) / (perimeters**2)
    circularity = np.clip(circularity, 0.0, 1.0)
    centroid_shift = np.linalg.norm(centroids - centroids.mean(axis=0), axis=1)
    shift_norm = centroid_shift / max(centroid_shift.max(), 1e-6)

    # Lower score is better: small area + low centroid drift + circular interface.
    score = 0.60 * area_norm + 0.25 * shift_norm + 0.15 * (1.0 - circularity) + 0.05 * z_norm
    idx = int(score.argmin())
    return centroids[idx], float(areas[idx])


def locate_interface_on_part(
    mesh: trimesh.Trimesh,
    region: str = "bottom",
    scan_ratio: float = 0.28,
    scan_steps: int = 80,
) -> tuple[np.ndarray, float]:
    """Locate robust socket/peg anchor on a part by scanning top/bottom interface slices."""
    z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
    height = z_max - z_min
    if height <= 0:
        raise ValueError("Invalid mesh height.")
    scan_ratio = float(np.clip(scan_ratio, 0.05, 0.95))

    if region == "bottom":
        z_start = z_min + 0.01 * height
        z_end = z_min + scan_ratio * height
    elif region == "top":
        z_start = z_max - scan_ratio * height
        z_end = z_max - 0.01 * height
    else:
        raise ValueError("region must be 'bottom' or 'top'")

    samples = _section_samples(mesh, z_start=z_start, z_end=z_end, scan_steps=scan_steps)
    center, area = _pick_min_area_stable(samples)
    return center, area


def locate_neck(
    model: trimesh.Trimesh,
    estimated_neck_height_percentage: float = 0.70,
    scan_range: float = 0.4,
    scan_steps: int = 80,
) -> tuple[np.ndarray, float]:
    """Find neck position in full character by minimum cross-section in a vertical scan window."""
    z_min, z_max = model.vertices[:, 2].min(), model.vertices[:, 2].max()
    z_min_scan = z_min + (z_max - z_min) * (estimated_neck_height_percentage - scan_range / 2)
    z_max_scan = z_min + (z_max - z_min) * (estimated_neck_height_percentage + scan_range / 2)

    samples = _section_samples(model, z_start=z_min_scan, z_end=z_max_scan, scan_steps=scan_steps)
    if not samples:
        raise ValueError("No valid cross section found. Adjust neck scan parameters.")
    # Full model uses strict minimum area.
    areas = np.array([r["area"] for r in samples], dtype=float)
    idx = int(areas.argmin())
    neck_center = samples[idx]["centroid"]
    neck_area = float(samples[idx]["area"])
    print(f"Neck center position: {neck_center}")
    print(f"Neck cross-section area: {neck_area:.3f}")
    return neck_center, neck_area


def map_color(old_model: trimesh.Trimesh, new_model: trimesh.Trimesh) -> trimesh.Trimesh:
    old_vertices = old_model.vertices.copy()
    old_colors = old_model.visual.vertex_colors.copy()

    tree = cKDTree(old_vertices)
    _, indices = tree.query(new_model.vertices)
    new_model.visual = trimesh.visual.ColorVisuals(mesh=new_model, vertex_colors=old_colors[indices])
    return new_model


def filter_tiny_objects(
    meshes: list[trimesh.Trimesh],
    threshold: float = 0.05,
    criterion: str = "volume",
) -> list[trimesh.Trimesh]:
    if not meshes:
        return []
    attrs = [getattr(m, criterion) for m in meshes]
    max_val = max(attrs)
    return [m for m, val in zip(meshes, attrs) if val > max_val * threshold]


def cut_through_neck(
    model: trimesh.Trimesh,
    neck_center: Sequence[float],
    neck_area: float,
    tool_height: float = 2.0,
    tool_size_ratio: float = 1.1,
    boolean_engine: str | None = "manifold",
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    
    parts = model.split(only_watertight=False)
    print(f"Model has {len(parts)} connected components before cut.")
    cutter_mask_radius = (neck_area / np.pi) ** 0.5 * tool_size_ratio
    cutter_mask = trimesh.creation.cylinder(radius=cutter_mask_radius, height=tool_height)
    cutter_mask.apply_translation(neck_center)

    mesh_after_cut = model.difference(cutter_mask, engine=boolean_engine)
    parts_filtered = filter_tiny_objects(mesh_after_cut.split())
    if len(parts_filtered) != 2:
        print(f"Warning: Expected 2 parts after cut, got {len(parts_filtered)} parts.")
        raise ValueError("Model was not split into two parts. Try a larger tool_size_ratio.")

    parts_filtered = sorted(parts_filtered, key=lambda x: x.centroid[2], reverse=True)
    head_part = map_color(model, parts_filtered[0])
    body_part = map_color(model, parts_filtered[1])
    return head_part, body_part


def compute_joint_anchors(
    head_part: trimesh.Trimesh,
    body_part: trimesh.Trimesh,
    fallback_center: Sequence[float],
    fallback_area: float,
    scan_ratio: float = 0.28,
    scan_steps: int = 80,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute robust head-bottom and body-top anchors, with safe fallback."""
    try:
        head_center, head_area = locate_interface_on_part(
            head_part, region="bottom", scan_ratio=scan_ratio, scan_steps=scan_steps
        )
    except Exception:
        head_center = np.array(fallback_center, dtype=float)
        head_area = float(fallback_area)

    try:
        body_center, body_area = locate_interface_on_part(
            body_part, region="top", scan_ratio=scan_ratio, scan_steps=scan_steps
        )
    except Exception:
        body_center = np.array(fallback_center, dtype=float)
        body_area = float(fallback_area)

    interface_area = float(min(head_area, body_area))
    return head_center, body_center, interface_area


def make_joint(
    head_part: trimesh.Trimesh,
    body_part: trimesh.Trimesh,
    head_anchor_center: Sequence[float],
    body_anchor_center: Sequence[float],
    interface_area: float,
    tool_height: float = 2.0,
    padding_ratio: float = 1.3,
    male_tool_path: str = "male_tool.stl",
    female_tool_path: str = "female_tool.stl",
    boolean_engine: str | None = "manifold",
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    male_tool = trimesh.load(male_tool_path)
    female_tool = trimesh.load(female_tool_path)

    tool_diameter = female_tool.vertices[:, 0].max() - female_tool.vertices[:, 0].min()
    section_diameter = (interface_area / np.pi) ** 0.5 * 2
    print(f"Tool diameter: {tool_diameter:.3f}, target interface diameter: {section_diameter:.3f}")

    if tool_diameter * padding_ratio > section_diameter:
        scale = section_diameter / max(1.3 * tool_diameter, 1e-6)
        female_tool.apply_scale(scale)
        print(f"Scale down tools by {scale:.3f}")
    else:
        scale = 1.0
        print("Tool size accepted.")

    hx, hy, hz = np.asarray(head_anchor_center, dtype=float)
    hz += tool_height / 2.0 + female_tool.centroid[2] - female_tool.vertices[:, 2].min()
    female_tool.apply_translation(np.array([hx, hy, hz]) - female_tool.centroid)
    head_slot = map_color(head_part, head_part.difference(female_tool, engine=boolean_engine))

    male_tool.apply_scale(scale)
    bx, by, bz = np.asarray(body_anchor_center, dtype=float)
    bz += -tool_height / 2.0 + male_tool.centroid[2] - male_tool.vertices[:, 2].min()
    male_tool.apply_translation(np.array([bx, by, bz]) - male_tool.centroid)
    body_peg = map_color(body_part, body_part.union(male_tool, engine=boolean_engine))
    return head_slot, body_peg


def _to_mesh(geometry: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(geometry, trimesh.Trimesh):
        return geometry
    if isinstance(geometry, trimesh.Scene):
        meshes = [g for g in geometry.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Scene contains no mesh geometry.")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported geometry type: {type(geometry)}")


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    return _to_mesh(trimesh.load(path))


def keep_largest_component(
    mesh: trimesh.Trimesh,
    criterion: str = "faces",
) -> tuple[trimesh.Trimesh, int]:
    """
    Keep only the largest connected component and drop all smaller components.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh that may contain disconnected components.
    criterion : str
        Ranking metric for "largest". Recommended: 'faces' (robust for non-watertight meshes).
        Other valid values include 'volume' and 'area'.

    Returns
    -------
    largest_component : trimesh.Trimesh
        Mesh with only the largest connected component.
    removed_count : int
        Number of removed components.
    """
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh, 0

    if criterion == "faces":
        scores = [len(c.faces) for c in components]
    else:
        scores = [getattr(c, criterion) for c in components]

    largest_idx = int(np.argmax(scores))
    largest = components[largest_idx]
    largest = map_color(mesh, largest)
    removed_count = len(components) - 1
    return largest, removed_count


def get_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = mesh.visual.vertex_colors
    if colors is None or len(colors) == 0:
        raise ValueError("Input mesh has no vertex colors to split by color.")
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        alpha = np.full((colors.shape[0], 1), 255, dtype=colors.dtype)
        colors = np.hstack([colors, alpha])
    return colors.astype(np.uint8)


def quantize_vertex_colors(colors_rgba: np.ndarray, tolerance: int = 8) -> np.ndarray:
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    rgb = colors_rgba[:, :3].astype(np.int16)
    qrgb = ((rgb + tolerance // 2) // tolerance) * tolerance
    return np.clip(qrgb, 0, 255).astype(np.uint8)


def split_mesh_by_color(
    mesh: trimesh.Trimesh,
    tolerance: int = 8,
    min_faces: int = 50,
) -> list[tuple[np.ndarray, trimesh.Trimesh]]:
    qrgb = quantize_vertex_colors(get_vertex_colors(mesh), tolerance=tolerance)
    face_rgb = qrgb[mesh.faces].mean(axis=1).round().astype(np.uint8)
    palette, face_color_ids = np.unique(face_rgb, axis=0, return_inverse=True)

    pieces: list[tuple[np.ndarray, trimesh.Trimesh]] = []
    for color_id, color_rgb in enumerate(palette):
        face_idx = np.where(face_color_ids == color_id)[0]
        if len(face_idx) < min_faces:
            continue
        part = mesh.submesh([face_idx], append=True, repair=False)
        if part is None or len(part.faces) == 0:
            continue
        pieces.append((color_rgb, part))
    return pieces


def _semantic_zone_label(
    centroid: np.ndarray,
    bounds: np.ndarray,
    face_count: int,
    part_kind: str,
) -> str:
    mins = bounds[0]
    maxs = bounds[1]
    span = np.maximum(maxs - mins, 1e-6)
    norm = (centroid - mins) / span
    x_n, _, z_n = norm

    if part_kind == "head":
        if z_n < 0.22:
            base = "neck_interface"
        elif z_n > 0.82:
            base = "crown"
        elif x_n < 0.25:
            base = "left_side"
        elif x_n > 0.75:
            base = "right_side"
        else:
            base = "mid_head"
    else:
        if z_n > 0.82:
            base = "upper_body_interface"
        elif z_n < 0.25:
            base = "lower_body"
        elif x_n < 0.3:
            base = "left_body"
        elif x_n > 0.7:
            base = "right_body"
        else:
            base = "body_center"

    if face_count < 400:
        return f"{base}_detail"
    return base


def split_mesh_semantic(
    mesh: trimesh.Trimesh,
    part_kind: str,
    tolerance: int = 8,
    min_faces: int = 50,
) -> list[tuple[str, np.ndarray, trimesh.Trimesh]]:
    """Heuristic semantic split: color clustering + connected components + spatial labeling."""
    bounds = mesh.bounds.copy()
    semantic_parts: list[tuple[str, np.ndarray, trimesh.Trimesh]] = []

    for rgb, color_chunk in split_mesh_by_color(mesh, tolerance=tolerance, min_faces=min_faces):
        components = color_chunk.split(only_watertight=False)
        for comp in components:
            if len(comp.faces) < min_faces:
                continue
            label = _semantic_zone_label(comp.centroid, bounds=bounds, face_count=len(comp.faces), part_kind=part_kind)
            semantic_parts.append((label, rgb, comp))
    return semantic_parts


def export_color_parts(
    mesh: trimesh.Trimesh,
    out_dir: str | Path,
    prefix: str,
    tolerance: int = 8,
    min_faces: int = 50,
    file_type: str = "stl",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for idx, (rgb, part) in enumerate(
        split_mesh_by_color(mesh, tolerance=tolerance, min_faces=min_faces),
        start=1,
    ):
        name = (
            f"{prefix}_part_{idx:02d}_rgb_{int(rgb[0]):03d}_{int(rgb[1]):03d}_{int(rgb[2]):03d}.{file_type}"
        )
        target = out_dir / name
        part.export(target)
        exported.append(target)
    return exported


def export_semantic_parts(
    mesh: trimesh.Trimesh,
    out_dir: str | Path,
    prefix: str,
    part_kind: str,
    tolerance: int = 8,
    min_faces: int = 50,
    file_type: str = "stl",
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict] = []
    for idx, (label, rgb, part) in enumerate(
        split_mesh_semantic(mesh, part_kind=part_kind, tolerance=tolerance, min_faces=min_faces),
        start=1,
    ):
        name = (
            f"{prefix}_{label}_{idx:02d}_rgb_{int(rgb[0]):03d}_{int(rgb[1]):03d}_{int(rgb[2]):03d}.{file_type}"
        )
        target = out_dir / name
        part.export(target)
        exported.append({"label": label, "rgb": [int(x) for x in rgb], "path": str(target.resolve())})
    return exported


def load_vertex_labels_json(
    json_path: str | Path,
    expected_vertex_count: int,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Load external semantic labels from json.

    Expected schema:
    {
      "vertex_count": 12345,
      "labels": [0, 0, 17, ...],
      "id_to_name": {"17": "hair", "4": "l_eye", ...}
    }
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    labels = np.asarray(data.get("labels", []), dtype=np.int32)
    if len(labels) != expected_vertex_count:
        raise ValueError(
            f"Label count mismatch: labels={len(labels)} vertices={expected_vertex_count}"
        )

    raw_map = data.get("id_to_name", {})
    id_to_name: dict[int, str] = {}
    for k, v in raw_map.items():
        id_to_name[int(k)] = str(v)
    return labels, id_to_name


def extract_labeled_submesh(
    mesh: trimesh.Trimesh,
    vertex_labels: np.ndarray,
    selected_label_ids: set[int],
    min_faces: int = 20,
) -> trimesh.Trimesh | None:
    """
    Extract faces whose >=2 vertices belong to selected labels.
    """
    face_labels = vertex_labels[mesh.faces]
    selected_count = np.isin(face_labels, list(selected_label_ids)).sum(axis=1)
    face_idx = np.where(selected_count >= 2)[0]
    if len(face_idx) < min_faces:
        return None
    part = mesh.submesh([face_idx], append=True, repair=False)
    if part is None or len(part.faces) < min_faces:
        return None
    return part


def export_fine_face_parts_from_labels(
    mesh: trimesh.Trimesh,
    vertex_labels: np.ndarray,
    id_to_name: dict[int, str],
    out_dir: str | Path,
    prefix: str,
    file_type: str = "stl",
    min_faces: int = 20,
) -> list[dict]:
    """
    Export precise face parts (hair/eyes/lips) from external per-vertex labels.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    normalized = {idx: name.lower().strip() for idx, name in id_to_name.items()}

    target_groups = {
        "hair": {"hair"},
        "eyes": {"l_eye", "r_eye", "left_eye", "right_eye", "eye"},
        "lips": {"u_lip", "l_lip", "upper_lip", "lower_lip", "lip", "mouth"},
    }

    exported: list[dict] = []
    for group_name, aliases in target_groups.items():
        selected = {idx for idx, name in normalized.items() if name in aliases}
        if not selected:
            continue
        part = extract_labeled_submesh(
            mesh=mesh,
            vertex_labels=vertex_labels,
            selected_label_ids=selected,
            min_faces=min_faces,
        )
        if part is None:
            continue
        target = out_path / f"{prefix}_{group_name}.{file_type}"
        part.export(target)
        exported.append(
            {
                "label": group_name,
                "label_ids": sorted(list(selected)),
                "path": str(target.resolve()),
                "faces": int(len(part.faces)),
            }
        )
    return exported

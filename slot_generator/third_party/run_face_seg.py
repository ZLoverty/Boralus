from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CELEBA_LABEL_MAP = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-view 3D face parsing: project 2D semantic labels back to mesh vertices."
    )
    parser.add_argument("--mesh", required=True, help="Input mesh path")
    parser.add_argument("--out", required=True, help="Output json path")
    parser.add_argument(
        "--torchscript-model",
        required=True,
        help="TorchScript face parsing model path (expects logits [1,C,H,W])",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Render/inference image size")
    parser.add_argument("--yaw-count", type=int, default=8, help="Number of yaw views")
    parser.add_argument("--pitch-deg", type=float, default=8.0, help="Pitch angle in degrees")
    parser.add_argument("--fov-deg", type=float, default=30.0, help="Perspective FOV in degrees")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda/cpu")
    parser.add_argument("--occlusion-threshold", type=float, default=0.02, help="Depth tolerance")
    parser.add_argument("--smooth-k", type=int, default=12, help="KNN smoothing neighborhood size")
    parser.add_argument("--smooth-steps", type=int, default=1, help="KNN smoothing iterations")
    parser.add_argument("--debug-dir", default="", help="Optional directory to save debug masks")
    return parser


def load_mesh(path: str | Path):
    import trimesh

    geom = trimesh.load(path)
    if isinstance(geom, trimesh.Trimesh):
        return geom
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No mesh in scene.")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported mesh type: {type(geom)}")


def ensure_vertex_colors(mesh):
    import numpy as np

    colors = mesh.visual.vertex_colors
    if colors is None or len(colors) == 0:
        white = np.full((len(mesh.vertices), 4), 255, dtype=np.uint8)
        mesh.visual.vertex_colors = white
        return white
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        alpha = np.full((colors.shape[0], 1), 255, dtype=colors.dtype)
        colors = np.hstack([colors, alpha])
        mesh.visual.vertex_colors = colors
    return colors.astype(np.uint8)


def look_at(camera_pos, target, up):
    import numpy as np

    forward = target - camera_pos
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-9)
    true_up = np.cross(right, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-9)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_pos
    return pose


def make_camera_poses(
    mesh,
    yaw_count: int,
    pitch_deg: float,
) -> tuple:
    import numpy as np

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    radius = float(np.linalg.norm(extent)) * 0.9
    pitch = math.radians(pitch_deg)

    poses: list[np.ndarray] = []
    for i in range(yaw_count):
        yaw = 2.0 * math.pi * i / max(yaw_count, 1)
        x = center[0] + radius * math.cos(yaw) * math.cos(pitch)
        y = center[1] + radius * math.sin(yaw) * math.cos(pitch)
        z = center[2] + radius * math.sin(pitch)
        poses.append(look_at(np.array([x, y, z]), center, np.array([0.0, 0.0, 1.0])))
    return center, poses


class TorchScriptFaceParser:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def parse(self, image_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        import numpy as np

        img = image_rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # Common normalization for face parsing models.
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            labels = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int32)
        return labels


def project_vertices(
    vertices_world,
    camera_pose,
    yfov_rad: float,
    width: int,
    height: int,
) -> tuple:
    import numpy as np

    w2c = np.linalg.inv(camera_pose)
    v_h = np.hstack([vertices_world, np.ones((len(vertices_world), 1), dtype=np.float64)])
    v_cam = (w2c @ v_h.T).T[:, :3]

    z = v_cam[:, 2]
    visible_front = z < -1e-6
    x = v_cam[:, 0]
    y = v_cam[:, 1]
    aspect = width / max(height, 1)
    tan_half = math.tan(yfov_rad / 2.0)
    x_ndc = x / (-z * tan_half * aspect + 1e-9)
    y_ndc = y / (-z * tan_half + 1e-9)
    u = ((x_ndc + 1.0) * 0.5 * (width - 1)).round().astype(np.int32)
    v = ((1.0 - y_ndc) * 0.5 * (height - 1)).round().astype(np.int32)

    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid = visible_front & in_frame
    depth = -z
    return u, v, depth, valid


def smooth_vertex_labels(labels, vertices, k: int, steps: int):
    import numpy as np

    if k <= 1 or steps <= 0:
        return labels
    from scipy.spatial import cKDTree

    out = labels.copy()
    tree = cKDTree(vertices)
    _, idx = tree.query(vertices, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]

    for _ in range(steps):
        next_out = out.copy()
        for i in range(len(out)):
            nn = out[idx[i]]
            vals, counts = np.unique(nn, return_counts=True)
            next_out[i] = vals[counts.argmax()]
        out = next_out
    return out


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import pyrender

    mesh = load_mesh(args.mesh)
    ensure_vertex_colors(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    parser = TorchScriptFaceParser(args.torchscript_model, device=args.device)
    center, poses = make_camera_poses(mesh, yaw_count=args.yaw_count, pitch_deg=args.pitch_deg)
    _ = center

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.02, 0.02, 0.02])
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    mesh_node = scene.add(pr_mesh)
    camera = pyrender.PerspectiveCamera(yfov=math.radians(args.fov_deg))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_node = scene.add(camera, pose=np.eye(4))
    light_node = scene.add(light, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(args.image_size, args.image_size)

    votes = np.zeros((len(vertices), len(CELEBA_LABEL_MAP)), dtype=np.int32)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for i, pose in enumerate(poses):
        scene.set_pose(cam_node, pose)
        scene.set_pose(light_node, pose)
        color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
        rgb = color[..., :3]
        labels_2d = parser.parse(rgb)

        u, v, d, valid = project_vertices(
            vertices_world=vertices,
            camera_pose=pose,
            yfov_rad=math.radians(args.fov_deg),
            width=args.image_size,
            height=args.image_size,
        )
        sample_depth = depth[v[valid], u[valid]]
        occ_ok = np.abs(sample_depth - d[valid]) < args.occlusion_threshold
        valid_idx = np.where(valid)[0][occ_ok]
        if len(valid_idx) > 0:
            sampled_labels = labels_2d[v[valid_idx], u[valid_idx]]
            sampled_labels = np.clip(sampled_labels, 0, len(CELEBA_LABEL_MAP) - 1)
            votes[valid_idx, sampled_labels] += 1

        if debug_dir is not None:
            dbg = np.zeros_like(labels_2d, dtype=np.uint8)
            dbg[np.where(labels_2d == 17)] = 255
            (debug_dir / f"view_{i:02d}_hair_mask.npy").write_bytes(dbg.tobytes())

    renderer.delete()
    scene.remove_node(mesh_node)
    scene.remove_node(cam_node)
    scene.remove_node(light_node)

    labels = votes.argmax(axis=1).astype(np.int32)
    unseen = votes.sum(axis=1) == 0
    labels[unseen] = 0
    labels = smooth_vertex_labels(
        labels=labels, vertices=vertices, k=args.smooth_k, steps=args.smooth_steps
    )

    out = {
        "vertex_count": int(len(vertices)),
        "labels": labels.tolist(),
        "id_to_name": {str(k): v for k, v in CELEBA_LABEL_MAP.items()},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Saved labels json: {out_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

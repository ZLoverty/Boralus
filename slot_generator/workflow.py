from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Automated workflow: semantic split + robust bottom-of-head slot anchoring."
    )
    parser.add_argument("--input", required=True, help="Input mesh path, e.g. .obj/.ply/.glb/.stl")
    parser.add_argument("--out-dir", default=str(here / "outputs"), help="Output directory")
    parser.add_argument("--male-tool", default=str(here / "male_tool.stl"), help="Male connector tool STL")
    parser.add_argument("--female-tool", default=str(here / "female_tool.stl"), help="Female connector tool STL")
    parser.add_argument("--boolean-engine", default="manifold", help="Boolean engine, e.g. manifold/blender")

    parser.add_argument("--neck-height-pct", type=float, default=0.70, help="Estimated neck height ratio")
    parser.add_argument("--neck-scan-range", type=float, default=0.40, help="Search range around neck height")
    parser.add_argument("--neck-scan-steps", type=int, default=80, help="Neck scan steps")

    parser.add_argument("--anchor-scan-ratio", type=float, default=0.28, help="Top/bottom scan range ratio")
    parser.add_argument("--anchor-scan-steps", type=int, default=80, help="Top/bottom scan steps")

    parser.add_argument("--cut-tool-height", type=float, default=2.0, help="Neck cutter cylinder height")
    parser.add_argument("--cut-tool-size-ratio", type=float, default=1.1, help="Cutter size ratio")
    parser.add_argument("--joint-padding-ratio", type=float, default=1.3, help="Joint fit padding ratio")

    parser.add_argument("--color-tolerance", type=int, default=8, help="RGB quantization tolerance")
    parser.add_argument("--min-faces", type=int, default=50, help="Ignore tiny components")
    parser.add_argument("--split-mode", choices=["color", "semantic", "both"], default="semantic")
    parser.add_argument("--export-type", default="stl", help="Export type, e.g. stl/ply/obj")

    parser.add_argument(
        "--external-seg-cmd",
        default="",
        help=(
            "External segmentation command template. Use placeholders: "
            "{input_mesh} {output_json}. Example: "
            "python third_party\\run_seg.py --mesh {input_mesh} --out {output_json}"
        ),
    )
    parser.add_argument(
        "--external-labels-json",
        default="",
        help="Use existing external labels json directly (skip command execution).",
    )
    parser.add_argument(
        "--fine-face-min-faces",
        type=int,
        default=20,
        help="Minimum faces for exported fine face parts (hair/eyes/lips).",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    from utils import (
        compute_joint_anchors,
        cut_through_neck,
        export_color_parts,
        export_fine_face_parts_from_labels,
        export_semantic_parts,
        keep_largest_component,
        load_mesh,
        load_vertex_labels_json,
        locate_neck,
        make_joint,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.input)
    components_before = len(mesh.split(only_watertight=False))
    mesh, removed_components = keep_largest_component(mesh)
    components_after = len(mesh.split(only_watertight=False))
    print(f"Loaded mesh: {args.input}")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(
        f"Pre-check components: before={components_before}, after={components_after}, removed={removed_components}"
    )

    neck_center, neck_area = locate_neck(
        mesh,
        estimated_neck_height_percentage=args.neck_height_pct,
        scan_range=args.neck_scan_range,
        scan_steps=args.neck_scan_steps,
    )

    head_part, body_part = cut_through_neck(
        mesh,
        neck_center=neck_center,
        neck_area=neck_area,
        tool_height=args.cut_tool_height,
        tool_size_ratio=args.cut_tool_size_ratio,
        boolean_engine=args.boolean_engine,
    )

    head_anchor, body_anchor, interface_area = compute_joint_anchors(
        head_part,
        body_part,
        fallback_center=neck_center,
        fallback_area=neck_area,
        scan_ratio=args.anchor_scan_ratio,
        scan_steps=args.anchor_scan_steps,
    )
    print(f"Head anchor: {head_anchor}, Body anchor: {body_anchor}, Interface area: {interface_area:.3f}")

    head_slot, body_peg = make_joint(
        head_part=head_part,
        body_part=body_part,
        head_anchor_center=head_anchor,
        body_anchor_center=body_anchor,
        interface_area=interface_area,
        tool_height=args.cut_tool_height,
        padding_ratio=args.joint_padding_ratio,
        male_tool_path=args.male_tool,
        female_tool_path=args.female_tool,
        boolean_engine=args.boolean_engine,
    )

    head_full = out_dir / f"head_with_slot.{args.export_type}"
    body_full = out_dir / f"body_with_peg.{args.export_type}"
    head_slot.export(head_full)
    body_peg.export(body_full)

    summary: dict = {
        "input": str(Path(args.input).resolve()),
        "neck_center": [float(x) for x in neck_center],
        "neck_area": float(neck_area),
        "head_anchor": [float(x) for x in head_anchor],
        "body_anchor": [float(x) for x in body_anchor],
        "interface_area": float(interface_area),
        "head_full": str(head_full.resolve()),
        "body_full": str(body_full.resolve()),
        "split_mode": args.split_mode,
        "color_tolerance": args.color_tolerance,
        "min_faces": args.min_faces,
    }

    if args.split_mode in ("color", "both"):
        head_color_parts = export_color_parts(
            head_slot,
            out_dir=out_dir / "head_color_parts",
            prefix="head",
            tolerance=args.color_tolerance,
            min_faces=args.min_faces,
            file_type=args.export_type,
        )
        body_color_parts = export_color_parts(
            body_peg,
            out_dir=out_dir / "body_color_parts",
            prefix="body",
            tolerance=args.color_tolerance,
            min_faces=args.min_faces,
            file_type=args.export_type,
        )
        summary["head_color_parts"] = [str(p.resolve()) for p in head_color_parts]
        summary["body_color_parts"] = [str(p.resolve()) for p in body_color_parts]
        print(f"Head color parts: {len(head_color_parts)}")
        print(f"Body color parts: {len(body_color_parts)}")

    if args.split_mode in ("semantic", "both"):
        head_semantic_parts = export_semantic_parts(
            head_slot,
            out_dir=out_dir / "head_semantic_parts",
            prefix="head",
            part_kind="head",
            tolerance=args.color_tolerance,
            min_faces=args.min_faces,
            file_type=args.export_type,
        )
        body_semantic_parts = export_semantic_parts(
            body_peg,
            out_dir=out_dir / "body_semantic_parts",
            prefix="body",
            part_kind="body",
            tolerance=args.color_tolerance,
            min_faces=args.min_faces,
            file_type=args.export_type,
        )
        summary["head_semantic_parts"] = head_semantic_parts
        summary["body_semantic_parts"] = body_semantic_parts
        print(f"Head semantic parts: {len(head_semantic_parts)}")
        print(f"Body semantic parts: {len(body_semantic_parts)}")

    external_labels_json = args.external_labels_json.strip()
    if not external_labels_json and args.external_seg_cmd.strip():
        ext_dir = out_dir / "external_seg"
        ext_dir.mkdir(parents=True, exist_ok=True)
        mesh_for_external = ext_dir / "head_for_external.ply"
        output_json = ext_dir / "head_vertex_labels.json"
        head_slot.export(mesh_for_external)

        cmd = args.external_seg_cmd.format(
            input_mesh=str(mesh_for_external.resolve()),
            output_json=str(output_json.resolve()),
        )
        print(f"Running external segmentation: {cmd}")
        subprocess.run(cmd, check=True, shell=True)
        external_labels_json = str(output_json)

    if external_labels_json:
        labels, id_to_name = load_vertex_labels_json(
            external_labels_json, expected_vertex_count=len(head_slot.vertices)
        )
        fine_parts = export_fine_face_parts_from_labels(
            mesh=head_slot,
            vertex_labels=labels,
            id_to_name=id_to_name,
            out_dir=out_dir / "head_fine_face_parts",
            prefix="head",
            file_type=args.export_type,
            min_faces=args.fine_face_min_faces,
        )
        summary["external_labels_json"] = str(Path(external_labels_json).resolve())
        summary["head_fine_face_parts"] = fine_parts
        print(f"Fine face parts (hair/eyes/lips): {len(fine_parts)}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Export complete: {out_dir}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

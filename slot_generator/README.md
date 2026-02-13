# Slot Generator Workflow

Automated pipeline for colored human models:
1. Locate neck automatically.
2. Cut model into head/body.
3. Locate robust interface anchors (head bottom + body top) for joint placement.
4. Add female slot on head and male peg on body.
5. Split both parts into print pieces by color and/or semantic zones.

## Requirements

- Python 3.10+
- `trimesh`
- `numpy`
- `scipy`
- Boolean backend for trimesh (recommended: `manifold3d`)

Install example:

```powershell
pip install trimesh numpy scipy manifold3d
```

## Quick Start

From `slot_generator` directory:

```powershell
python workflow.py --input girl_model.stl --out-dir outputs\girl
```

Semantic split (recommended):

```powershell
python workflow.py --input girl_model.stl --split-mode semantic --out-dir outputs\girl_sem
```

Fine face parts via external model (hair/eyes/lips):

```powershell
python workflow.py `
  --input girl_model.stl `
  --out-dir outputs\girl_fine `
  --external-seg-cmd "python third_party\run_face_seg.py --mesh {input_mesh} --out {output_json} --torchscript-model models\face_parsing.ts" `
  --export-type stl
```

## Useful Parameters

```powershell
python workflow.py `
  --input your_model.obj `
  --out-dir outputs\case01 `
  --split-mode both `
  --color-tolerance 10 `
  --min-faces 120 `
  --neck-height-pct 0.68 `
  --neck-scan-range 0.35 `
  --anchor-scan-ratio 0.30 `
  --cut-tool-size-ratio 1.12 `
  --export-type stl
```

## Output Structure

- `head_with_slot.stl`: complete head with socket.
- `body_with_peg.stl`: complete body with peg.
- `head_color_parts/*.stl`: head split by color (`--split-mode color/both`).
- `body_color_parts/*.stl`: body split by color (`--split-mode color/both`).
- `head_semantic_parts/*.stl`: head split by semantic zones (`--split-mode semantic/both`).
- `body_semantic_parts/*.stl`: body split by semantic zones (`--split-mode semantic/both`).
- `head_fine_face_parts/head_hair.stl|head_eyes.stl|head_lips.stl`: exported from external per-vertex labels.
- `summary.json`: neck position, exported files, and runtime parameters.

## External Tool Contract

To enable precise `hair/eyes/lips`, external tool must output per-vertex labels json:

```json
{
  "vertex_count": 12345,
  "labels": [0, 0, 17, 17, 4, 5, 12, 13],
  "id_to_name": {
    "4": "l_eye",
    "5": "r_eye",
    "12": "u_lip",
    "13": "l_lip",
    "17": "hair"
  }
}
```

Notes:
- `labels.length` must equal head mesh vertex count used for external inference.
- `id_to_name` supports aliases for eyes/lips/hair (`l_eye`, `r_eye`, `u_lip`, `l_lip`, `hair`, etc.).
- You can pass an existing json directly with `--external-labels-json`.

### Included helper

This repo now includes `slot_generator/third_party/run_face_seg.py`, which:
- renders the head mesh from multi-view angles (`pyrender`)
- runs a TorchScript face parsing model per view (`torch`)
- projects 2D labels back to 3D vertices with depth/occlusion checks
- outputs the required label json schema

Install dependencies for helper script:

```powershell
pip install torch trimesh pyrender pyglet PyOpenGL scipy numpy
```

## Notes

- If neck split fails, increase `--cut-tool-size-ratio` slightly (for example from `1.10` to `1.18`).
- If slot position is unstable, increase `--anchor-scan-ratio` (for example `0.28 -> 0.35`) or `--anchor-scan-steps`.
- If colors are over-split into too many tiny parts, increase `--color-tolerance` and `--min-faces`.
- If boolean fails, try another backend via `--boolean-engine blender` (requires Blender available to trimesh).

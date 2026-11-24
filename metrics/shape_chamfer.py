#!/usr/bin/env python3
"""Per-instance Chamfer Distance between generated GLB and reference GLB.

Inputs:
    reference: data/<inst>/glb/<inst>.glb
    generated: results/<task>/<model>/<inst>/glb/<inst>.glb

Pipeline (per instance):
    1. Load both meshes with trimesh; concatenate scene meshes if any.
    2. Surface-sample N points each (default 8192) with a fixed RNG.
    3. Normalize each cloud independently:
         - center: subtract centroid (mean of sampled points)
         - scale:  divide by max distance to origin (unit-sphere fit)
    4. Compute Chamfer Distance = mean(||r->g||^2) + mean(||g->r||^2),
       where each direction uses cKDTree nearest neighbor.
    5. Yaw-aligned variant: enumerate 4 yaw rotations of the generated
       cloud (0/90/180/270 deg around Y, matching glTF Y-up export) and
       report the minimum CD.
       This is the shape-side analogue of the Hungarian matching used
       in image_similarity.py to absorb canonical-orientation mismatch.

Aggregation (matches image_similarity.py story):
    conditional_mean — averaged over instances where the generated mesh
                       loaded successfully.
    penalized_mean   — failures count as the per-task fallback CD (the
                       worst observed CD across all valid instances of
                       this run, * 1.5). This keeps the metric finite
                       and bounded for instances with no GLB at all.

Output:
    <model_dir>/_metrics/shape_chamfer.json

Usage (one model):
    python eval/metrics/shape_chamfer.py \\
        --results-root results/text_to_3D \\
        --model gemini-3-flash-preview

Usage (all main-run dirs):
    for r in results/text_to_3D results/image_to_3D; do
      for m in $r/*/; do
        python eval/metrics/shape_chamfer.py --results-root "$r" \\
               --model "$(basename $m)"
      done
    done
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

EVAL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = EVAL_ROOT / "data"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",        required=True)
    p.add_argument("--results-root", type=Path, required=True,
                   help="e.g. results/text_to_3D")
    p.add_argument("--data-root",    type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--n-points",     type=int, default=8192)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--instances",    nargs="*", default=None)
    p.add_argument("--limit",        type=int, default=None,
                   help="Cap number of instances (debug).")
    return p.parse_args()


def load_mesh_points(glb_path, n_points, rng):
    """Load a GLB and return n_points sampled on its surface, or None."""
    try:
        scene_or_mesh = trimesh.load(glb_path, force="mesh")
    except Exception:
        return None
    if scene_or_mesh is None or scene_or_mesh.is_empty:
        return None
    mesh = scene_or_mesh
    if mesh.faces is None or len(mesh.faces) == 0:
        return None
    # Surface-sample.
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(0, 2**31-1)))
    except Exception:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < n_points // 4:
        return None
    return pts


def normalize_unit_sphere(pts):
    """Center at centroid, scale so max ||p|| = 1."""
    pts = pts - pts.mean(0, keepdims=True)
    r = np.linalg.norm(pts, axis=1).max()
    if r < 1e-9:
        return pts
    return pts / r


def chamfer_squared(a, b):
    """Symmetric mean squared Chamfer:
        CD = mean_i min_j ||a_i - b_j||^2 + mean_j min_i ||a_i - b_j||^2.
    """
    tree_b = cKDTree(b)
    da, _  = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    db, _  = tree_a.query(b, k=1)
    return float((da ** 2).mean() + (db ** 2).mean())


def yaw_rotated(pts, deg):
    """Rotate Y-up glTF point cloud around vertical Y axis by `deg` degrees."""
    if deg == 0:
        return pts
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    # core/export_glb.py exports GLBs with export_yup=True (glTF Y-up),
    # and trimesh.load preserves Y-up coordinates, so horizontal yaw rotates around Y.
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=pts.dtype)
    return pts @ R.T


def chamfer_with_yaw(ref_pts, gen_pts):
    """CD without rotation, plus the min over 4 yaw rotations of gen."""
    cd0 = chamfer_squared(ref_pts, gen_pts)
    cd_yaw = [cd0]
    for deg in (90, 180, 270):
        cd_yaw.append(chamfer_squared(ref_pts, yaw_rotated(gen_pts, deg)))
    cd_yaw_min = float(min(cd_yaw))
    cd_yaw_argmin = int([0, 90, 180, 270][int(np.argmin(cd_yaw))])
    return cd0, cd_yaw_min, cd_yaw_argmin, cd_yaw


def list_instances(model_dir, override=None):
    insts = sorted(d for d in model_dir.iterdir()
                   if d.is_dir() and not d.name.startswith("_"))
    if override:
        keep = set(override)
        insts = [d for d in insts if d.name in keep]
    return insts


def main():
    args = parse_args()
    model_dir = args.results_root / args.model
    if not model_dir.exists():
        raise SystemExit(f"No such model dir: {model_dir}")

    insts = list_instances(model_dir, args.instances)
    if args.limit:
        insts = insts[:args.limit]
    if not insts:
        raise SystemExit("No instances.")
    print(f"Model: {args.model}  ({len(insts)} instances)")
    print(f"Sampling {args.n_points} points per cloud, normalize unit-sphere, "
          f"4-yaw alignment.")

    rng = np.random.default_rng(args.seed)
    per_instance = []
    t0 = time.time()
    for i, inst in enumerate(insts):
        ref_glb = args.data_root / inst.name / "glb" / f"{inst.name}.glb"
        gen_glb = inst / "glb" / f"{inst.name}.glb"

        rec = {"instance": inst.name, "ref_exists": ref_glb.exists(),
               "gen_exists": gen_glb.exists(),
               "cd": None, "cd_yawmin": None, "yaw_argmin": None,
               "cd_per_yaw": None, "n_ref_pts": None, "n_gen_pts": None,
               "status": None}

        if not gen_glb.exists():
            rec["status"] = "NO_GEN_GLB"
        elif not ref_glb.exists():
            rec["status"] = "NO_REF_GLB"
        else:
            ref_pts = load_mesh_points(ref_glb, args.n_points, rng)
            gen_pts = load_mesh_points(gen_glb, args.n_points, rng)
            if ref_pts is None:
                rec["status"] = "BAD_REF_MESH"
            elif gen_pts is None:
                rec["status"] = "BAD_GEN_MESH"
            else:
                ref_pts = normalize_unit_sphere(ref_pts)
                gen_pts = normalize_unit_sphere(gen_pts)
                cd0, cd_min, yaw, cd_yaw = chamfer_with_yaw(ref_pts, gen_pts)
                rec.update({
                    "cd":          cd0,
                    "cd_yawmin":   cd_min,
                    "yaw_argmin":  yaw,
                    "cd_per_yaw":  cd_yaw,
                    "n_ref_pts":   int(ref_pts.shape[0]),
                    "n_gen_pts":   int(gen_pts.shape[0]),
                    "status":      "OK",
                })
        per_instance.append(rec)
        if (i + 1) % 25 == 0 or i == len(insts) - 1:
            ok = sum(1 for r in per_instance if r["status"] == "OK")
            print(f"  [{i+1}/{len(insts)}]  ok={ok}  "
                  f"elapsed={time.time()-t0:.1f}s")

    n = len(per_instance)
    ok_rows = [r for r in per_instance if r["status"] == "OK"]
    n_ok = len(ok_rows)

    # Penalty CD for missing/failed: use 1.5x worst observed within this run.
    # (Bounded, run-relative; still strictly worse than any successful sample.)
    if ok_rows:
        worst_cd = max(r["cd"]        for r in ok_rows)
        worst_cd_yaw = max(r["cd_yawmin"] for r in ok_rows)
        penalty_cd = 1.5 * worst_cd
        penalty_cd_yaw = 1.5 * worst_cd_yaw
    else:
        worst_cd = worst_cd_yaw = None
        penalty_cd = penalty_cd_yaw = None

    def agg(field, penalty):
        if not ok_rows:
            return {"conditional_mean": None, "penalized_mean": None}
        cond = float(np.mean([r[field] for r in ok_rows]))
        pen = []
        for r in per_instance:
            pen.append(r[field] if r["status"] == "OK" else penalty)
        return {"conditional_mean": cond, "penalized_mean": float(np.mean(pen))}

    summary = {
        "model":         args.model,
        "results_root":  str(args.results_root),
        "n_instances":   n,
        "n_ok":          n_ok,
        "n_points":      args.n_points,
        "normalization": "unit_sphere",
        "yaw_alignment": [0, 90, 180, 270],
        "penalty_cd":         penalty_cd,
        "penalty_cd_yawmin":  penalty_cd_yaw,
        "cd":         agg("cd",        penalty_cd),
        "cd_yawmin":  agg("cd_yawmin", penalty_cd_yaw),
        "per_instance": per_instance,
    }

    out_dir = model_dir / "_metrics"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "shape_chamfer.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Stdout
    def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "  --  "
    print()
    print(f"=== shape Chamfer — {args.model} ===")
    print(f"  Instances:                {n}")
    print(f"  Generated GLBs loaded:    {n_ok}")
    print(f"  CD             cond={fmt(summary['cd']['conditional_mean']):>9}  "
          f"pen={fmt(summary['cd']['penalized_mean']):>9}")
    print(f"  CD (yaw-min)   cond={fmt(summary['cd_yawmin']['conditional_mean']):>9}  "
          f"pen={fmt(summary['cd_yawmin']['penalized_mean']):>9}")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()

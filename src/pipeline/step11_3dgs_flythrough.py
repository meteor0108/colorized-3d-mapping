"""Step11: 3DGS 학습 → 궤적 flythrough 영상 (렌더 산출물, 3DGS=2DGS보다 sharp).

같은 GS dataset(step7c) 재사용. photometric(L1+SSIM)만으로 3DGS 학습 후
포즈 보간 flythrough mp4 생성.
실행: kctc_gs env
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R, Slerp

from src.common import Calibration, load_config
from src.pipeline.gs_backbone_3dgs import train_3dgs, load_3dgs, make_cam


def _interp(c2w_list, n):
    out = []
    for i in range(len(c2w_list) - 1):
        A, B = c2w_list[i], c2w_list[i + 1]
        sl = Slerp([0, 1], R.from_matrix(np.stack([A[:3, :3], B[:3, :3]])))
        for t in np.linspace(0, 1, n, endpoint=False):
            T = np.eye(4); T[:3, :3] = sl([t])[0].as_matrix()
            T[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]; out.append(T)
    out.append(c2w_list[-1]); return out


def run(ds_dir, out_mp4, cfg, iterations=None, interp=4, fps=30, extract_only=False):
    ds = Path(ds_dir)
    cams = json.load(open(ds / "cameras.json"))["frames"]
    frames = [{"stem": Path(c["img"]).stem, "img": c["img"], "w2c": np.array(c["w2c"]),
               "K": np.array(c["K"]), "width": int(c["width"]), "height": int(c["height"])}
              for c in cams]
    gs = cfg.mesh.gaussian
    if iterations:
        gs.train["iterations"] = iterations

    if extract_only:
        model = load_3dgs(frames, ds, gs)
    else:
        init = o3d.io.read_point_cloud(str(ds / "points3D.ply"))
        model = train_3dgs(frames, ds, init, gs, Calibration.from_config(cfg))
    from gaussian_renderer import render
    g, pipe, FoVx, FoVy, W, H = model
    bg = torch.zeros(3, device="cuda")

    poses = _interp([np.linalg.inv(f["w2c"]) for f in frames], interp)
    print(f"[step11] {len(frames)} cams → {len(poses)} 프레임", flush=True)
    out = Path(out_mp4); out.parent.mkdir(parents=True, exist_ok=True)
    fdir = out.parent / (out.stem + "_frames"); fdir.mkdir(exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    with torch.no_grad():
        for i, c2w in enumerate(poses):
            cam = make_cam(np.linalg.inv(c2w), FoVx, FoVy, W, H)
            rgb = render(cam, g, pipe, bg)["render"].clamp(0, 1)
            bgr = (rgb * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            vw.write(np.ascontiguousarray(bgr)); cv2.imwrite(str(fdir / f"{i:04d}.png"), bgr)
            if i % 30 == 0:
                print(f"  render {i}/{len(poses)}", flush=True)
    vw.release()
    print(f"[step11] 저장: {out}  ({len(poses)} frames @ {fps}fps)\n[step11] 프레임: {fdir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Step11: 3DGS flythrough")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--interp", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--extract-only", action="store_true", help="학습 생략, 체크포인트로 렌더만")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(args.dataset, args.output, cfg, iterations=args.iterations, interp=args.interp,
        fps=args.fps, extract_only=args.extract_only)


if __name__ == "__main__":
    main()

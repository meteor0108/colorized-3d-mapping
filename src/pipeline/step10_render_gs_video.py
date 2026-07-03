"""Step10: 학습된 2DGS로 궤적 따라 novel-view 플라이스루 영상 생성 (최종 산출물).

결론: 이 데이터(회전LiDAR+전방카메라+야외주행)는 mesh가 아니라 2DGS 렌더링이 깔끔.
→ 카메라 포즈를 SLERP/LERP 보간해 부드러운 flythrough 영상(mp4) + 프레임 출력.

입력: step7c dense GS dataset(+ point_cloud_2dgs.ply 체크포인트)
실행: kctc_gs env
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

from src.common import Calibration, load_config
from src.pipeline.gs_backbone_2dgs import load_2dgs


def _fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def _interp_poses(c2w_list, n_between):
    """연속 c2w 사이를 LERP(위치)+SLERP(회전) 보간."""
    poses = []
    for i in range(len(c2w_list) - 1):
        A, B = c2w_list[i], c2w_list[i + 1]
        slerp = Slerp([0, 1], R.from_matrix(np.stack([A[:3, :3], B[:3, :3]])))
        for t in np.linspace(0, 1, n_between, endpoint=False):
            T = np.eye(4)
            T[:3, :3] = slerp([t])[0].as_matrix()
            T[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]
            poses.append(T)
    poses.append(c2w_list[-1])
    return poses


def run(ds_dir, out_mp4, cfg, interp=4, fps=30, save_frames=True):
    calib = Calibration.from_config(cfg)
    ds = Path(ds_dir)
    cams = json.load(open(ds / "cameras.json"))["frames"]
    frames = [{"stem": Path(c["img"]).stem, "img": c["img"], "w2c": np.array(c["w2c"]),
               "K": np.array(c["K"]), "width": int(c["width"]), "height": int(c["height"])}
              for c in cams]

    model = load_2dgs(frames, ds, cfg.mesh.gaussian, calib)   # repo 경로 추가 + gaussians
    from gaussian_renderer import render
    from scene.cameras import Camera
    gaussians, pipe, _, _ = model
    bg = torch.zeros(3, device="cuda")

    K = frames[0]["K"]; W, H = frames[0]["width"], frames[0]["height"]
    FoVx, FoVy = _fov(K[0, 0], W), _fov(K[1, 1], H)
    c2w_list = [np.linalg.inv(f["w2c"]) for f in frames]
    poses = _interp_poses(c2w_list, interp)
    print(f"[step10] {len(frames)} cams → {len(poses)} 보간 프레임 (interp={interp})", flush=True)

    out = Path(out_mp4); out.parent.mkdir(parents=True, exist_ok=True)
    fdir = out.parent / (out.stem + "_frames")
    if save_frames:
        fdir.mkdir(exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    dummy = torch.zeros(3, H, W)
    with torch.no_grad():
        for i, c2w in enumerate(poses):
            w2c = np.linalg.inv(c2w)
            cam = Camera(colmap_id=i, R=w2c[:3, :3].T, T=w2c[:3, 3], FoVx=FoVx, FoVy=FoVy,
                         image=dummy, gt_alpha_mask=None, image_name=str(i), uid=i)
            rgb = render(cam, gaussians, pipe, bg)["render"].clamp(0, 1)
            bgr = (rgb * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            vw.write(np.ascontiguousarray(bgr))
            if save_frames:
                cv2.imwrite(str(fdir / f"{i:04d}.png"), bgr)
            if i % 30 == 0:
                print(f"  render {i}/{len(poses)}", flush=True)
    vw.release()
    print(f"[step10] 저장: {out}  ({len(poses)} frames @ {fps}fps)", flush=True)
    if save_frames:
        print(f"[step10] 프레임: {fdir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Step10: 2DGS flythrough 영상")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", "-o", required=True, help="출력 mp4")
    ap.add_argument("--interp", type=int, default=4, help="프레임 사이 보간 수")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(args.dataset, args.output, cfg, interp=args.interp, fps=args.fps)


if __name__ == "__main__":
    main()

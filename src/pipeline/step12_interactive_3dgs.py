"""Step12: 학습된 3DGS를 실시간 렌더하며 키보드로 자유 탐색하는 인터랙티브 3D 뷰어.

flythrough(고정경로 영상)와 달리, 3DGS 장면을 자유 시점으로 둘러본다.
4090 실시간 렌더 + cv2 창(DISPLAY 필요). 외부 뷰어 불필요.

조작:
  W/S 전/후   A/D 좌/우   Q/E 상/하
  J/L 좌우회전(yaw)   I/K 상하회전(pitch)
  +/- 이동속도   R 시작위치   ESC 종료

실행: DISPLAY=:0 kctc_gs env
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.common import load_config
from src.pipeline.gs_backbone_3dgs import load_3dgs, make_cam


def _rot(axis, deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    if axis == "y":   # yaw (world up = z) → 회전축 world z
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return None


def run(ds_dir, cfg, scale=0.5):
    ds = Path(ds_dir)
    cams = json.load(open(ds / "cameras.json"))["frames"]
    frames = [{"stem": Path(c["img"]).stem, "img": c["img"], "w2c": np.array(c["w2c"]),
               "K": np.array(c["K"]), "width": int(c["width"]), "height": int(c["height"])}
              for c in cams]
    g, pipe, FoVx, FoVy, W0, H0 = load_3dgs(frames, ds, cfg.mesh.gaussian)
    from gaussian_renderer import render
    W, H = int(W0 * scale), int(H0 * scale)
    bg = torch.zeros(3, device="cuda")

    c2w0 = np.linalg.inv(frames[len(frames) // 3]["w2c"])   # 시작 = 1/3 지점 카메라
    c2w = c2w0.copy()
    speed = 0.3

    print("[step12] 뷰어 시작. 창에서 W/A/S/D Q/E 이동, J/L I/K 회전, +/- 속도, R 리셋, ESC 종료", flush=True)
    cv2.namedWindow("3DGS interactive", cv2.WINDOW_NORMAL); cv2.resizeWindow("3DGS interactive", W, H)
    while True:
        with torch.no_grad():
            cam = make_cam(np.linalg.inv(c2w), FoVx, FoVy, W, H)
            rgb = render(cam, g, pipe, bg)["render"].clamp(0, 1)
        bgr = (rgb * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        cv2.imshow("3DGS interactive", np.ascontiguousarray(bgr))
        k = cv2.waitKey(15) & 0xFF
        if k == 27:
            break
        R = c2w[:3, :3]
        right, up, fwd = R[:, 0], -R[:, 1], R[:, 2]   # OpenCV: +x우 +y하 +z전
        if k in (ord('w'), ord('W')):   c2w[:3, 3] += fwd * speed
        elif k in (ord('s'), ord('S')): c2w[:3, 3] -= fwd * speed
        elif k in (ord('d'), ord('D')): c2w[:3, 3] += right * speed
        elif k in (ord('a'), ord('A')): c2w[:3, 3] -= right * speed
        elif k in (ord('e'), ord('E')): c2w[:3, 3] += np.array([0, 0, 1]) * speed
        elif k in (ord('q'), ord('Q')): c2w[:3, 3] -= np.array([0, 0, 1]) * speed
        elif k in (ord('l'), ord('L')): c2w[:3, :3] = _rot("y", -3) @ c2w[:3, :3]
        elif k in (ord('j'), ord('J')): c2w[:3, :3] = _rot("y", 3) @ c2w[:3, :3]
        elif k in (ord('i'), ord('I')):
            Rp = _axis_rot(c2w[:3, 0], -3); c2w[:3, :3] = Rp @ c2w[:3, :3]
        elif k in (ord('k'), ord('K')):
            Rp = _axis_rot(c2w[:3, 0], 3); c2w[:3, :3] = Rp @ c2w[:3, :3]
        elif k in (ord('+'), ord('=')): speed = min(speed * 1.3, 5.0)
        elif k in (ord('-'), ord('_')): speed = max(speed / 1.3, 0.02)
        elif k in (ord('r'), ord('R')): c2w = c2w0.copy()
    cv2.destroyAllWindows()
    print("[step12] 종료", flush=True)


def _axis_rot(axis, deg):
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a); x, y, z = axis
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s,   c + y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s,   z*y*(1-c)+x*s, c + z*z*(1-c)]])


def main():
    ap = argparse.ArgumentParser(description="Step12: 3DGS 인터랙티브 뷰어")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--dataset", required=True, help="3DGS 체크포인트가 있는 dataset")
    ap.add_argument("--scale", type=float, default=0.5, help="렌더 해상도 배율(낮을수록 빠름)")
    args = ap.parse_args()
    run(args.dataset, load_config(args.config), scale=args.scale)


if __name__ == "__main__":
    main()

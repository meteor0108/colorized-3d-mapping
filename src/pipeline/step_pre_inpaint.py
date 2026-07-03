"""사전패스: SAM3로 이동체(차/사람/자전거...) 마스킹 → LaMa inpaint → blackfly_nodyn/ 저장.

융합(step14_seq)에서 SAM3+LaMa+mono 3모델 동시 로드 시 CPU RAM(15GB) 초과 → 분리.
이 스크립트는 SAM3+LaMa만 로드(낮은 RAM), inpaint된 이미지를 디스크에 저장.
이후 step14_seq --img-dir blackfly_nodyn 로 검증된 mono-only 융합(낮은 베이스라인) 실행.

raw(왜곡) 이미지에 마스킹 → inpaint → 저장. 융합이 나중에 undistort(중복 없음).
마스크는 cutoff/비네팅 안쪽으로 제한(에고 보닛 등 폐기영역 inpaint 낭비 방지).
재개 가능: 이미 저장된 프레임은 건너뜀.

실행: kctc_gs env (transformers Sam3*, simple_lama_inpainting)
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.common import Calibration, FileManager, load_config
from src.pipeline.step14_seq_mono_fusion import DYN_PROMPTS, SAM3_THR, _sam3_dynmask


def run(data_dir, cfg, out_sub="blackfly_nodyn", dilate=15):
    import torch
    from transformers import Sam3Processor, Sam3Model
    from simple_lama_inpainting import SimpleLama
    calib = Calibration.from_config(cfg)
    mask_p = calib.mask_params(); cutoff = calib.cutoff_y()
    data = Path(data_dir)
    src = data / "blackfly"; dst = data / out_sub; dst.mkdir(parents=True, exist_ok=True)
    imgs, _ = FileManager.get_files_and_times(str(src), ext=".png")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(dev).eval()
    lama = SimpleLama()
    print(f"[pre] SAM3({DYN_PROMPTS}, thr{SAM3_THR}) + LaMa @ {dev} | {len(imgs)} frames → {dst}", flush=True)

    n_done = n_inp = 0
    for i, name in enumerate(imgs):
        op = dst / name
        if op.exists() or op.is_symlink():
            n_done += 1; continue
        bgr = cv2.imread(str(src / name))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        dmb = _sam3_dynmask(proc, model, bgr, H, W, dev)
        yy, xx = np.mgrid[0:H, 0:W]                       # 폐기영역(보닛/비네팅 밖) inpaint 생략
        if cutoff is not None:
            dmb &= (yy < cutoff)
        if mask_p is not None:
            cx, cy, r = mask_p; dmb &= ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        cov = float(dmb.mean() * 100)
        if dmb.any():
            dm = cv2.dilate(dmb.astype(np.uint8) * 255, np.ones((dilate, dilate), np.uint8))
            cln = lama(Image.fromarray(bgr[:, :, ::-1]), Image.fromarray(dm))
            out = cv2.cvtColor(np.array(cln.convert("RGB")), cv2.COLOR_RGB2BGR)
            if out.shape[:2] != (H, W):
                out = cv2.resize(out, (W, H))
            cv2.imwrite(str(op), out)
            n_inp += 1
            del cln, out, dm
        else:                                             # 동적객체 없음: 원본 심볼릭(디스크 절약)
            os.symlink(os.path.abspath(src / name), op)
        del bgr, dmb, yy, xx
        n_done += 1
        if i % 25 == 0:
            gc.collect()
            print(f"[pre] {i+1}/{len(imgs)}  inpaint={n_inp}  cov={cov:.1f}%", flush=True)
    print(f"[pre] 완료: {n_done} 저장 (inpaint {n_inp}) → {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="사전 SAM3+LaMa inpaint → blackfly_nodyn")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--data", required=True, help="native 추출 폴더 (blackfly 포함)")
    ap.add_argument("--out-sub", default="blackfly_nodyn")
    args = ap.parse_args()
    run(args.data, load_config(args.config), out_sub=args.out_sub)


if __name__ == "__main__":
    main()

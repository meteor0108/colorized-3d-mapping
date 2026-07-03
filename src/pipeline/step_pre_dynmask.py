"""사전패스(마스크-only 융합용): SAM3로 이동체 마스크 → dilate → dynmask/<name>.png 저장.

다중시점 기하보간 방식: 융합에서 이 마스크 영역의 depth를 0으로(제외) → 차량 뒤 배경을
다른 프레임의 실제 관측으로 TSDF가 채움. LaMa 환각을 메쉬에 굽지 않음(맵 퀄리티↑).

raw(왜곡) 이미지에 마스킹 → 저장. 융합이 이미지와 함께 마스크도 undistort.
재개 가능. 동적객체 없으면 all-zero 마스크(작음).
실행: kctc_gs env (transformers Sam3*)
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import cv2
import numpy as np

from src.common import Calibration, FileManager, load_config
from src.pipeline.step14_seq_mono_fusion import DYN_PROMPTS, SAM3_THR, _sam3_dynmask


def run(data_dir, cfg, out_sub="dynmask", dilate=15):
    import torch
    from transformers import Sam3Processor, Sam3Model
    data = Path(data_dir)
    src = data / "blackfly"; dst = data / out_sub; dst.mkdir(parents=True, exist_ok=True)
    imgs, _ = FileManager.get_files_and_times(str(src), ext=".png")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(dev).eval()
    print(f"[dynmask] SAM3({DYN_PROMPTS}, thr{SAM3_THR}) @ {dev} | {len(imgs)} frames → {dst}", flush=True)

    n_done = n_dyn = 0
    for i, name in enumerate(imgs):
        op = dst / name
        if op.exists():
            n_done += 1; continue
        bgr = cv2.imread(str(src / name))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        dmb = _sam3_dynmask(proc, model, bgr, H, W, dev)   # 전체 차량 마스크(cutoff/비네팅 제한 안 함)
        cov = float(dmb.mean() * 100)
        if dmb.any():
            dm = cv2.dilate(dmb.astype(np.uint8) * 255, np.ones((dilate, dilate), np.uint8))
            n_dyn += 1
        else:
            dm = np.zeros((H, W), np.uint8)
        cv2.imwrite(str(op), dm)
        del bgr, dmb
        n_done += 1
        if i % 25 == 0:
            gc.collect()
            print(f"[dynmask] {i+1}/{len(imgs)}  dyn={n_dyn}  cov={cov:.1f}%", flush=True)
    print(f"[dynmask] 완료: {n_done} 저장 (dyn {n_dyn}) → {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="사전 SAM3 동적 마스크 → dynmask/")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--data", required=True, help="native 추출 폴더 (blackfly 포함)")
    ap.add_argument("--out-sub", default="dynmask")
    args = ap.parse_args()
    run(args.data, load_config(args.config), out_sub=args.out_sub)


if __name__ == "__main__":
    main()

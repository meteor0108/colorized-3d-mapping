"""Step14-seq: 한 시퀀스 전체를 순수 mono 융합 → chunk별 dense 컬러 메쉬.

전체 route는 단일 TSDF로 RAM 초과 → 윈도우(chunk) 단위로 타일링.
각 chunk = 검증된 설정(60프레임, voxel 0.05, trunc 40). odom 포즈 + 프레임별 LiDAR scan으로 스케일.
저장 depth/*.npy(반전 버그)는 안 씀 — LiDAR scan을 직접 카메라 투영.

실행: kctc_gs env (transformers, open3d)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R

from src.common import Calibration, FileManager, PointCloudProcessor, TimestampParser, load_config

POSE_LIDAR = "ouster2"


def _nav(path):
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    tcol = [c for c in df.columns if "time" in c.lower() or "stamp" in c.lower()][0]
    return df, df[tcol].values


def _pose(df, navts, t, origin):
    i = int(np.abs(navts - t).argmin())
    p = np.array([df.position_x.iloc[i], df.position_y.iloc[i], df.position_z.iloc[i]])
    q = np.array([df.orientation_x.iloc[i], df.orientation_y.iloc[i],
                  df.orientation_z.iloc[i], df.orientation_w.iloc[i]])
    T = np.eye(4); T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = p - origin
    return T


DYN_PROMPTS = ["car", "truck", "bus", "person", "bicycle", "motorcycle"]  # SAM3 텍스트 개념(이동체)
SAM3_THR = 0.5


def _sam3_dynmask(proc, model, bgr, H, W, device):
    """SAM3 텍스트 프롬프트로 이동체 인스턴스 합집합 마스크(bool HxW). 프레임마다 GPU 해제."""
    import torch
    img = Image.fromarray(bgr[:, :, ::-1])
    dm = np.zeros((H, W), bool)
    for p in DYN_PROMPTS:
        inp = proc(images=img, text=p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_instance_segmentation(out, threshold=SAM3_THR, target_sizes=[(H, W)])[0]
        if res["masks"] is not None:
            for mk, sc in zip(res["masks"], res["scores"]):
                if float(sc) > SAM3_THR:
                    dm |= (mk.cpu().numpy() > 0.5)
        del inp, out, res
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    return dm


def run(data_dir, out_dir, cfg, stride=2, chunk=60, voxel=0.05, depth_trunc=40.0, near=1.0,
        model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf", remove_dynamic=False,
        img_dir="blackfly", resume=False, max_new=0, dynmask_dir=None, holefill_mesh=None):
    from transformers import pipeline
    calib = Calibration.from_config(cfg)
    K = np.asarray(calib.intrinsic); dist = np.asarray(calib.distortion)
    g2l = calib.gps_to_lidar[POSE_LIDAR]; l2c = np.asarray(calib.lidar_to_camera[POSE_LIDAR])
    mask_p = calib.mask_params(); cutoff = calib.cutoff_y()
    data = Path(data_dir)
    df, navts = _nav(data / "navigation.csv")
    # 동적제거를 사전패스(blackfly_nodyn)로 분리했으면 inpaint된 폴더에서, 타임스탬프는 파일명(epoch) 기준
    imgs, imts = FileManager.get_files_and_times(str(data / img_dir), ext=".png")
    lf, lt = FileManager.get_files_and_times(str(data / POSE_LIDAR / "points"))
    sel = list(range(0, len(imgs), stride))
    i0 = int(np.abs(navts - imts[sel[0]]).argmin())
    origin = np.array([df.position_x.iloc[i0], df.position_y.iloc[i0], df.position_z.iloc[i0]])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    import torch
    mono_dev = 1 if torch.cuda.device_count() > 1 else 0   # mono=GPU1, Open3D TSDF=GPU0 (충돌 회피)
    pipe = pipeline("depth-estimation", model=model, device=mono_dev)
    dev = o3c.Device("CUDA:0")
    sam_proc = sam_model = lama = None
    sam_dev = "cuda:0" if torch.cuda.is_available() else "cpu"   # TSDF=CPU라 GPU0 비어있음
    if remove_dynamic:
        from transformers import Sam3Processor, Sam3Model
        from simple_lama_inpainting import SimpleLama
        sam_proc = Sam3Processor.from_pretrained("facebook/sam3")
        sam_model = Sam3Model.from_pretrained("facebook/sam3").to(sam_dev).eval()
        lama = SimpleLama()
        print(f"[14seq] 동적객체 제거 ON: SAM3({DYN_PROMPTS}, thr{SAM3_THR}) @ {sam_dev} + LaMa inpaint", flush=True)
    print(f"[14seq] mono=GPU{mono_dev}, Open3D TSDF=GPU0", flush=True)
    print(f"[14seq] frames={len(sel)} (stride {stride}) → {int(np.ceil(len(sel)/chunk))} chunks (GPU TSDF)", flush=True)
    hf_geo = hf_ren = hf_mat = None
    if holefill_mesh is not None:                        # 구멍보완: 마스크-only 메쉬를 렌더해 never-observed만 LaMa로 채움
        hf_geo = o3d.io.read_triangle_mesh(holefill_mesh); hf_geo.compute_vertex_normals()
        hf_mat = o3d.visualization.rendering.MaterialRecord(); hf_mat.shader = "defaultUnlit"
        print(f"[14seq] 구멍보완 ON: {holefill_mesh} ({len(hf_geo.vertices):,}v) 렌더 기반", flush=True)

    W = H = None
    n_new = 0
    for ci, c0 in enumerate(range(0, len(sel), chunk)):
        p = out / f"chunk_{ci:03d}.ply"
        if resume and p.exists():                       # 이미 만든 chunk 건너뜀(누수 회피 재시작)
            continue
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=voxel * 4,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        n = 0
        for k in sel[c0:c0 + chunk]:
            t = imts[k]
            bgr = cv2.imread(str(data / img_dir / imgs[k]))
            if bgr is None:
                continue
            bgr = cv2.undistort(bgr, K, dist)
            H, W = bgr.shape[:2]
            if remove_dynamic:                              # 움직이는 차량 확실히 제거: SAM3 전 이동체 마스킹 후 inpaint
                dmb = _sam3_dynmask(sam_proc, sam_model, bgr, H, W, sam_dev)
                yy0, xx0 = np.mgrid[0:H, 0:W]               # inpaint를 융합영역으로 제한(보닛/비네팅 밖 낭비 제거)
                if cutoff is not None:
                    dmb &= (yy0 < cutoff)
                if mask_p is not None:
                    cx0, cy0, r0 = mask_p; dmb &= ((xx0 - cx0) ** 2 + (yy0 - cy0) ** 2) < r0 ** 2
                if dmb.any():
                    dm = cv2.dilate(dmb.astype(np.uint8) * 255, np.ones((15, 15), np.uint8))  # 모션블러 경계까지
                    cln = lama(Image.fromarray(bgr[:, :, ::-1]), Image.fromarray(dm))
                    bgr = cv2.cvtColor(np.array(cln.convert("RGB")), cv2.COLOR_RGB2BGR)
                    if bgr.shape[:2] != (H, W):
                        bgr = cv2.resize(bgr, (W, H))
            import torch, gc
            with torch.no_grad():
                m = pipe(Image.fromarray(bgr[:, :, ::-1]))["predicted_depth"].squeeze().float().cpu().numpy()
            torch.cuda.empty_cache(); gc.collect()
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H))
            # 프레임별 LiDAR scan → scale
            j = int(np.abs(lt - t).argmin())
            s = 1.0
            if abs(lt[j] - t) < 0.2:
                pc = PointCloudProcessor.range_filter(
                    o3d.io.read_point_cloud(str(data / POSE_LIDAR / "points" / lf[j])), distance_threshold=depth_trunc)
                P = np.asarray(pc.points)
                cam = (l2c @ np.vstack((P.T, np.ones(len(P)))))[:3].T
                z = cam[:, 2]; fr = z > near
                uv = (K @ (cam[fr] / z[fr, None]).T).T
                u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int); zz = z[fr]
                ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if ins.sum() > 300:
                    mv = m[v[ins], u[ins]]; ok = mv > 0.1
                    r = zz[ins][ok] / mv[ok]
                    s = float(np.median(r[(r > np.percentile(r, 10)) & (r < np.percentile(r, 90))]))
            depth = (m * s).astype(np.float32)
            keep = (depth > near) & (depth < depth_trunc)
            yy, xx = np.mgrid[0:H, 0:W]
            if cutoff is not None:
                keep &= yy < cutoff
            if mask_p is not None:
                cx, cy, rr = mask_p; keep &= ((xx - cx) ** 2 + (yy - cy) ** 2) < rr ** 2
            depth[~keep] = 0.0
            w2c = np.linalg.inv(_pose(df, navts, t, origin) @ g2l @ np.linalg.inv(l2c))
            if dynmask_dir is not None:
                dmp = data / dynmask_dir / imgs[k]
                if dmp.exists():
                    dmk = cv2.imread(str(dmp), cv2.IMREAD_GRAYSCALE)
                    dmk = cv2.undistort(dmk, K, dist) > 127   # 이미지와 동일 보정
                    if holefill_mesh is not None:            # 구멍보완: 마스크-only가 덮은 곳만 제외, 구멍은 LaMa depth 유지
                        if hf_ren is None:
                            hf_ren = o3d.visualization.rendering.OffscreenRenderer(W, H)
                            hf_ren.scene.add_geometry("hf", hf_geo, hf_mat)
                        hf_ren.setup_camera(K.astype(np.float64), w2c.astype(np.float64), W, H)
                        dr = np.asarray(hf_ren.render_to_depth_image(z_in_view_space=True))
                        covered = dmk & np.isfinite(dr) & (dr > near) & (dr < depth_trunc)
                        depth[covered] = 0.0                 # 마스크-only가 이미 보유 → 중복 제외(LaMa 환각 방지)
                    else:                                    # 순수 마스크-only: 동적영역 전부 제외
                        depth[dmk] = 0.0
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(bgr[:, :, ::-1])), o3d.geometry.Image(depth),
                depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
            intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            vol.integrate(rgbd, intr, w2c)
            n += 1
            del bgr, m, depth, rgbd, keep, yy, xx
        import gc
        mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
        del vol; gc.collect()
        o3d.io.write_triangle_mesh(str(p), mesh, write_vertex_colors=True)   # raw 먼저(안전)
        if 0 < len(mesh.triangles) < 6_000_000:                              # 작을 때만 후처리(RAM)
            tc, nt, _ = mesh.cluster_connected_triangles(); tc = np.asarray(tc); nt = np.asarray(nt)
            mesh.remove_triangles_by_mask(nt[tc] < max(200, int(nt.max() * 0.02)))
            mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(p), mesh, write_vertex_colors=True)
        print(f"[14seq] chunk {ci} ({n}f): {len(mesh.vertices):,}v/{len(mesh.triangles):,}f → {p.name}", flush=True)
        del mesh; gc.collect()
        n_new += 1
        if max_new and n_new >= max_new:                # 누수 회피: N개 만들고 프로세스 종료(재시작이 RAM 해제)
            print(f"[14seq] max_new={max_new} 도달 → 종료(재시작 대기)", flush=True)
            return
    print(f"[14seq] 완료 → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Step14-seq: 시퀀스 전체 pure-mono 융합 (chunked)")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--data", required=True, help="native 추출 폴더")
    ap.add_argument("--out", required=True, help="chunk 메쉬 출력 폴더")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--depth-trunc", type=float, default=40.0)
    ap.add_argument("--remove-dynamic", action="store_true", help="SAM3+LaMa로 동적객체(차/사람/자전거) 제거 (인-루프)")
    ap.add_argument("--img-dir", default="blackfly",
                    help="이미지 하위폴더 (사전 inpaint면 blackfly_nodyn). 이 경우 --remove-dynamic 불필요")
    ap.add_argument("--resume", action="store_true", help="이미 만든 chunk_*.ply 건너뜀")
    ap.add_argument("--max-new", type=int, default=0, help="새 chunk N개 만들고 종료(누수 회피 재시작용, 0=전체)")
    ap.add_argument("--dynmask-dir", default=None, help="마스크-only 다중시점: 이 폴더의 동적마스크로 depth=0 (img-dir=blackfly와 함께)")
    ap.add_argument("--holefill-mesh", default=None, help="구멍보완: 마스크-only 메쉬 렌더로 never-observed만 LaMa로 채움 (img-dir=blackfly_nodyn + dynmask-dir)")
    args = ap.parse_args()
    run(args.data, args.out, load_config(args.config), stride=args.stride, chunk=args.chunk,
        voxel=args.voxel, depth_trunc=args.depth_trunc, remove_dynamic=args.remove_dynamic,
        img_dir=args.img_dir, resume=args.resume, max_new=args.max_new, dynmask_dir=args.dynmask_dir,
        holefill_mesh=args.holefill_mesh)


if __name__ == "__main__":
    main()

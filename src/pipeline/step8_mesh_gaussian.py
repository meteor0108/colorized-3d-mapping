"""Step8: 2D Gaussian Splatting 기반 mesh 생성 (submap 단위, 기하 정확도 우선).

파이프라인:
    1. step7b가 만든 GS dataset 로드 (images / cameras.json / depth / points3D.ply)
    2. 2DGS 학습 — photometric(L1+SSIM) + LiDAR depth L1 + normal consistency
       (backbone: 2DGS 공식 repo 또는 gsplat. 설치 후 _train_backbone 연결)
    3. 학습 뷰 depth 렌더 → Open3D TSDF fusion → marching cubes mesh
    4. step6 post-process 재사용

현재 상태:
    - 데이터 로드 / TSDF 추출 / 후처리: 동작 구현 완료
    - GS 학습(_train_backbone) + depth 렌더(_render_depths): backbone 설치 후 연결 (TODO)
      backbone 미설치 상태에서도 import는 안전. dataset/TSDF 단계는 독립 실행 가능.

설정: configs/mesh.yaml > mesh.gaussian.{train, extract}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from src.common import Calibration, load_config


class GaussianMeshReconstructor:
    def __init__(self, cfg, dataset_dir: str):
        self.cfg = cfg
        self.gs = cfg.mesh.gaussian
        self.dataset_dir = Path(dataset_dir)
        self.train_cfg = self.gs.train
        self.extract_cfg = self.gs.extract
        self.calib = Calibration.from_config(cfg)

    # ------------------------------------------------------------------ load
    def load_dataset(self):
        with open(self.dataset_dir / "cameras.json") as f:
            cams = json.load(f)["frames"]
        if not cams:
            raise SystemExit(f"빈 dataset: {self.dataset_dir}")

        frames = []
        for c in cams:
            frames.append({
                "stem": Path(c["img"]).stem,
                "img": c["img"],                       # dataset_dir 기준 상대경로
                "w2c": np.array(c["w2c"], np.float64),
                "K": np.array(c["K"], np.float64),
                "width": int(c["width"]),
                "height": int(c["height"]),
            })
        init_ply = self.dataset_dir / "points3D.ply"
        init_pcd = o3d.io.read_point_cloud(str(init_ply)) if init_ply.exists() else None
        print(f"[step8] dataset 로드: frames={len(frames)} "
              f"init_pts={len(init_pcd.points) if init_pcd else 0:,}")
        return frames, init_pcd

    # -------------------------------------------------------------- training
    def train(self, frames, init_pcd):
        """2DGS 학습. backbone 어댑터를 연결한다."""
        backbone = str(self.gs.backbone)
        return self._train_backbone(backbone, frames, init_pcd)

    def _train_backbone(self, backbone, frames, init_pcd):
        if backbone not in ("2dgs", "gsplat-2dgs"):
            raise SystemExit(f"미지원 backbone: {backbone}")
        if init_pcd is None:
            raise SystemExit("init points3D.ply 없음 — step7b를 먼저 실행")
        from src.pipeline.gs_backbone_2dgs import train_2dgs
        return train_2dgs(frames, self.dataset_dir, init_pcd, self.gs, self.calib)

    def load_model(self, frames):
        """저장된 체크포인트에서 모델 로드 (재학습 없이 mesh 재추출)."""
        from src.pipeline.gs_backbone_2dgs import load_2dgs
        return load_2dgs(frames, self.dataset_dir, self.gs, self.calib)

    def _render_depths(self, model, frames):
        from src.pipeline.gs_backbone_2dgs import render_depths_2dgs
        return render_depths_2dgs(model, frames, alpha_thresh=float(self.extract_cfg.alpha_thresh))

    # ------------------------------------------------------------- extraction
    def extract_mesh_tsdf(self, views):
        """dense depth 뷰 → Open3D TSDF fusion → triangle mesh.

        Args:
            views: list of dict {rgb(HxWx3 uint8 RGB), depth(HxW m), w2c(4x4), K, width, height}
                   GS 렌더 결과 또는 임의의 posed RGB-D.
        """
        voxel = float(self.extract_cfg.tsdf_voxel)
        sdf_trunc = voxel * float(self.extract_cfg.sdf_trunc_mult)
        depth_trunc = float(self.extract_cfg.depth_trunc)

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        for v in views:
            h, w = v["height"], v["width"]
            color = o3d.geometry.Image(np.ascontiguousarray(v["rgb"]))
            depth = o3d.geometry.Image(np.ascontiguousarray(v["depth"].astype(np.float32)))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1.0, depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            K = v["K"]
            intr = o3d.camera.PinholeCameraIntrinsic(
                w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            volume.integrate(rgbd, intr, v["w2c"])  # extrinsic = world→camera

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        print(f"[step8] TSDF mesh: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")
        return mesh

    # ------------------------------------------------------------ post / save
    def post_process(self, mesh):
        """Open3D 벡터화 후처리. step6의 Python per-face 루프는 수백만 face에서
        수 분이 걸려 사용하지 않는다 (모두 C++ 연산으로 대체)."""
        if not self.extract_cfg.run_post_process:
            return mesh
        pp = self.cfg.mesh.post_process
        import time
        try:
            # 1) 작은 연결요소 제거 (cluster_connected_triangles = C++ 벡터화)
            if pp.remove_small_components.enabled:
                t = time.time()
                clusters, n_tri, _ = mesh.cluster_connected_triangles()
                clusters = np.asarray(clusters)
                n_tri = np.asarray(n_tri)
                small = n_tri[clusters] < int(pp.remove_small_components.min_triangles)
                mesh.remove_triangles_by_mask(small)
                mesh.remove_unreferenced_vertices()
                print(f"[step8] small-component 제거 {time.time()-t:.1f}s "
                      f"→ {len(mesh.triangles):,} tris")
            # 2) 통계적 이상치 제거 (vertex pcd, C++)
            if pp.remove_outliers.enabled:
                t = time.time()
                vpcd = o3d.geometry.PointCloud(mesh.vertices)
                _, ind = vpcd.remove_statistical_outlier(
                    int(pp.remove_outliers.nb_neighbors), float(pp.remove_outliers.std_ratio))
                mesh = mesh.select_by_index(ind)
                print(f"[step8] outlier 제거 {time.time()-t:.1f}s → {len(mesh.vertices):,} verts")
            # 3) 단순화 (quadric decimation, C++)
            if pp.simplify.enabled and len(mesh.triangles) > int(pp.simplify.target_triangles):
                t = time.time()
                mesh = mesh.simplify_quadric_decimation(int(pp.simplify.target_triangles))
                print(f"[step8] simplify {time.time()-t:.1f}s → {len(mesh.triangles):,} tris")
            # 4) 스무딩 (Taubin, C++)
            if pp.smooth.enabled:
                mesh = mesh.filter_smooth_taubin(int(pp.smooth.number_of_iterations))
            mesh.compute_vertex_normals()
        except Exception as e:  # 후처리 실패해도 raw mesh는 보존
            print(f"[step8] post-process 부분 실패(raw 유지): {e}")
        return mesh

    def save(self, mesh, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
        print(f"[step8] 저장: {output_path}")

    # ------------------------------------------------------------------- run
    def run(self, output_path, extract_only=False):
        frames, init_pcd = self.load_dataset()
        if extract_only:
            model = self.load_model(frames)           # 학습 생략, 체크포인트 로드
        else:
            model = self.train(frames, init_pcd)
        views = self._render_depths(model, frames)
        mesh = self.extract_mesh_tsdf(views)
        mesh = self.post_process(mesh)
        self.save(mesh, output_path)
        return mesh


def main():
    parser = argparse.ArgumentParser(description="Step8: 2DGS 기반 mesh 생성")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--dataset", required=True, help="step7b 출력 dataset 폴더")
    parser.add_argument("--output", "-o", default=None, help="출력 PLY")
    parser.add_argument("--iterations", type=int, default=None, help="학습 iter override (스모크 테스트용)")
    parser.add_argument("--extract-only", action="store_true",
                        help="학습 생략, 저장된 체크포인트(point_cloud_2dgs.ply)에서 mesh만 재추출")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.iterations is not None:
        cfg.mesh.gaussian.train["iterations"] = args.iterations
    output = args.output or str(Path(cfg.mesh.gaussian.dataset.output_dir).parent
                                / "output_mesh" / f"{Path(args.dataset.rstrip('/')).name}_gs.ply")
    GaussianMeshReconstructor(cfg, args.dataset).run(output, extract_only=args.extract_only)


if __name__ == "__main__":
    main()

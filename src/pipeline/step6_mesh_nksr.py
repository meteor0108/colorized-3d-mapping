import torch
import numpy as np
import open3d as o3d
import nksr
from tqdm import tqdm
import argparse
import os


class PointCloudMeshReconstructor:
    """
    NKSR 기반 Point Cloud to Mesh 변환기
    GPU 가속을 활용하여 대용량 컬러 포인트 클라우드를 고품질 메쉬로 변환
    """
    
    def __init__(self, device='cuda:0', chunk_size=5000000):
        """
        Args:
            device: CUDA 디바이스 ('cuda:0', 'cuda:1', etc.)
            chunk_size: 메모리 효율을 위한 청크 크기
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
        self.reconstructor = nksr.Reconstructor(self.device)
        
        print(f"🚀 Initialized on device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    
    def load_point_cloud(self, pcd_path):
        """PCD 파일 로드"""
        print(f"\n📂 Loading point cloud from: {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)

        print("   🧹 Applying Voxel Downsampling (leaf_size=0.02m)...")
        pcd = pcd.voxel_down_sample(voxel_size=0.02)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        print(f"   Points: {len(points):,}")
        print(f"   Has colors: {colors is not None}")
        print(f"   Bounds: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}] "
              f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}] "
              f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
        
        return pcd, points, colors
    
    
    def compute_normals_gpu(self, pcd, points, k=50, radius=0.5):
        """
        GPU 가속 Normal 계산
        
        Args:
            pcd: Open3D PointCloud 객체
            points: numpy array of points
            k: KNN 이웃 개수
            radius: 검색 반경 (미터)
        """
        print(f"\n🔍 Computing normals (k={k}, radius={radius}m)...")
        
        # Open3D의 GPU 가속 Normal 추정
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k)
        )
        
        # Normal 방향 일관성 확보 (중요!)
        pcd.orient_normals_consistent_tangent_plane(k=k)
        
        normals = np.asarray(pcd.normals)
        
        # NaN 체크
        valid_mask = ~np.isnan(normals).any(axis=1)
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            print(f"   ⚠️  Found {invalid_count} invalid normals, filtering...")
            points = points[valid_mask]
            normals = normals[valid_mask]
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)[valid_mask]
            else:
                colors = None
        else:
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        print(f"   ✓ Valid normals: {len(normals):,}")
        
        return points, normals, colors
    
    
    def downsample_if_needed(self, points, normals, colors, max_points=10000000):
        """메모리 제약이 있을 경우 다운샘플링"""
        if len(points) > max_points:
            print(f"\n⚡ Downsampling from {len(points):,} to ~{max_points:,} points...")
            
            # 균일 샘플링
            step = len(points) // max_points
            indices = np.arange(0, len(points), step)
            
            points = points[indices]
            normals = normals[indices]
            if colors is not None:
                colors = colors[indices]
            
            print(f"   Final points: {len(points):,}")
        
        return points, normals, colors
    
    
    def preprocess_for_nksr(self, points, normals, colors):
        """NKSR을 위한 데이터 전처리"""
        print("\n🔧 Preprocessing for NKSR...")
        
        # 센터링 및 정규화
        centroid = points.mean(axis=0)
        points_centered = points - centroid
        scale = np.abs(points_centered).max()
        points_normalized = points_centered / scale
        
        print(f"   Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
        print(f"   Scale: {scale:.2f}")
        
        # Torch tensor로 변환 (GPU)
        points_torch = torch.from_numpy(points_normalized).float().to(self.device)
        normals_torch = torch.from_numpy(normals).float().to(self.device)
        
        if colors is not None:
            colors_torch = torch.from_numpy(colors).float().to(self.device)
        else:
            colors_torch = None
        
        return points_torch, normals_torch, colors_torch, centroid, scale
    
    
    def reconstruct_mesh(self, points_torch, normals_torch, colors_torch, 
                        centroid, scale, mise_iter=2):
        """
        NKSR로 메쉬 재구성
        
        Args:
            mise_iter: MISE 반복 횟수 (높을수록 고해상도, 2-3 권장)
        """
        print(f"\n🎨 Reconstructing mesh with NKSR (mise_iter={mise_iter})...")
        print("   This may take several minutes for large point clouds...")
        
        # NKSR 재구성
        with torch.no_grad():
            field = self.reconstructor.reconstruct(
                points_torch, 
                normals_torch,
                detail_level=None  # 자동으로 스케일 결정
            )
            
            # 컬러 텍스처 필드 설정
            if colors_torch is not None:
                print("   Adding color texture field...")
                field.set_texture_field(
                    nksr.fields.PCNNField(points_torch, colors_torch)
                )
            
            # Dual mesh 추출
            print(f"   Extracting dual mesh...")
            mesh = field.extract_dual_mesh(mise_iter=mise_iter)
        
        print(f"   ✓ Mesh vertices: {len(mesh.v):,}")
        print(f"   ✓ Mesh faces: {len(mesh.f):,}")
        
        # 원래 스케일로 복원
        vertices = mesh.v.cpu().numpy() * scale + centroid
        faces = mesh.f.cpu().numpy()
        vertex_colors = mesh.c.cpu().numpy() if mesh.c is not None else None
        
        return vertices, faces, vertex_colors
    
    
    def save_mesh(self, vertices, faces, vertex_colors, output_path):
        """PLY 형식으로 메쉬 저장"""
        print(f"\n💾 Saving mesh to: {output_path}")
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        
        if vertex_colors is not None:
            # 컬러 범위 클램핑 [0, 1]
            vertex_colors = np.clip(vertex_colors, 0, 1)
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            print("   ✓ Vertex colors included")
        
        # Normal 재계산 (부드러운 렌더링)
        mesh_o3d.compute_vertex_normals()
        
        # PLY 저장
        o3d.io.write_triangle_mesh(output_path, mesh_o3d, write_vertex_colors=True)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✓ Saved successfully ({file_size:.2f} MB)")
    
    
    def process(self, input_pcd_path, output_ply_path, 
                normal_k=30, normal_radius=0.5, 
                max_points=10000000, mise_iter=2):
        """
        전체 파이프라인 실행
        
        Args:
            input_pcd_path: 입력 PCD 파일 경로
            output_ply_path: 출력 PLY 파일 경로
            normal_k: Normal 계산 KNN
            normal_radius: Normal 계산 반경
            max_points: 최대 처리 포인트 (메모리 제약시 다운샘플링)
            mise_iter: 메쉬 해상도 (2-3 권장)
        """
        print("="*70)
        print("🌍 NKSR Point Cloud to Mesh Reconstruction Pipeline")
        print("="*70)
        
        # 1. 로드
        pcd, points, colors = self.load_point_cloud(input_pcd_path)
        
        # 2. Normal 계산
        points, normals, colors = self.compute_normals_gpu(
            pcd, points, k=normal_k, radius=normal_radius
        )
        
        # 3. 다운샘플링 (필요시)
        points, normals, colors = self.downsample_if_needed(
            points, normals, colors, max_points=max_points
        )
        
        # 4. 전처리
        points_torch, normals_torch, colors_torch, centroid, scale = \
            self.preprocess_for_nksr(points, normals, colors)
        
        # 5. 메쉬 재구성
        vertices, faces, vertex_colors = self.reconstruct_mesh(
            points_torch, normals_torch, colors_torch, 
            centroid, scale, mise_iter=mise_iter
        )
        
        # 6. 저장
        self.save_mesh(vertices, faces, vertex_colors, output_ply_path)
        
        print("\n" + "="*70)
        print("✅ Pipeline completed successfully!")
        print("="*70)


def main():
    """기본값은 configs/mesh.yaml > nksr. CLI 인자가 우선."""
    from src.common import load_config

    parser = argparse.ArgumentParser(description="NKSR Point Cloud to Mesh Reconstruction")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--input", "-i", required=True, help="Input PCD")
    parser.add_argument("--output", "-o", required=True, help="Output PLY")
    parser.add_argument("--device", default=None, help="CUDA device")
    parser.add_argument("--normal-k", type=int, default=None)
    parser.add_argument("--normal-radius", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--mise-iter", type=int, default=None)

    args = parser.parse_args()
    nk = load_config(args.config).mesh.nksr
    if args.device is None:
        args.device = nk.device
    if args.normal_k is None:
        args.normal_k = int(nk.normal_k)
    if args.normal_radius is None:
        args.normal_radius = float(nk.normal_radius)
    if args.max_points is None:
        args.max_points = int(nk.max_points)
    if args.mise_iter is None:
        args.mise_iter = int(nk.mise_iter)
    
    # 재구성기 초기화
    reconstructor = PointCloudMeshReconstructor(
        device=args.device
    )
    
    # 파이프라인 실행
    reconstructor.process(
        input_pcd_path=args.input,
        output_ply_path=args.output,
        normal_k=args.normal_k,
        normal_radius=args.normal_radius,
        max_points=args.max_points,
        mise_iter=args.mise_iter
    )


if __name__ == '__main__':
    main()
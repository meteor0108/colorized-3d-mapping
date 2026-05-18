import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from typing import Optional, Tuple, List
import time
import gc
from dataclasses import dataclass

@dataclass
class BoundingBox:
    """바운딩 박스 정보"""
    min_bound: np.ndarray
    max_bound: np.ndarray
    center: np.ndarray
    extent: np.ndarray

class ChunkedPoissonReconstruction:
    """
    메모리 효율적인 청크 기반 Poisson Surface Reconstruction
    대용량 포인트 클라우드를 공간적으로 분할하여 처리
    """
    
    def __init__(self, use_cuda: bool = True, verbose: bool = True):
        """
        Args:
            use_cuda: CUDA 사용 여부
            verbose: 상세 로그 출력
        """
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        if self.verbose:
            print(f"Using device: {self.device}")
            if self.device.type == 'cuda':
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"  Memory: {mem:.2f} GB")
    
    def load_point_cloud(self, file_path: str) -> o3d.geometry.PointCloud:
        """포인트 클라우드 로드"""
        if self.verbose:
            print(f"\nLoading point cloud from {file_path}...")
        start_time = time.time()
        
        pcd = o3d.io.read_point_cloud(file_path)
        
        if self.verbose:
            print(f"Loaded {len(pcd.points):,} points in {time.time() - start_time:.2f}s")
            print(f"Has colors: {pcd.has_colors()}")
            print(f"Has normals: {pcd.has_normals()}")
        
        return pcd
    
    def get_bounding_box(self, pcd: o3d.geometry.PointCloud) -> BoundingBox:
        """바운딩 박스 계산"""
        points = np.asarray(pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (min_bound + max_bound) / 2
        extent = max_bound - min_bound
        
        return BoundingBox(min_bound, max_bound, center, extent)
    
    def create_spatial_chunks(self, 
                             pcd: o3d.geometry.PointCloud,
                             chunk_size: float = None,
                             overlap: float = 0.1) -> List[Tuple[BoundingBox, o3d.geometry.PointCloud]]:
        """
        포인트 클라우드를 공간적으로 청크로 분할
        
        Args:
            pcd: 입력 포인트 클라우드
            chunk_size: 청크 크기 (None이면 자동 계산)
            overlap: 청크 간 중복 비율 (0.1 = 10%)
        
        Returns:
            (바운딩박스, 청크 포인트클라우드) 리스트
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CREATING SPATIAL CHUNKS")
            print("="*60)
        
        bbox = self.get_bounding_box(pcd)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        if self.verbose:
            print(f"Bounding box: {bbox.min_bound} to {bbox.max_bound}")
            print(f"Extent: {bbox.extent}")
        
        # 자동으로 청크 크기 계산 (메모리 기반)
        if chunk_size is None:
            num_points = len(points)
            # 청크당 최대 50만 포인트를 목표로
            target_points_per_chunk = 500000
            volume = np.prod(bbox.extent)
            point_density = num_points / volume
            chunk_volume = target_points_per_chunk / point_density
            chunk_size = chunk_volume ** (1/3)
            
            if self.verbose:
                print(f"Auto-calculated chunk size: {chunk_size:.2f}")
        
        # 각 축별로 청크 개수 계산
        overlap_size = chunk_size * overlap
        effective_chunk_size = chunk_size - overlap_size
        
        num_chunks_x = int(np.ceil(bbox.extent[0] / effective_chunk_size))
        num_chunks_y = int(np.ceil(bbox.extent[1] / effective_chunk_size))
        num_chunks_z = int(np.ceil(bbox.extent[2] / effective_chunk_size))
        
        total_chunks = num_chunks_x * num_chunks_y * num_chunks_z
        
        if self.verbose:
            print(f"Chunk grid: {num_chunks_x} x {num_chunks_y} x {num_chunks_z} = {total_chunks} chunks")
            print(f"Chunk size: {chunk_size:.2f} (overlap: {overlap*100:.0f}%)")
        
        chunks = []
        chunk_idx = 0
        
        for i in range(num_chunks_x):
            for j in range(num_chunks_y):
                for k in range(num_chunks_z):
                    # 청크 바운딩 박스 계산 (오버랩 포함)
                    chunk_min = bbox.min_bound + np.array([
                        i * effective_chunk_size - overlap_size/2,
                        j * effective_chunk_size - overlap_size/2,
                        k * effective_chunk_size - overlap_size/2
                    ])
                    chunk_max = chunk_min + chunk_size
                    
                    # 청크 내 포인트 필터링
                    mask = np.all((points >= chunk_min) & (points <= chunk_max), axis=1)
                    chunk_points = points[mask]
                    
                    if len(chunk_points) < 100:  # 너무 작은 청크는 스킵
                        continue
                    
                    # 청크 포인트 클라우드 생성
                    chunk_pcd = o3d.geometry.PointCloud()
                    chunk_pcd.points = o3d.utility.Vector3dVector(chunk_points)
                    
                    if colors is not None:
                        chunk_colors = colors[mask]
                        chunk_pcd.colors = o3d.utility.Vector3dVector(chunk_colors)
                    
                    chunk_bbox = BoundingBox(
                        chunk_min, chunk_max,
                        (chunk_min + chunk_max) / 2,
                        chunk_max - chunk_min
                    )
                    
                    chunks.append((chunk_bbox, chunk_pcd))
                    chunk_idx += 1
                    
                    if self.verbose and chunk_idx % 10 == 0:
                        print(f"  Created {chunk_idx} chunks...")
        
        if self.verbose:
            print(f"\nTotal valid chunks: {len(chunks)}")
            avg_points = np.mean([len(c[1].points) for c in chunks])
            print(f"Average points per chunk: {avg_points:,.0f}")
        
        return chunks
    
    def preprocess_point_cloud(self, 
                               pcd: o3d.geometry.PointCloud,
                               voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
        """
        포인트 클라우드 전처리 (메모리 효율적)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("PREPROCESSING")
            print("="*60)
        
        start_time = time.time()
        original_count = len(pcd.points)
        
        # 다운샘플링
        if self.verbose:
            print(f"Downsampling (voxel_size={voxel_size})...")
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        if self.verbose:
            print(f"Points: {original_count:,} → {len(pcd_down.points):,}")
            print(f"Time: {time.time() - start_time:.2f}s")
        
        # 메모리 정리
        del pcd
        gc.collect()
        
        return pcd_down
    
    def estimate_normals_chunked(self,
                                chunks: List[Tuple[BoundingBox, o3d.geometry.PointCloud]],
                                radius: float = 0.1,
                                max_nn: int = 30) -> List[Tuple[BoundingBox, o3d.geometry.PointCloud]]:
        """
        청크별로 법선 추정 (메모리 효율적)
        
        Args:
            chunks: 청크 리스트
            radius: 법선 추정 반경
            max_nn: 최대 이웃 수
        
        Returns:
            법선이 추가된 청크 리스트
        """
        if self.verbose:
            print("\n" + "="*60)
            print("ESTIMATING NORMALS (CHUNKED)")
            print("="*60)
        
        start_time = time.time()
        processed_chunks = []
        
        for idx, (bbox, chunk_pcd) in enumerate(chunks):
            if self.verbose and idx % 10 == 0:
                print(f"Processing chunk {idx+1}/{len(chunks)}...")
            
            # 청크별로 법선 추정
            chunk_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius,
                    max_nn=max_nn
                )
            )
            
            # 법선 방향 정렬 (간단한 방법 - 위쪽 방향 선호)
            normals = np.asarray(chunk_pcd.normals)
            # Z 방향이 음수인 법선을 뒤집음 (야외 환경 가정)
            flip_mask = normals[:, 2] < 0
            normals[flip_mask] *= -1
            chunk_pcd.normals = o3d.utility.Vector3dVector(normals)
            
            processed_chunks.append((bbox, chunk_pcd))
            
            # 주기적으로 메모리 정리
            if idx % 20 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        if self.verbose:
            print(f"\nNormal estimation completed in {time.time() - start_time:.2f}s")
        
        return processed_chunks
    
    def merge_chunks(self, 
                    chunks: List[Tuple[BoundingBox, o3d.geometry.PointCloud]]) -> o3d.geometry.PointCloud:
        """
        청크들을 하나의 포인트 클라우드로 병합
        
        Args:
            chunks: 청크 리스트
        
        Returns:
            병합된 포인트 클라우드
        """
        if self.verbose:
            print("\n" + "="*60)
            print("MERGING CHUNKS")
            print("="*60)
        
        start_time = time.time()
        
        all_points = []
        all_normals = []
        all_colors = []
        
        for idx, (bbox, chunk_pcd) in enumerate(chunks):
            all_points.append(np.asarray(chunk_pcd.points))
            all_normals.append(np.asarray(chunk_pcd.normals))
            if chunk_pcd.has_colors():
                all_colors.append(np.asarray(chunk_pcd.colors))
            
            # 메모리 절약을 위해 주기적으로 병합
            if idx % 50 == 0 and idx > 0:
                gc.collect()
        
        # 최종 병합
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        merged_pcd.normals = o3d.utility.Vector3dVector(np.vstack(all_normals))
        if all_colors:
            merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        
        if self.verbose:
            print(f"Merged {len(chunks)} chunks into {len(merged_pcd.points):,} points")
            print(f"Time: {time.time() - start_time:.2f}s")
        
        # 메모리 정리
        del chunks
        gc.collect()
        
        return merged_pcd
    
    def poisson_reconstruction(self,
                              pcd: o3d.geometry.PointCloud,
                              depth: int = 10) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """
        Poisson Surface Reconstruction
        """
        if self.verbose:
            print("\n" + "="*60)
            print("POISSON RECONSTRUCTION")
            print("="*60)
            print(f"Depth: {depth}")
            print(f"Points: {len(pcd.points):,}")
        
        start_time = time.time()
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        if self.verbose:
            print(f"\nReconstruction completed!")
            print(f"Vertices: {len(mesh.vertices):,}")
            print(f"Triangles: {len(mesh.triangles):,}")
            print(f"Time: {time.time() - start_time:.2f}s")
        
        # 메모리 정리
        del pcd
        gc.collect()
        
        return mesh, np.asarray(densities)
    
    def filter_mesh(self,
                   mesh: o3d.geometry.TriangleMesh,
                   densities: np.ndarray,
                   quantile: float = 0.01) -> o3d.geometry.TriangleMesh:
        """밀도 기반 필터링"""
        if self.verbose:
            print("\n" + "="*60)
            print("FILTERING MESH")
            print("="*60)
        
        start_time = time.time()
        threshold = np.quantile(densities, quantile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        if self.verbose:
            print(f"Filtered vertices")
            print(f"Remaining: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
            print(f"Time: {time.time() - start_time:.2f}s")
        
        return mesh
    
    def colorize_mesh(self,
                     mesh: o3d.geometry.TriangleMesh,
                     pcd: o3d.geometry.PointCloud,
                     batch_size: int = 100000) -> o3d.geometry.TriangleMesh:
        """
        배치 처리로 메쉬 컬러라이제이션 (메모리 효율적)
        """
        if not pcd.has_colors():
            return mesh
        
        if self.verbose:
            print("\n" + "="*60)
            print("COLORIZING MESH (BATCHED)")
            print("="*60)
        
        start_time = time.time()
        vertices = np.asarray(mesh.vertices)
        num_vertices = len(vertices)
        mesh_colors = np.zeros((num_vertices, 3))
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        pcd_colors = np.asarray(pcd.colors)
        
        # 배치 단위로 처리
        num_batches = (num_vertices + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_vertices)
            
            for i in range(start_idx, end_idx):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(vertices[i], 1)
                mesh_colors[i] = pcd_colors[idx[0]]
            
            if self.verbose and (batch_idx + 1) % 10 == 0:
                progress = (end_idx / num_vertices) * 100
                print(f"  Progress: {progress:.1f}% ({end_idx:,}/{num_vertices:,})")
            
            # 주기적 메모리 정리
            if batch_idx % 20 == 0:
                gc.collect()
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        
        if self.verbose:
            print(f"\nColorization completed in {time.time() - start_time:.2f}s")
        
        return mesh
    
    def post_process_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """메쉬 후처리"""
        if self.verbose:
            print("\n" + "="*60)
            print("POST-PROCESSING")
            print("="*60)
        
        start_time = time.time()
        
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        
        if self.verbose:
            print(f"Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
            print(f"Time: {time.time() - start_time:.2f}s")
        
        return mesh
    
    def save_mesh_ply(self, mesh: o3d.geometry.TriangleMesh, output_path: str):
        """PLY 형식으로 저장"""
        if self.verbose:
            print("\n" + "="*60)
            print("SAVING MESH")
            print("="*60)
        
        if not output_path.endswith('.ply'):
            output_path += '.ply'
        
        start_time = time.time()
        success = o3d.io.write_triangle_mesh(
            output_path,
            mesh,
            write_ascii=False,
            compressed=True,
            write_vertex_normals=True,
            write_vertex_colors=mesh.has_vertex_colors()
        )
        
        if success:
            file_size = Path(output_path).stat().st_size / (1024 * 1024)
            if self.verbose:
                print(f"✓ Saved to: {output_path}")
                print(f"  Size: {file_size:.2f} MB")
                print(f"  Time: {time.time() - start_time:.2f}s")
        else:
            print("✗ Failed to save mesh")
    
    def reconstruct(self,
                   input_path: str,
                   output_path: str,
                   voxel_size: float = 0.03,
                   chunk_size: float = None,
                   depth: int = 10,
                   density_quantile: float = 0.01) -> o3d.geometry.TriangleMesh:
        """
        메모리 효율적인 전체 재구성 파이프라인
        
        Args:
            input_path: 입력 PCD 파일
            output_path: 출력 PLY 파일
            voxel_size: 다운샘플링 크기
            chunk_size: 청크 크기 (None=자동)
            depth: Poisson 깊이
            density_quantile: 필터링 임계값
        """
        print("\n" + "="*60)
        print("MEMORY-EFFICIENT POISSON RECONSTRUCTION")
        print("="*60)
        total_start = time.time()
        
        # 1. 로드
        pcd = self.load_point_cloud(input_path)
        
        # 2. 전처리
        pcd = self.preprocess_point_cloud(pcd, voxel_size=voxel_size)
        
        # 3. 청크 생성
        chunks = self.create_spatial_chunks(pcd, chunk_size=chunk_size, overlap=0.15)
        del pcd
        gc.collect()
        
        # 4. 청크별 법선 추정
        chunks = self.estimate_normals_chunked(chunks, radius=voxel_size*3, max_nn=30)
        
        # 5. 청크 병합
        pcd_with_normals = self.merge_chunks(chunks)
        del chunks
        gc.collect()
        
        # 6. Poisson 재구성
        mesh, densities = self.poisson_reconstruction(pcd_with_normals, depth=depth)
        
        # 7. 필터링
        mesh = self.filter_mesh(mesh, densities, quantile=density_quantile)
        del densities
        gc.collect()
        
        # 8. 컬러라이제이션
        mesh = self.colorize_mesh(mesh, pcd_with_normals, batch_size=50000)
        del pcd_with_normals
        gc.collect()
        
        # 9. 후처리
        mesh = self.post_process_mesh(mesh)
        
        # 10. 저장
        self.save_mesh_ply(mesh, output_path)
        
        print("\n" + "="*60)
        print(f"TOTAL TIME: {time.time() - total_start:.2f}s")
        print("="*60 + "\n")
        
        return mesh


# 사용 예제 (configs/mesh.yaml > poisson)
if __name__ == "__main__":
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step7: Chunked Poisson surface reconstruction")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--input", required=True, help="입력 PCD")
    parser.add_argument("--output", required=True, help="출력 PLY")
    args = parser.parse_args()
    cfg = load_config(args.config)
    poisson_cfg = cfg.mesh.poisson

    reconstructor = ChunkedPoissonReconstruction(
        use_cuda=bool(poisson_cfg.use_cuda),
        verbose=bool(poisson_cfg.verbose),
    )

    mesh = reconstructor.reconstruct(
        input_path=args.input,
        output_path=args.output,
        voxel_size=0.03,
        chunk_size=None,
        depth=int(poisson_cfg.depth),
        density_quantile=0.1,
    )
    
    # 시각화
    print("\nLaunching visualizer...")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Reconstructed Mesh",
        width=1920,
        height=1080,
        mesh_show_back_face=True
    )
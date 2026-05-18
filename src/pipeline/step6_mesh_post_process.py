import open3d as o3d
import numpy as np
import argparse


class MeshPostProcessor:
    """메쉬 후처리 유틸리티"""
    
    @staticmethod
    def load_mesh(mesh_path):
        """메쉬 로드"""
        print(f"📂 Loading mesh from: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        print(f"   Vertices: {len(mesh.vertices):,}")
        print(f"   Faces: {len(mesh.triangles):,}")
        print(f"   Has colors: {mesh.has_vertex_colors()}")
        return mesh
    
    @staticmethod
    def save_mesh(mesh, output_path):
        """메쉬 저장"""
        print(f"💾 Saving mesh to: {output_path}")
        o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
        print("   ✓ Saved successfully")
    
    @staticmethod
    def remove_outliers(mesh, nb_neighbors=20, std_ratio=2.0):
        """
        통계적 이상치 제거
        
        Args:
            nb_neighbors: 이웃 개수
            std_ratio: 표준편차 비율 (작을수록 공격적)
        """
        print(f"\n🔍 Removing outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
        
        # Point Cloud로 변환
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        
        # 이상치 필터링
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        
        # 필터링된 vertex로 새 메쉬 생성
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        
        # 유효한 vertex만 유지
        valid_vertices = set(ind)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(ind)}
        
        # Face 필터링 (모든 vertex가 유효한 face만)
        valid_faces = []
        for face in faces:
            if all(v in valid_vertices for v in face):
                valid_faces.append([vertex_map[v] for v in face])
        
        # 새 메쉬 생성
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(vertices[ind])
        new_mesh.triangles = o3d.utility.Vector3iVector(valid_faces)
        
        if colors is not None:
            new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[ind])
        
        removed = len(mesh.vertices) - len(new_mesh.vertices)
        print(f"   ✓ Removed {removed:,} vertices ({removed/len(mesh.vertices)*100:.1f}%)")
        print(f"   ✓ New vertex count: {len(new_mesh.vertices):,}")
        
        return new_mesh
    
    @staticmethod
    def simplify_mesh(mesh, target_triangles=1000000):
        """
        메쉬 단순화 (삼각형 수 감소)
        
        Args:
            target_triangles: 목표 삼각형 수
        """
        print(f"\n⚡ Simplifying mesh to ~{target_triangles:,} triangles...")
        
        current_triangles = len(mesh.triangles)
        if current_triangles <= target_triangles:
            print(f"   ℹ️  Already below target ({current_triangles:,} triangles)")
            return mesh
        
        # Quadric decimation
        simplified = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
        
        reduction = 100 * (1 - len(simplified.triangles) / current_triangles)
        print(f"   ✓ Reduced from {current_triangles:,} to {len(simplified.triangles):,} triangles")
        print(f"   ✓ Reduction: {reduction:.1f}%")
        
        return simplified
    
    @staticmethod
    def smooth_mesh(mesh, iterations=1, lambda_filter=0.5):
        """
        메쉬 스무딩 (Laplacian smoothing)
        
        Args:
            iterations: 반복 횟수
            lambda_filter: 스무딩 강도 (0-1)
        """
        print(f"\n🎨 Smoothing mesh (iterations={iterations}, lambda={lambda_filter})...")
        
        smoothed = mesh.filter_smooth_laplacian(
            number_of_iterations=iterations,
            lambda_filter=lambda_filter
        )
        
        print("   ✓ Smoothing completed")
        return smoothed
    
    @staticmethod
    def remove_small_components(mesh, min_triangles=100):
        """
        작은 연결 요소 제거
        
        Args:
            min_triangles: 최소 삼각형 수
        """
        print(f"\n🧹 Removing small components (min_triangles={min_triangles})...")
        
        # 연결 요소 분할
        triangle_clusters, cluster_n_triangles, cluster_area = \
            mesh.cluster_connected_triangles()
        
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # 큰 요소만 유지
        large_clusters = np.where(cluster_n_triangles >= min_triangles)[0]
        triangles_to_keep = np.isin(triangle_clusters, large_clusters)
        
        mesh_cleaned = mesh.select_by_index(np.where(triangles_to_keep)[0])
        
        removed_clusters = len(cluster_n_triangles) - len(large_clusters)
        print(f"   ✓ Removed {removed_clusters} small components")
        print(f"   ✓ Kept {len(large_clusters)} large components")
        
        return mesh_cleaned
    
    @staticmethod
    def compute_statistics(mesh):
        """메쉬 통계 출력"""
        print("\n📊 Mesh Statistics:")
        print(f"   Vertices: {len(mesh.vertices):,}")
        print(f"   Triangles: {len(mesh.triangles):,}")
        print(f"   Has colors: {mesh.has_vertex_colors()}")
        print(f"   Has normals: {mesh.has_vertex_normals()}")
        
        # Bounding box
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        print(f"   Bounding box: [{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}]")
        
        # Surface area
        area = mesh.get_surface_area()
        print(f"   Surface area: {area:.2f} m²")
        
        # Volume (watertight mesh only)
        if mesh.is_watertight():
            volume = mesh.get_volume()
            print(f"   Volume: {volume:.2f} m³")
            print(f"   ✓ Mesh is watertight")
        else:
            print(f"   ⚠️  Mesh is NOT watertight")
    
    @staticmethod
    def visualize_mesh(mesh):
        """메쉬 시각화"""
        print("\n👁️  Visualizing mesh...")
        print("   Controls:")
        print("   - Mouse drag: Rotate")
        print("   - Mouse wheel: Zoom")
        print("   - Shift + Mouse drag: Pan")
        
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Mesh Viewer",
            width=1280,
            height=720
        )


def main():
    parser = argparse.ArgumentParser(description='Mesh Post-processing Utilities')
    parser.add_argument('--input', '-i', required=True, help='Input mesh file')
    parser.add_argument('--output', '-o', help='Output mesh file')
    
    # Post-processing options
    parser.add_argument('--remove-outliers', action='store_true',
                       help='Remove statistical outliers')
    parser.add_argument('--outlier-neighbors', type=int, default=20,
                       help='Neighbors for outlier removal (default: 20)')
    parser.add_argument('--outlier-std', type=float, default=2.0,
                       help='Std ratio for outlier removal (default: 2.0)')
    
    parser.add_argument('--simplify', type=int, metavar='TRIANGLES',
                       help='Simplify to target triangle count')
    
    parser.add_argument('--smooth', action='store_true',
                       help='Apply Laplacian smoothing')
    parser.add_argument('--smooth-iter', type=int, default=1,
                       help='Smoothing iterations (default: 1)')
    parser.add_argument('--smooth-lambda', type=float, default=0.5,
                       help='Smoothing strength 0-1 (default: 0.5)')
    
    parser.add_argument('--remove-small', type=int, metavar='MIN_TRIANGLES',
                       help='Remove components smaller than MIN_TRIANGLES')
    
    parser.add_argument('--stats', action='store_true',
                       help='Print mesh statistics')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the mesh')
    
    args = parser.parse_args()
    
    # Load mesh
    processor = MeshPostProcessor()
    mesh = processor.load_mesh(args.input)
    
    # Apply post-processing
    if args.remove_outliers:
        mesh = processor.remove_outliers(
            mesh, 
            nb_neighbors=args.outlier_neighbors,
            std_ratio=args.outlier_std
        )
    
    if args.simplify:
        mesh = processor.simplify_mesh(mesh, target_triangles=args.simplify)
    
    if args.smooth:
        mesh = processor.smooth_mesh(
            mesh,
            iterations=args.smooth_iter,
            lambda_filter=args.smooth_lambda
        )
    
    if args.remove_small:
        mesh = processor.remove_small_components(mesh, min_triangles=args.remove_small)
    
    # Recompute normals
    mesh.compute_vertex_normals()
    
    # Statistics
    if args.stats:
        processor.compute_statistics(mesh)
    
    # Save
    if args.output:
        processor.save_mesh(mesh, args.output)
    
    # Visualize
    if args.visualize:
        processor.visualize_mesh(mesh)


if __name__ == '__main__':
    main()
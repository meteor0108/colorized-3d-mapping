import os
import copy
import numpy as np
import open3d as o3d
import natsort

def get_first_pcd_path(folder_path):
    files = os.listdir(folder_path)
    files = natsort.natsorted(files)
    for f in files:
        if f.endswith('.pcd'):
            return os.path.join(folder_path, f)
    return None

def main():
    # ==========================================
    # 1. 설정 및 경로 (configs/projection.yaml input_folder)
    # ==========================================
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step0: Calibration result test")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--folder", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    file_diractory = args.folder or cfg.projection.input_folder

    pointcloud_folder1 = os.path.join(file_diractory, "ouster1/points")
    pointcloud_folder2 = os.path.join(file_diractory, "ouster2/points")
    pointcloud_folder3 = os.path.join(file_diractory, "ouster3/points")

    # ==========================================
    # 2. 변환 행렬 정의 (기존 코드와 동일)
    # ==========================================
    extrinsic_ouster1 = np.array([
        [-4.72864607e-01, -8.76671114e-01 ,-8.85822813e-02, -5.17475772e-01],
        [ 1.25955567e-02, 9.37965264e-02, -9.95511709e-01, -6.61842700e-04,],
        [ 8.81045070e-01, -4.71857997e-01, -3.33108970e-02, -6.78935532e-02],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    extrinsic_ouster2 = np.array([
        [ 9.06068473e-03, -9.99958607e-01, -8.29860879e-04, -2.71261603e-02],
        [-1.29545341e-02, 7.12443573e-04, -9.99915833e-01, -1.03037610e-01],
        [ 9.99875034e-01, 9.07067258e-03, -1.29475426e-02, -8.95660947e-02],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    extrinsic_ouster3 = np.array([
        [ 0.49316712, -0.86663967, 0.075643, 0.4635232 ],
        [ 0.0065567, -0.08324712, -0.99650736, -0.01027599],
        [ 0.86990988, 0.49194064, -0.03537245, -0.07933007],
        [ 0., 0., 0., 1. ]
    ])

    extrinsic_gps2ouster2_raw = np.array([
        [-0.986173 ,  0.0251448, 0.163798  ,-0.406935],
        [-0.0267946, -0.99961  ,-0.00787014, 0.179598],
        [ 0.163536 , -0.0121502, 0.986463  , 0.221061],
        [ 0.       ,  0.       , 0.        , 1.      ]
    ])

    # 기존 코드의 보정 로직 적용
    R_fix = np.array([
        [-1,  0,  0,  0], 
        [ 0, -1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ])
    
    extrinsic_gps2ouster2 = extrinsic_gps2ouster2_raw @ R_fix
    extrinsic_gps2ouster2[1, 3] = -extrinsic_gps2ouster2[1, 3] # Y축 이동값 반전

    # L1, L3의 GPS 기준 변환 행렬 계산 (L1->Cam->L2->GPS)
    # L2 to Cam (L2 기준 역행렬)
    T_L2_to_Cam = np.linalg.inv(extrinsic_ouster2) 

    # 최종 변환 행렬 계산
    ext_gps_L1 = extrinsic_gps2ouster2 @ T_L2_to_Cam @ extrinsic_ouster1
    ext_gps_L2 = extrinsic_gps2ouster2 
    ext_gps_L3 = extrinsic_gps2ouster2 @ T_L2_to_Cam @ extrinsic_ouster3

    # ==========================================
    # 3. PCD 로드 및 시각화
    # ==========================================
    paths = [
        (pointcloud_folder1, ext_gps_L1, [1, 0, 0], "Ouster1 (Red)"),   # 빨강
        (pointcloud_folder2, ext_gps_L2, [0, 1, 0], "Ouster2 (Green)"), # 초록
        (pointcloud_folder3, ext_gps_L3, [0, 0, 1], "Ouster3 (Blue)")   # 파랑
    ]

    vis_geometry = []
    
    # 기준 좌표축 (GPS/Vehicle Origin) 추가
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    vis_geometry.append(axis)

    print("=== Processing Point Clouds ===")
    for folder, transform_matrix, color, label in paths:
        pcd_path = get_first_pcd_path(folder)
        
        if pcd_path and os.path.exists(pcd_path):
            print(f"Loading {label}: {pcd_path}")
            pcd = o3d.io.read_point_cloud(pcd_path)
            
            # 1. 색상 입히기 (구분을 위해 단색 처리)
            pcd.paint_uniform_color(color)
            
            # 2. 좌표 변환 (Local LiDAR -> GPS/Vehicle Frame)
            pcd.transform(transform_matrix)
            
            vis_geometry.append(pcd)
        else:
            print(f"[Warning] 파일을 찾을 수 없음: {folder}")

    if len(vis_geometry) <= 1:
        print("시각화할 포인트 클라우드가 없습니다.")
        return

    print("\n=== Visualization Start ===")
    print("Red: Ouster1 | Green: Ouster2 | Blue: Ouster3")
    print("X: Red Axis, Y: Green Axis, Z: Blue Axis (Center is GPS/Vehicle Origin)")
    
    o3d.visualization.draw_geometries(vis_geometry, 
                                      window_name="Multi-LiDAR Coordinate Check",
                                      width=1200, height=800)

if __name__ == "__main__":
    main()
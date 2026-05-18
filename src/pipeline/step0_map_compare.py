import os
import glob
import re
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import natsort
from collections import defaultdict

# ==========================================
# 설정 구간 (configs/trajectory.yaml > map_compare)
# ==========================================
from src.common import load_config
_cfg = load_config("default.yaml")
PCD_ROOT_DIR = _cfg.trajectory.map_compare.pcd_root
TRAJ_ROOT_DIR = _cfg.paths.gps_results_root
SAVE_DIR = _cfg.trajectory.map_compare.save_dir
SEARCH_DEPTH = _cfg.trajectory.map_compare.search_depth

# ==========================================
# 유틸리티 함수
# ==========================================

def find_files_with_depth(root_dir, extension, max_depth):
    file_paths = []
    root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir): return []
    root_depth = root_dir.rstrip(os.sep).count(os.sep)

    for root, dirs, files in os.walk(root_dir):
        current_depth = root.rstrip(os.sep).count(os.sep)
        if current_depth - root_depth > max_depth: continue
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

def parse_key(filename):
    """
    파일명에서 _ouster2_part... 부분을 제거하여 폴더명(Key)만 추출
    예: '2024-03-19..._ouster2_part3.pcd' -> '2024-03-19...'
    """
    basename = os.path.basename(filename)
    # _ouster2 앞부분을 Key로 사용
    if "_ouster2" in basename:
        return basename.split("_ouster2")[0]
    return os.path.splitext(basename)[0]

def get_csv_key(filename):
    return os.path.splitext(os.path.basename(filename))[0]

# ==========================================
# 메인 로직
# ==========================================

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print(f"[Info] Grouping Mode: 같은 데이터셋의 Part들을 하나로 합칩니다.")

    # 1. 파일 검색
    pcd_files = find_files_with_depth(PCD_ROOT_DIR, '.pcd', SEARCH_DEPTH)
    csv_files = find_files_with_depth(TRAJ_ROOT_DIR, '.csv', SEARCH_DEPTH)
    
    csv_dict = {get_csv_key(f): f for f in csv_files}

    # 2. PCD 파일 그룹화 (Key: 데이터셋 이름, Value: 파일 경로 리스트)
    pcd_groups = defaultdict(list)
    for f in pcd_files:
        key = parse_key(f)
        pcd_groups[key].append(f)

    # 키 정렬 (날짜순 처리)
    sorted_keys = natsort.natsorted(pcd_groups.keys())
    
    print(f"[Info] 총 {len(sorted_keys)}개의 데이터셋(폴더)을 처리합니다.\n")

    # 3. 그룹별 처리 루프
    for idx, key in enumerate(sorted_keys):
        pcd_list = natsort.natsorted(pcd_groups[key]) # part0, part1 순서 정렬
        
        # CSV 매칭 확인
        if key not in csv_dict:
            # print(f"[Skip] CSV not found for {key}")
            continue

        csv_path = csv_dict[key]
        save_filename = f"{key}_Merged.png"
        save_path = os.path.join(SAVE_DIR, save_filename)
        
        # 이미 존재하면 스킵
        if os.path.exists(save_path):
            print(f"[{idx+1}/{len(sorted_keys)}] Skipping {key} (Already exists)")
            continue

        print(f"[{idx+1}/{len(sorted_keys)}] Merging {key} ({len(pcd_list)} parts)...")

        try:
            # --- [Step 1] 전체 맵 포인트 로드 및 병합 ---
            all_x = []
            all_y = []
            
            for p_path in pcd_list:
                # 메모리 효율을 위해 읽자마자 numpy 변환 후 o3d 객체 삭제
                pcd = o3d.io.read_point_cloud(p_path)
                if pcd.is_empty(): continue
                
                # 시각화용 다운샘플링 (너무 빽빽하면 그리기 느림)
                pcd = pcd.voxel_down_sample(voxel_size=0.5)
                pts = np.asarray(pcd.points)
                
                if len(pts) > 0:
                    all_x.append(pts[:, 0])
                    all_y.append(pts[:, 1])
                
                del pcd # 메모리 해제

            if not all_x:
                print("  -> Empty map data.")
                continue

            # 리스트 합치기 (Numpy Concat)
            map_x = np.concatenate(all_x)
            map_y = np.concatenate(all_y)

            # --- [Step 2] 전체 CSV 로드 ---
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            if 'utm_x' not in df.columns:
                print("  -> CSV format error")
                continue

            # 전체 궤적의 시작점을 0,0으로 맞춤 (PCD 데이터와 좌표계 통일)
            start_x = df['utm_x'].iloc[0]
            start_y = df['utm_y'].iloc[0]

            traj_x = df['utm_x'].values - start_x
            traj_y = df['utm_y'].values - start_y

            # --- [Step 3] 시각화 ---
            fig, ax = plt.subplots(figsize=(12, 12))

            # 맵 (검은색 점)
            ax.scatter(map_x, map_y, c='black', s=0.1, alpha=0.3, label='Merged Map')
            
            # 궤적 (빨간색 실선)
            ax.plot(traj_x, traj_y, c='red', linewidth=1.0, alpha=0.8, label='Full Trajectory')
            
            # 시작점/끝점
            ax.scatter(traj_x[0], traj_y[0], c='blue', marker='*', s=150, label='Start', zorder=10)
            ax.scatter(traj_x[-1], traj_y[-1], c='green', marker='x', s=100, label='End', zorder=10)

            ax.set_title(f"Full Map & Trajectory: {key}")
            ax.axis('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend()

            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved: {save_filename}")

        except Exception as e:
            print(f"  [Error] {e}")

    print("\n[Done] 모든 병합 및 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()
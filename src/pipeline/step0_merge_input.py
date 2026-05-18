import open3d as o3d
import numpy as np
import pandas as pd
import os
import glob
import gc
import natsort
from dataclasses import dataclass
from typing import List

# ==============================================================================
# 구조체 정의: 하나의 '주행 데이터(폴더)' 단위를 관리
# ==============================================================================
@dataclass
class FolderMapInfo:
    folder_name: str          # 예: "2024-03-19-15-05-15_0"
    part_paths: List[str]     # 해당 폴더에 속한 모든 part.pcd 파일 경로들
    abs_pos: np.ndarray       # [x, y, z] (절대 UTM 좌표)
    processed: bool = False

def main():
    # ==================================================================
    # 1. 설정 (configs/merge.yaml input_merge)
    # ==================================================================
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step0: Input-list UTM merge")
    parser.add_argument("--config", default="default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    im = cfg.merge.input_merge

    map_root_path = im.map_root
    data_root_path = cfg.paths.data_root

    save_dir = os.path.join(map_root_path, im.save_subdir)
    os.makedirs(save_dir, exist_ok=True)

    INPUT_LIST = list(im.input_list)
    SEARCH_PATTERN_TEMPLATE = im.part_pattern
    SAVE_FILENAME = im.save_filename

    # GPU 설정
    device = o3d.core.Device("CUDA:0" if o3d.core.cuda.is_available() else "CPU")
    print(f"System Device: {device} | Processing {len(INPUT_LIST)} folders\n")

    # ==================================================================
    # 2. 인풋 리스트 기반 파일 스캔 및 좌표 매칭
    # ==================================================================
    folder_info_list = []
    print(f"Linking navigation data for INPUT_LIST...")

    for folder_name in INPUT_LIST:
        # 2-1. 해당 폴더의 part 파일 검색 (폴더명이 파일명에 포함된다고 가정)
        # 예: 2024-03-19-15-05-15_0_ouster2_part0.pcd
        search_pattern = os.path.join(map_root_path, SEARCH_PATTERN_TEMPLATE.format(folder_name=folder_name))
        part_files = sorted(glob.glob(search_pattern))
        
        if not part_files:
            print(f"  [Warning] No pcd files found for {folder_name}, skipping.")
            continue

        # 2-2. 해당 폴더의 Navigation 파일 찾기 (원본 데이터 경로 탐색)
        nav_files = glob.glob(os.path.join(data_root_path, "**", folder_name, "navigation.csv"), recursive=True)
        
        if not nav_files:
            print(f"  [Warning] No nav file for {folder_name}, skipping.")
            continue
        
        try:
            # 2-3. UTM 좌표 읽기 (Shift만 적용, Rotation 미적용)
            df = pd.read_csv(nav_files[0])
            sx, sy, sz = df.iloc[0][['position_x', 'position_y', 'position_z']]
            
            # [수정됨] Universal Origin 제거 -> 실제 UTM 좌표 그대로 사용
            abs_pos = np.array([sx, sy, sz])
            
            # 정보 저장
            info = FolderMapInfo(
                folder_name=folder_name,
                part_paths=natsort.natsorted(part_files), # part 순서 정렬
                abs_pos=abs_pos,
                processed=False
            )
            folder_info_list.append(info)
            print(f"  -> Added: {folder_name} | UTM Origin: {abs_pos}")
            
        except Exception as e:
            print(f"Error reading nav for {folder_name}: {e}")
            continue

    print(f"\n-> Ready to process {len(folder_info_list)} valid folders.\n")

    # ==================================================================
    # 3. 병합 루프 (Input List 전체를 하나로 병합)
    # ==================================================================
    if not folder_info_list:
        print("No valid folders to merge.")
        return

    # 저장 파일명
    save_name = SAVE_FILENAME
    save_path = os.path.join(save_dir, save_name)
    
    print(f"=== Merging {len(folder_info_list)} folders into actual UTM coordinates ===")

    # GPU 메모리 관리를 위한 Tensor PointCloud 초기화
    group_pcd_t = None 
    
    for folder in folder_info_list:
        print(f"  -> Processing Folder: {folder.folder_name} ({len(folder.part_paths)} parts)")
        
        # 해당 폴더의 모든 파트 파일 순회
        for part_path in folder.part_paths:
            try:
                # 1. 파일 로드 (CPU)
                pcd_part = o3d.io.read_point_cloud(part_path)
                if len(pcd_part.points) == 0: continue
                
                # 2. 위치 보정 (Actual UTM Shift)
                # 로컬(0,0,0) 기준 맵을 UTM 절대 좌표로 이동 (Rotation 없음)
                pcd_part.translate(folder.abs_pos)
                
                # 3. GPU로 변환 및 병합
                # UTM 좌표계는 숫자가 매우 크므로 float64 사용 권장 (Open3D Tensor가 지원 시)
                part_t = o3d.t.geometry.PointCloud.from_legacy(pcd_part, device=device)
                
                if group_pcd_t is None:
                    group_pcd_t = part_t
                else:
                    group_pcd_t = group_pcd_t.append(part_t)
                
                # 4. 메모리 정리
                del pcd_part, part_t
                
            except Exception as e:
                print(f"    Error on {os.path.basename(part_path)}: {e}")
        
        # [수정됨] 다운샘플링 로직 삭제됨 (요청사항 반영)
        
        folder.processed = True
        gc.collect()

    # ==================================================================
    # 4. 결과 저장
    # ==================================================================
    if group_pcd_t is not None:
        print(f"  -> Saving merged map...")
        # Tensor -> Legacy 변환 후 저장
        final_pcd = group_pcd_t.to_legacy()
        o3d.io.write_point_cloud(save_path, final_pcd, write_ascii=False)
        print(f"  -> Saved: {save_path}")
        
        del group_pcd_t, final_pcd
    else:
        print("Result is empty.")

    o3d.core.cuda.release_cache()
    print("\nProcess finished.")

if __name__ == "__main__":
    main()
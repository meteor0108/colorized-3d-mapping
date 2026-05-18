import open3d as o3d
import numpy as np
import pandas as pd
import os
import glob
import gc
import natsort
from dataclasses import dataclass
from typing import List

@dataclass
class FolderMapInfo:
    folder_name: str
    part_paths: List[str]
    start_pos: np.ndarray
    processed: bool = False

def main():
    # ==================================================================
    # 1. 설정 (configs/merge.yaml auto_merge)
    # ==================================================================
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step0: Auto-grouped global map merge")
    parser.add_argument("--config", default="default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    am = cfg.merge.auto_merge

    map_root_path = am.map_root
    data_root_path = cfg.paths.data_root

    save_dir = os.path.join(map_root_path, am.save_subdir)
    os.makedirs(save_dir, exist_ok=True)

    voxel_size = am.voxel_size
    GROUP_RADIUS = am.group_radius
    MAX_FOLDERS_PER_GROUP = am.max_folders_per_group
    BAN_LIST = list(am.ban_list or [])

    # [설정] GPU 2대 디바이스 정의 (use_dual_gpu=false면 단일 GPU 사용)
    device_0 = o3d.core.Device("CUDA:0")
    device_1 = o3d.core.Device("CUDA:1" if am.use_dual_gpu else "CUDA:0")
    device_cpu = o3d.core.Device("CPU:0")
    
    # GPU 사용 가능 여부 확인
    if not o3d.core.cuda.is_available():
        print("Error: CUDA not available. Using CPU only.")
        device_0 = device_cpu
        device_1 = device_cpu
    
    print(f"Using Devices -> Primary: {device_0} | Secondary: {device_1}")
    print(f"Merge Strategy -> Load Balancing on GPUs -> Accumulate on CPU RAM\n")

    # ==================================================================
    # 2. 파일 스캔 및 Universal Origin 설정
    # ==================================================================
    print("Scanning part files...")
    all_part_files = sorted(glob.glob(os.path.join(map_root_path, "*_part*.pcd")))
    
    if not all_part_files:
        print("No files found.")
        return

    first_file_name = os.path.basename(all_part_files[0])
    first_folder_name = first_file_name.split("_full3_vgicp_pgo")[0] 
    
    nav_search = glob.glob(os.path.join(data_root_path, "**", first_folder_name, "navigation.csv"), recursive=True)
    if not nav_search: return
        
    df_origin = pd.read_csv(nav_search[0])
    uni_x, uni_y, uni_z = df_origin.iloc[0][['position_x', 'position_y', 'position_z']]
    universal_origin = np.array([uni_x, uni_y, uni_z])
    print(f"✅ Universal Origin: {universal_origin}\n")

    # ==================================================================
    # 3. 폴더별 그룹핑
    # ==================================================================
    folder_dict = {}
    for pcd_path in all_part_files:
        filename = os.path.basename(pcd_path)
        folder_name = filename.split("_full3_vgicp_pgo")[0]
        if folder_name in BAN_LIST: continue
        if folder_name not in folder_dict: folder_dict[folder_name] = []
        folder_dict[folder_name].append(pcd_path)

    folder_info_list = []
    for folder_name, paths in folder_dict.items():
        nav_files = glob.glob(os.path.join(data_root_path, "**", folder_name, "navigation.csv"), recursive=True)
        if not nav_files: continue
        try:
            df = pd.read_csv(nav_files[0])
            sx, sy, sz = df.iloc[0][['position_x', 'position_y', 'position_z']]
            local_pos = np.array([sx, sy, sz]) - universal_origin
            info = FolderMapInfo(folder_name, natsort.natsorted(paths), local_pos)
            folder_info_list.append(info)
        except: continue

    # ==================================================================
    # 5. 병합 루프
    # ==================================================================
    group_id = 0
    
    while True:
        unprocessed = [f for f in folder_info_list if not f.processed]
        if not unprocessed: break
            
        seed_folder = unprocessed[0]
        current_group = []
        for target in unprocessed:
            dist = np.linalg.norm(seed_folder.start_pos - target.start_pos)
            if dist <= GROUP_RADIUS:
                current_group.append(target)
                if len(current_group) >= MAX_FOLDERS_PER_GROUP: break
        
        group_id += 1
        save_name = f"Global_Map_Area_{group_id:03d}_(Center_{seed_folder.folder_name}).pcd"
        save_path = os.path.join(save_dir, save_name)
        
        print(f"=== Group {group_id}: Merging {len(current_group)} folders ===")
        
        if os.path.exists(save_path):
            print(f"  -> Skipping (Exists)")
            for f in current_group: f.processed = True
            continue

        accumulated_parts_cpu = [] 
        
        for folder in current_group:
            print(f"  -> Folder: {folder.folder_name} ({len(folder.part_paths)} parts)")
            
            for i, part_path in enumerate(folder.part_paths):
                try:
                    # 1. 로드 (CPU)
                    pcd_part = o3d.io.read_point_cloud(part_path)
                    if len(pcd_part.points) == 0: continue
                    
                    # 2. 위치 이동 (CPU)
                    pcd_part.translate(folder.start_pos)
                    
                    # 3. 1차 다운샘플링 (CPU)
                    pcd_part = pcd_part.voxel_down_sample(voxel_size)

                    # 4. GPU 분산 처리 (Round Robin)
                    target_device = device_0 if i % 2 == 0 else device_1
                    
                    # 5. Tensor 변환 (GPU로 전송)
                    part_t = o3d.t.geometry.PointCloud.from_legacy(pcd_part, device=target_device)
                    
                    # 6. GPU 가속 다운샘플링
                    part_t = part_t.voxel_down_sample(voxel_size)
                    
                    # 7. CPU로 회수하여 리스트에 저장
                    part_t_cpu = part_t.to(device_cpu)
                    accumulated_parts_cpu.append(part_t_cpu)
                    
                    del pcd_part, part_t
                    
                except Exception as e:
                    print(f"    Error on {os.path.basename(part_path)}: {e}")
            
            gc.collect()
            o3d.core.cuda.release_cache()
            folder.processed = True

        # [FIXED] 5-4. CPU에서 최종 병합 및 저장 (수정된 부분)
        if accumulated_parts_cpu:
            print(f"  -> Merging {len(accumulated_parts_cpu)} parts on CPU...")
            
            try:
                # 리스트의 첫 번째 요소를 시작점으로 잡습니다.
                merged_pcd_t = accumulated_parts_cpu[0]
                
                # 나머지 요소들을 순차적으로 append 합니다.
                # (PointCloud.concatenate는 존재하지 않으므로 루프를 사용)
                for i in range(1, len(accumulated_parts_cpu)):
                    merged_pcd_t = merged_pcd_t.append(accumulated_parts_cpu[i])
                
                # 병합 후 최종 다운샘플링 (CPU 연산)
                merged_pcd_t = merged_pcd_t.voxel_down_sample(voxel_size)
                
                # 저장
                final_pcd = merged_pcd_t.to_legacy()
                o3d.io.write_point_cloud(save_path, final_pcd, write_ascii=False)
                print(f"  -> Saved: {save_path}")
                
                del merged_pcd_t, final_pcd
            except Exception as e:
                print(f"  -> Merge/Save Error: {e}")
        
        del accumulated_parts_cpu
        gc.collect()
        o3d.core.cuda.release_cache()
        print("")

    print("\nAll submaps processed and merged.")

if __name__ == "__main__":
    main()
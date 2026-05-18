import os
import pandas as pd
import natsort
import numpy as np

# =========================================================
# 설정 구간 (configs/paths.yaml data_root)
# =========================================================
from src.common import load_config
_cfg = load_config("default.yaml")
DATA_ROOT = _cfg.paths.data_root.rstrip("/")
MAX_DEPTH = 3  # 검색할 하위 폴더 깊이 (2~3 권장)

def find_and_check_datasets():
    print(f"[Start] '{DATA_ROOT}' 경로에서 데이터셋 검색 시작 (Depth: {MAX_DEPTH})...")
    
    found_datasets = []
    
    # 1. 깊이 제한을 둔 폴더 탐색 (os.walk 활용)
    base_depth = DATA_ROOT.rstrip(os.sep).count(os.sep)
    
    for root, dirs, files in os.walk(DATA_ROOT):
        # 현재 깊이 계산
        current_depth = root.rstrip(os.sep).count(os.sep) - base_depth
        
        # 설정한 깊이를 초과하면 더 이상 하위로 들어가지 않음
        if current_depth >= MAX_DEPTH:
            dirs[:] = []  # 하위 디렉토리 비우기 (탐색 중단)
            continue

        # [핵심] 여기가 '데이터 폴더'인지 확인하는 조건
        # 조건: 같은 폴더 안에 'navigation.csv' 파일이 있고, 'blackfly' 폴더가 있어야 함
        if 'navigation.csv' in files and 'blackfly' in dirs:
            found_datasets.append(root)

    print(f"[Info] 총 {len(found_datasets)}개의 유효한 데이터셋 폴더를 발견했습니다.\n")

    if not found_datasets:
        print("❌ 발견된 데이터셋이 없습니다. DATA_ROOT 경로나 폴더 구조를 확인해주세요.")
        return

    # 2. 발견된 각 폴더에 대해 진단 수행
    for idx, folder_path in enumerate(found_datasets):
        folder_name = os.path.basename(folder_path)
        print(f"[{idx+1}/{len(found_datasets)}] Checking: {folder_path}")

        nav_path = os.path.join(folder_path, "navigation.csv")
        image_dir = os.path.join(folder_path, "blackfly")
        
        # --- (1) 데이터 개수 비교 ---
        try:
            # CSV 읽기
            df = pd.read_csv(nav_path)
            df.columns = df.columns.str.strip() # 공백 제거
            gps_count = len(df)
            
            # 이미지 파일 리스트
            img_files = natsort.natsorted(os.listdir(image_dir))
            img_count = len(img_files)
            
            print(f"   📊 [Counts] GPS: {gps_count} rows | Images: {img_count} files")
            
            if img_count == 0:
                print("   ❌ [Error] 이미지가 없습니다.")
                continue

        except Exception as e:
            print(f"   ❌ [Error] 데이터 로드 실패: {e}")
            continue

        # --- (2) 타임스탬프 비교 (인덱스 매칭 문제 확인) ---
        try:
            # 2-1. GPS 타임스탬프 (컬럼명 자동 찾기)
            time_cols = [c for c in df.columns if 'time' in c.lower() or 'stamp' in c.lower()]
            if not time_cols:
                print("   ⚠️ [Skip] CSV에 시간 컬럼이 없습니다.")
                continue
            
            t_col = time_cols[0]
            gps_start = df[t_col].iloc[0]
            gps_end = df[t_col].iloc[-1]
            gps_duration = gps_end - gps_start
            
            # 2-2. 이미지 타임스탬프 (파일명 파싱)
            first_img_name = os.path.splitext(img_files[0])[0]
            last_img_name = os.path.splitext(img_files[-1])[0]
            
            # 파일명이 숫자인지 확인 (타임스탬프 가정)
            try:
                img_start = float(first_img_name)
                img_end = float(last_img_name)
                
                # 나노초 단위(19자리) 보정
                if img_start > 1e18: img_start /= 1e9; img_end /= 1e9
                if gps_start > 1e18: gps_start /= 1e9; gps_end /= 1e9
                
                img_duration = img_end - img_start
                
            except ValueError:
                print(f"   ⚠️ [Skip] 이미지 파일명이 숫자가 아닙니다 (예: {img_files[0]}).")
                print("       -> 인덱스 기반 파일명이라면 시간 동기화 비교 불가.")
                continue

            # 2-3. 비교 결과 출력
            diff = abs(gps_duration - img_duration)
            print(f"   ⏱️ [Time] GPS Duration: {gps_duration:.2f}s | Img Duration: {img_duration:.2f}s")
            print(f"   🔍 [Diff] 차이: {diff:.2f}s")
            
            # [진단 결론]
            if diff > 10.0:
                print("   🚨 [CRITICAL] 시간 길이가 다릅니다! (주파수 차이로 인한 맵 끊김 원인)")
                print("       -> 해결책: 인덱스(i)가 아니라 타임스탬프 기준으로 매칭해야 합니다.")
            elif gps_count > (img_count * 5): # GPS가 이미지보다 5배 이상 많은데 시간은 비슷하다? -> 주파수 차이 확실
                print("   ⚠️ [Warning] 시간은 비슷하지만 데이터 개수 차이가 큽니다.")
                print("       -> 현재 코드는 인덱스(loc[i])를 쓰면 앞부분만 맵핑하고 끝납니다.")
            else:
                print("   ✅ [OK] 시간 및 데이터 비율이 정상 범위입니다.")

        except Exception as e:
            print(f"   ❌ [Error] 타임스탬프 비교 중 에러: {e}")

        print("-" * 50)

if __name__ == "__main__":
    find_and_check_datasets()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 설정 (configs/trajectory.yaml > single_trajectory + projection.input_folder)
# ==========================================
import argparse
import os
from src.common import load_config

_parser = argparse.ArgumentParser(description="Step0: Trajectory plot from a single CSV")
_parser.add_argument("--config", default="default.yaml")
_parser.add_argument("--csv", default=None, help="navigation.csv 경로 (없으면 projection.input_folder/navigation.csv)")
_args, _ = _parser.parse_known_args()
_cfg = load_config(_args.config)
file_path = _args.csv or os.path.join(_cfg.projection.input_folder, "navigation.csv")
arrow_step = _cfg.trajectory.single_trajectory.arrow_step

# ==========================================
# 2. CSV 파일 읽기
# ==========================================
try:
    # 쉼표로 구분된 CSV 읽기
    df = pd.read_csv(file_path)
    print(f"파일을 성공적으로 읽었습니다. 총 데이터 개수: {len(df)}개")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# ==========================================
# 3. 데이터 전처리
# ==========================================
# 위치 데이터 추출
pos_x = df['position_x'].values
pos_y = df['position_y'].values

# 로컬 좌표계로 변환 (첫 시작점을 0,0으로 이동)
# * 이유: UTM 좌표는 숫자가 너무 커서 그래프 축이 보기 불편해집니다.
start_x = pos_x[0]
start_y = pos_y[0]
local_x = pos_x - start_x
local_y = pos_y - start_y

# 쿼터니언 추출 (x, y, z, w 순서) 및 Yaw(헤딩) 계산
quaternions = df[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].values

# Scipy를 이용해 쿼터니언 -> 오일러 각 변환
r = R.from_quat(quaternions)
yaw = r.as_euler('xyz', degrees=False)[:, 2]  # Z축 회전(Yaw)만 추출

# 화살표(Quiver) 방향 벡터 계산 (u: x성분, v: y성분)
u = np.cos(yaw)
v = np.sin(yaw)

# ==========================================
# 4. 시각화 (Plotting)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

# (1) 전체 이동 경로 그리기 (파란 선)
ax.plot(local_x, local_y, label='Path', color='blue', linewidth=2, alpha=0.6)

# (2) 로봇의 방향 화살표 그리기 (빨간 화살표)
# 모든 점에 그리면 징그러우므로 arrow_step 간격으로 그립니다.
ax.quiver(local_x[::arrow_step], local_y[::arrow_step], 
          u[::arrow_step], v[::arrow_step], 
          color='red', scale=30, width=0.003, headwidth=4, label='Heading', alpha=0.8)

# (3) 시작점(초록)과 끝점(보라) 표시
ax.scatter(local_x[0], local_y[0], color='green', s=150, label='Start', marker='*', zorder=5)
ax.scatter(local_x[-1], local_y[-1], color='purple', s=150, label='End', marker='X', zorder=5)

# (4) 그래프 스타일 설정
ax.set_title(f'Trajectory Analysis: {file_path}', fontsize=14)
ax.set_xlabel(f'X Distance [m] (Origin Offset: {start_x:.1f})')
ax.set_ylabel(f'Y Distance [m] (Origin Offset: {start_y:.1f})')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
ax.axis('equal')  # ★ 중요: 지도 비율을 1:1로 유지

plt.tight_layout()
plt.show()
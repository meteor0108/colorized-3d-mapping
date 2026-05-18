import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import natsort
import math

def main():
    # ------------------------------------------------------------------
    # 0. 설정 (configs/trajectory.yaml > nav_reader)
    # ------------------------------------------------------------------
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step1: Nav reader (PNG/matplotlib)")
    parser.add_argument("--config", default="default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    root_path = cfg.paths.data_root
    output_image_file = cfg.trajectory.nav_reader.png_output

    TARGET_X = cfg.trajectory.nav_reader.target_x
    TARGET_Y = cfg.trajectory.nav_reader.target_y
    RADIUS = cfg.trajectory.nav_reader.radius

    # ------------------------------------------------------------------
    # 1. 파일 탐색
    # ------------------------------------------------------------------
    print("Searching for folders...")
    depth3 = glob.glob(os.path.join(root_path, "*", "*", "20*"))
    depth2 = glob.glob(os.path.join(root_path, "*", "2024*"))
    target_folders = sorted(list(set(depth3 + depth2)))
    valid_folders = [f for f in target_folders if os.path.exists(os.path.join(f, "navigation.csv"))]
    valid_folders = natsort.natsorted(valid_folders)
    
    if not valid_folders:
        print("No navigation files found.")
        return

    # ------------------------------------------------------------------
    # 2. 데이터 필터링
    # ------------------------------------------------------------------
    print("\nProcessing data...")
    used_data_info = []
    all_xs_total = [] 
    all_ys_total = []

    for idx, folder_path in enumerate(valid_folders):
        folder_name = os.path.basename(folder_path)
        nav_path = os.path.join(folder_path, "navigation.csv")
        try:
            df = pd.read_csv(nav_path)
            xs = df['position_x'].values
            ys = df['position_y'].values
            
            dists = np.sqrt((xs - TARGET_X)**2 + (ys - TARGET_Y)**2)
            if np.any(dists <= RADIUS):
                used_data_info.append((folder_name, xs, ys))
                all_xs_total.extend(xs)
                all_ys_total.extend(ys)
        except:
            continue

    # ------------------------------------------------------------------
    # 3. 그래프 그리기 및 4방향 배치 로직
    # ------------------------------------------------------------------
    if len(used_data_info) > 0:
        count = len(used_data_info)
        print(f"\nPlotting {count} trajectories with 4-Way (Top/Bottom/Left/Right) Layout...")
        
        # 캔버스 크기 (데이터가 많으므로 크게 설정)
        fig, ax = plt.subplots(figsize=(60, 60))
        
        colors = plt.cm.jet(np.linspace(0, 1, count))
        endpoints = [] 

        # 전체 경로 그리기 및 끝점 수집
        for idx, (folder_name, xs, ys) in enumerate(used_data_info):
            label_num = idx + 1
            
            ax.plot(xs, ys, color=colors[idx], linewidth=2.5, alpha=0.6)
            ax.scatter(xs[0], ys[0], marker='o', s=80, color=colors[idx], edgecolors='white', zorder=5) # Start
            ax.scatter(xs[-1], ys[-1], marker='X', s=150, color=colors[idx], edgecolors='black', zorder=6) # End
            
            endpoints.append({
                'idx': idx,
                'label': str(label_num),
                'x': xs[-1],
                'y': ys[-1],
                'color': colors[idx],
                'name': folder_name
            })

        ax.scatter(TARGET_X, TARGET_Y, s=1000, c='red', marker='P', zorder=10, label='Target Center')

        # --------------------------------------------------------------
        # [4방향 분할 로직 시작]
        # --------------------------------------------------------------
        min_x, max_x = min(all_xs_total), max(all_xs_total)
        min_y, max_y = min(all_ys_total), max(all_ys_total)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 데이터의 중심점 계산 (Centroid)
        center_x = np.mean([pt['x'] for pt in endpoints])
        center_y = np.mean([pt['y'] for pt in endpoints])

        # 각 끝점이 중심에서 어느 방향에 있는지 각도로 판단
        # -45 ~ 45: Right, 45 ~ 135: Top, 135 ~ -135(225): Left, -135 ~ -45: Bottom
        sectors = {'right': [], 'top': [], 'left': [], 'bottom': []}
        
        for pt in endpoints:
            dx = pt['x'] - center_x
            dy = pt['y'] - center_y
            angle = math.degrees(math.atan2(dy, dx))
            
            if -45 <= angle < 45:
                sectors['right'].append(pt)
            elif 45 <= angle < 135:
                sectors['top'].append(pt)
            elif -135 <= angle < -45:
                sectors['bottom'].append(pt)
            else:
                sectors['left'].append(pt)

        print(f" -> Distribution: Left({len(sectors['left'])}), Right({len(sectors['right'])}), Top({len(sectors['top'])}), Bottom({len(sectors['bottom'])})")

        positions = []
        
        # --- 레이아웃 설정 파라미터 ---
        margin_factor = 0.05
        text_cols_per_side = 5  # 각 방향별 층(Layer) 개수
        
        # 각 방향별 배치 함수
        def layout_sector(pts, side):
            if not pts: return
            
            # 정렬 기준 설정 (상하는 X축 기준, 좌우는 Y축 기준 정렬)
            if side in ['top', 'bottom']:
                pts_sorted = sorted(pts, key=lambda p: p['x'])
                primary_axis_min, primary_axis_max = min_x, max_x
                secondary_axis_base = max_y if side == 'top' else min_y
                is_vertical_stack = True # 텍스트가 위/아래로 쌓임
            else: # left, right
                pts_sorted = sorted(pts, key=lambda p: p['y'])
                primary_axis_min, primary_axis_max = min_y, max_y
                secondary_axis_base = min_x if side == 'left' else max_x
                is_vertical_stack = False # 텍스트가 좌/우로 쌓임

            count = len(pts_sorted)
            cols = text_cols_per_side
            
            # 배치 영역 너비/높이 계산
            layout_span = primary_axis_max - primary_axis_min
            step = layout_span / (count + 1) if count > 0 else layout_span
            
            # 층(Row/Col) 간격 설정
            layer_depth = (height if is_vertical_stack else width) * 0.03

            for i, pt in enumerate(pts_sorted):
                # 1. 주축(Primary Axis) 위치 균등 분배
                # Top/Bottom이면 X좌표를 균등하게, Left/Right면 Y좌표를 균등하게
                pos_on_primary = primary_axis_min + step * (i + 1)
                
                # 2. 보조축(Secondary Axis) 위치 (Layering)
                # 안쪽에서 바깥쪽으로 층을 쌓음
                layer_idx = i % cols
                offset = (layer_idx + 1) * layer_depth
                
                if side == 'top':
                    text_x = pos_on_primary
                    text_y = secondary_axis_base + offset + (height * margin_factor)
                    rad = 0.0 # 화살표 곡률 조절 가능
                elif side == 'bottom':
                    text_x = pos_on_primary
                    text_y = secondary_axis_base - offset - (height * margin_factor)
                    rad = 0.0
                elif side == 'right':
                    text_x = secondary_axis_base + offset + (width * margin_factor)
                    text_y = pos_on_primary
                    rad = 0.2
                else: # left
                    text_x = secondary_axis_base - offset - (width * margin_factor)
                    text_y = pos_on_primary
                    rad = -0.2

                positions.append({
                    'label': pt['label'],
                    'endpoint_x': pt['x'],
                    'endpoint_y': pt['y'],
                    'text_x': text_x,
                    'text_y': text_y,
                    'color': pt['color'],
                    'name': pt['name'],
                    'rad': rad
                })

        # 4방향 각각 배치 실행
        layout_sector(sectors['left'], 'left')
        layout_sector(sectors['right'], 'right')
        layout_sector(sectors['top'], 'top')
        layout_sector(sectors['bottom'], 'bottom')

        # --------------------------------------------------------------
        # 그리기
        # --------------------------------------------------------------
        for pos in positions:
            ax.annotate(
                text=pos['label'],
                xy=(pos['endpoint_x'], pos['endpoint_y']),
                xytext=(pos['text_x'], pos['text_y']),
                xycoords='data',
                textcoords='data',
                fontsize=18,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle="circle,pad=0.3", fc=pos['color'], ec="black", alpha=1.0),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle=f"arc3,rad={pos['rad']}",
                    color=pos['color'],
                    linewidth=1.0,
                    alpha=0.6
                ),
                zorder=7
            )

        # 뷰 범위 자동 조정 (텍스트 포함하도록)
        all_tx = [p['text_x'] for p in positions] + all_xs_total
        all_ty = [p['text_y'] for p in positions] + all_ys_total
        
        pad_x = (max(all_tx) - min(all_tx)) * 0.05
        pad_y = (max(all_ty) - min(all_ty)) * 0.05
        
        ax.set_xlim(min(all_tx) - pad_x, max(all_tx) + pad_x)
        ax.set_ylim(min(all_ty) - pad_y, max(all_ty) + pad_y)

        ax.set_title(f"Trajectory Map ({count} Routes - 4-Way Layout)", fontsize=50)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 범례 (너무 많으면 나누거나 생략 고려)
        from matplotlib.lines import Line2D
        # 범례는 너무 많으므로 4열로 표기
        legend_elements = [Line2D([0], [0], color=pos['color'], lw=4, label=f"[{pos['label']}]") 
                          for pos in positions]
        legend_elements.sort(key=lambda x: int(x.get_label().split(']')[0].strip('[')))
        
        # 범례가 200개면 화면을 가리므로 별도 이미지를 만들거나 축소해야 함. 
        # 여기서는 하단 외부에 배치
        ax.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.02), ncol=10, fontsize=10, title="ID Map")

        print(f"Saving image to '{output_image_file}'...")
        plt.savefig(output_image_file, dpi=150, bbox_inches='tight')
        print(" -> Done.")
        
    else:
        print("No data found.")

if __name__ == "__main__":
    main()
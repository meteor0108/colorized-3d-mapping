import os
import glob
import pandas as pd
import numpy as np
import natsort
import plotly.graph_objects as go
import plotly.io as pio

def main():
    # ------------------------------------------------------------------
    # 0. 설정 (configs/trajectory.yaml > nav_reader)
    # ------------------------------------------------------------------
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step1: Nav reader (HTML/plotly)")
    parser.add_argument("--config", default="default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    root_path = cfg.paths.data_root
    output_html_file = cfg.trajectory.nav_reader.html_output

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
    # 2. 데이터 로딩 및 필터링
    # ------------------------------------------------------------------
    print("\nProcessing data...")
    trajectories = [] # (folder_name, xs, ys) 저장

    # 시작점과 끝점을 별도 리스트로 모아서 한 번에 그리기 (성능 최적화 및 가시성)
    start_points_x = []
    start_points_y = []
    start_points_text = []

    end_points_x = []
    end_points_y = []
    end_points_text = []

    for idx, folder_path in enumerate(valid_folders):
        folder_name = os.path.basename(folder_path)
        nav_path = os.path.join(folder_path, "navigation.csv")
        try:
            df = pd.read_csv(nav_path)
            xs = df['position_x'].values
            ys = df['position_y'].values
            
            # 타겟 반경 내 데이터가 하나라도 있으면 유효
            dists = np.sqrt((xs - TARGET_X)**2 + (ys - TARGET_Y)**2)
            if np.any(dists <= RADIUS):
                label = f"[{idx+1}] {folder_name}"
                trajectories.append({
                    'xs': xs,
                    'ys': ys,
                    'label': label
                })
                
                # 시작점/끝점 정보 수집
                start_points_x.append(xs[0])
                start_points_y.append(ys[0])
                start_points_text.append(label)
                
                end_points_x.append(xs[-1])
                end_points_y.append(ys[-1])
                end_points_text.append(label)

        except Exception as e:
            print(f"Error reading {folder_name}: {e}")
            continue

    # ------------------------------------------------------------------
    # 3. Plotly로 그리기
    # ------------------------------------------------------------------
    count = len(trajectories)
    if count > 0:
        print(f"\nGenerarting Interactive Plot for {count} trajectories...")
        
        fig = go.Figure()

        # (1) 타겟 중심점 추가
        fig.add_trace(go.Scattergl(
            x=[TARGET_X], y=[TARGET_Y],
            mode='markers',
            marker=dict(size=20, color='red', symbol='cross', line=dict(width=2, color='black')),
            name='Target Center',
            hoverinfo='name+x+y'
        ))

        # (2) 개별 궤적 그리기 (Scattergl 사용: 대용량 데이터 렌더링 최적화)
        # 색상은 Plotly가 자동으로 할당하거나, 필요시 colormap 적용 가능
        for i, traj in enumerate(trajectories):
            fig.add_trace(go.Scattergl(
                x=traj['xs'],
                y=traj['ys'],
                mode='lines',
                name=traj['label'],       # 범례에 표시될 이름
                line=dict(width=2),
                opacity=0.7,
                hovertemplate=f"<b>{traj['label']}</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>"
            ))

        # (3) 시작점 모음 (초록색 원)
        fig.add_trace(go.Scattergl(
            x=start_points_x,
            y=start_points_y,
            mode='markers',
            marker=dict(size=8, color='green', symbol='circle', line=dict(width=1, color='white')),
            name='Start Points',
            text=start_points_text,
            hovertemplate="<b>Start: %{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
        ))

        # (4) 끝점 모음 (검은색 X)
        fig.add_trace(go.Scattergl(
            x=end_points_x,
            y=end_points_y,
            mode='markers',
            marker=dict(size=10, color='black', symbol='x-thin', line=dict(width=1)),
            name='End Points',
            text=end_points_text,
            hovertemplate="<b>End: %{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
        ))

        # 레이아웃 설정
        fig.update_layout(
            title=f"Trajectory Interactive Map (Total: {count})",
            title_font_size=24,
            width=1400,   # 브라우저 창 크기에 따라 자동 조절되지만 기본값 설정
            height=1000,
            showlegend=True,
            hovermode="closest", # 마우스 근처의 데이터 정보 표시
            plot_bgcolor='white',
            # X, Y축 비율 1:1 고정 (지도 왜곡 방지)
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                gridcolor='lightgray'
            ),
            xaxis=dict(
                gridcolor='lightgray'
            )
        )

        print(f"Saving interactive HTML to '{output_html_file}'...")
        # HTML 파일로 저장
        fig.write_html(output_html_file)
        print(" -> Done! Open the HTML file in your browser.")
        
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()
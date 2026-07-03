"""Native-rate rosbag 추출 (ROS 독립, rosbags 사용).

기존 step_pre1은 카메라(1Hz)에 sync해 LiDAR도 1Hz로 깎였음 → mesh 밀도 병목.
이 스크립트는 **LiDAR/카메라/nav를 native rate 전량** 추출한다 (IR a65, ouster 보조이미지 제외).

추출 토픽:
  - /ouster{1,2,3}/points  (PointCloud2, 10Hz)  → ouster{1,2,3}/points/<ts>.pcd
  - /blackfly/image_raw/compressed (bayer_rggb8 png, 10Hz) → blackfly/<ts>.png
  - /novatel/oem7/odom (Odometry, 50Hz) → navigation.csv

파일명 = 메시지 header stamp epoch(float) → TimestampParser.parse(float) 로 정확 매칭.
실행: kctc_gs env (rosbags + open3d + cv2 필요)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from rosbags.highlevel import AnyReader

LIDAR_TOPICS = ["/ouster1/points", "/ouster2/points", "/ouster3/points"]
IMAGE_TOPIC = "/blackfly/image_raw/compressed"
ODOM_TOPIC = "/novatel/oem7/odom"


def _stamp_sec(msg) -> float:
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def _pc2_to_xyz(msg) -> np.ndarray:
    """PointCloud2 → Nx3 float32 (NaN/zero 제거). x,y,z offset은 fields에서."""
    off = {f.name: f.offset for f in msg.fields}
    step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, step)
    def col(name):
        o = off[name]
        return raw[:, o:o + 4].copy().view(np.float32).ravel()
    pts = np.stack([col("x"), col("y"), col("z")], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-3)
    return pts[valid]


def _decode_bayer(msg) -> np.ndarray:
    """bayer_rggb8 png compressed → BGR (legacy step3와 동일: COLOR_BayerBG2BGR)."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    return img


def extract(bag: str, out: str, max_msgs: int | None = None):
    out = Path(out)
    dirs = {t: out / f"ouster{i+1}" / "points" for i, t in enumerate(LIDAR_TOPICS)}
    img_dir = out / "blackfly"
    for d in list(dirs.values()) + [img_dir]:
        d.mkdir(parents=True, exist_ok=True)

    want = set(LIDAR_TOPICS) | {IMAGE_TOPIC, ODOM_TOPIC}
    counts = {t: 0 for t in want}
    nav_rows = []
    n = 0

    with AnyReader([Path(bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in want]
        total = sum(c.msgcount for c in conns)
        print(f"[native] bag={Path(bag).name}  대상 메시지={total}  토픽={sorted(want)}", flush=True)
        for conn, _t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            ts = _stamp_sec(msg)
            name = f"{ts:.6f}"

            if conn.topic in dirs:
                pts = _pc2_to_xyz(msg)
                if len(pts):
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                    o3d.io.write_point_cloud(str(dirs[conn.topic] / f"{name}.pcd"), pc,
                                             write_ascii=False, compressed=True)
                    counts[conn.topic] += 1
            elif conn.topic == IMAGE_TOPIC:
                img = _decode_bayer(msg)
                if img is not None:
                    cv2.imwrite(str(img_dir / f"{name}.png"), img)
                    counts[conn.topic] += 1
            elif conn.topic == ODOM_TOPIC:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                tw = msg.twist.twist.linear
                nav_rows.append({
                    "unixtime": ts,
                    "position_x": p.x, "position_y": p.y, "position_z": p.z,
                    "orientation_x": q.x, "orientation_y": q.y,
                    "orientation_z": q.z, "orientation_w": q.w,
                    "twist_linear_x": tw.x, "twist_linear_y": tw.y, "twist_linear_z": tw.z,
                })
                counts[conn.topic] += 1

            n += 1
            if n % 2000 == 0:
                print(f"  진행 {n}/{total}  " +
                      " ".join(f"{k.split('/')[-2] if k.endswith('points') else k.split('/')[-1]}={v}"
                               for k, v in counts.items()), flush=True)
            if max_msgs and n >= max_msgs:
                print("  [test] max_msgs 도달, 중단", flush=True)
                break

    if nav_rows:
        df = pd.DataFrame(nav_rows).sort_values("unixtime")
        df.to_csv(out / "navigation.csv", index=False)

    print("\n[native] 완료:", flush=True)
    for t in LIDAR_TOPICS:
        print(f"  {t}: {counts[t]} pcd", flush=True)
    print(f"  {IMAGE_TOPIC}: {counts[IMAGE_TOPIC]} png", flush=True)
    print(f"  navigation.csv: {len(nav_rows)} rows", flush=True)
    print(f"  → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Native-rate rosbag 추출 (lidar/camera/nav)")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-msgs", type=int, default=None, help="테스트용 메시지 수 제한")
    args = ap.parse_args()
    extract(args.bag, args.out, args.max_msgs)


if __name__ == "__main__":
    main()

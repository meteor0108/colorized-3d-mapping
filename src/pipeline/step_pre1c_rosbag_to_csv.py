"""step_pre1c: 여러 rosbag에서 특정 토픽을 CSV로 내보내기 (rostopic echo).

configs/extraction.yaml > extraction.csv_export 사용.
"""
import argparse
import os

from src.common import load_config


def main():
    parser = argparse.ArgumentParser(description="step_pre1c: rosbag → CSV via rostopic echo")
    parser.add_argument("--config", default="default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config).extraction.csv_export

    topic = cfg.topic
    output_dir = cfg.output_dir
    rosbag_name_list = list(cfg.rosbag_list)

    os.makedirs(output_dir, exist_ok=True)

    for rosbag_name in rosbag_name_list:
        out = os.path.join(output_dir, rosbag_name.split(".")[0] + ".csv")
        command = f"rostopic echo -b {rosbag_name} -p {topic} > {out}"
        print(command)
        os.system(command)


if __name__ == "__main__":
    main()

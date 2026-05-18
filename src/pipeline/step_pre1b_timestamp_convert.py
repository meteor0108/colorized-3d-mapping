
# ros odometry
# https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html
# BESTPOS
# https://docs.novatel.com/OEM7/Content/Logs/BESTPOS.htm
# INSPVA
# https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSPVA.htm
# https://docs.ros.org/en/noetic/api/novatel_oem7_driver/html/bestpos__handler_8cpp_source.html

import rosbag
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def main():
    """rosbag을 header timestamp 기준으로 reindex (msg.header.stamp 사용)."""
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="step_pre1b: timestamp reindex")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--bag", default=None, help="rosbag 파일 (없으면 extraction.rosbag_path)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    rosbag_file = args.bag or cfg.extraction.rosbag_path
    if not rosbag_file:
        rosbag_file = input("Put rosbag file or folder : ").split("'")[1]
    rosbag_file_name = rosbag_file.split('.')[0]
    rosbag_file_reindex_name = rosbag_file_name + '_reindex.bag'
    with rosbag.Bag(rosbag_file_reindex_name, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(rosbag_file).read_messages():
            try:
                outbag.write(topic, msg, msg.header.stamp)
                print(topic)
                print(msg.header.stamp, t)
            except :
                outbag.write(topic, msg, t)

    print("\n\nEND")
if __name__ == "__main__":
	main()



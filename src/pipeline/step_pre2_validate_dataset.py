
import rosbag

import os
import time

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import time


def omission_file_check(a65_image_list, data_count):
    print(len(a65_image_list))
    file_name_list = []
    for file_dir in a65_image_list:
        file_name = file_dir.split('/')[-1].split('.')[0]
        file_name_list.append(file_name)
        # print(file_name)
    for i in range(data_count):
        number_name = str(i+1).zfill(5)
        if number_name not in file_name_list:
            print(number_name)

def main():
    """추출된 데이터셋(폴더)에서 누락 파일 / sync 검증.

    configs/extraction.yaml > validation.data_root + extraction.topics 사용.
    """
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="step_pre2: dataset validation")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--folder", default=None, help="검증할 데이터 폴더 (없으면 validation.data_root)")
    args = parser.parse_args()
    _cfg = load_config(args.config)

    lidar_image_topic = [
        "/ouster1/nearir_image",
        "/ouster1/range_image",
        "/ouster1/reflec_image",
        "/ouster1/signal_image",
        "/ouster2/nearir_image",
        "/ouster2/range_image",
        "/ouster2/reflec_image",
        "/ouster2/signal_image",
        "/ouster3/nearir_image",
        "/ouster3/range_image",
        "/ouster3/reflec_image",
        "/ouster3/signal_image"
    ]
    raw_image_topic = [
        "/a65/image_raw",
        "/ouster1/nearir_image",
        "/ouster1/range_image",
        "/ouster1/reflec_image",
        "/ouster1/signal_image",
        "/ouster2/nearir_image",
        "/ouster2/range_image",
        "/ouster2/reflec_image",
        "/ouster2/signal_image",
        "/ouster3/nearir_image",
        "/ouster3/range_image",
        "/ouster3/reflec_image",
        "/ouster3/signal_image"
    ]
    # config에서 topic 목록 override (있으면 우선)
    raw_image_topic = list(_cfg.extraction.topics.raw_image)
    compressed_image_topic = list(_cfg.extraction.topics.compressed_image)
    pointcloud_topic = list(_cfg.extraction.topics.pointcloud)
    gps_topic = list(_cfg.extraction.topics.gps)
    lidar_image_topic = [t for t in raw_image_topic if "ouster" in t]

    topic_name_list = raw_image_topic + compressed_image_topic + pointcloud_topic + gps_topic

    bag_folder_dir = args.folder or _cfg.validation.data_root
    if not bag_folder_dir:
        raise ValueError("validation.data_root가 비어있고 --folder도 지정되지 않았습니다.")
    bag_file_dir = bag_folder_dir+'.bag'
    bag_file_name = bag_folder_dir.split("/")[-1].split(".")[0]
    bag_timestamp_csv_file_dir = bag_folder_dir+'/log/'+bag_file_name+'_timestamp.csv'
    sync_index_csv_file_dir = bag_folder_dir+'/log/'+bag_file_name+'_sync_index.csv'
    sync_timestamp_csv_file_dir = bag_folder_dir+'/log/'+bag_file_name+'_sync_timestamp.csv'
    sync_timeoffset_csv_file_dir = bag_folder_dir+'/log/'+bag_file_name+'_sync_timeoffset.csv'
    navigation_csv_file_dir = bag_folder_dir+'/navigation.csv'

    a65_folder = bag_folder_dir + "/a65/"
    blackfly_folder = bag_folder_dir + "/blackfly/"
    lidar_image_folder_list = []
    for lidar_image_folder_name in lidar_image_topic:
        lidar_image_folder_list.append(bag_folder_dir+lidar_image_folder_name+'/')
    lidar_pcd_folder_list = []
    for lidar_pcd_folder_name in pointcloud_topic:
        lidar_pcd_folder_list.append(bag_folder_dir+lidar_pcd_folder_name+'/')

    df = pd.read_csv(bag_timestamp_csv_file_dir)
    df_sync_index = pd.read_csv(sync_index_csv_file_dir)
    df_sync_timeoffset = pd.read_csv(sync_timeoffset_csv_file_dir)
    df_sync_timestamp = pd.read_csv(sync_timestamp_csv_file_dir)
    df_nav_data = pd.read_csv(navigation_csv_file_dir)

    data_count = df_sync_index.shape[0]
    # print(df_sync_index)
    print(df_sync_index.shape[0])
    print(df_sync_timeoffset.shape[0])
    print(df_sync_timestamp.shape[0])
    print(df_nav_data.shape[0])
    print("-----------------")
    

    a65_image_list = glob.glob(a65_folder+'/*.png')
    blackfly_image_list = glob.glob(blackfly_folder+'/*.png')
    print(len(a65_image_list))
    print(len(blackfly_image_list))
    print("---")
    lidar_image_list = []
    for lidar_image_folder_name in lidar_image_folder_list:
        lidar_image_file_list = glob.glob(lidar_image_folder_name+'/*.png')
        lidar_image_list.append(lidar_image_file_list)
        folder_name = lidar_image_folder_name.split('/')[-3]+'-'+lidar_image_folder_name.split('/')[-2]
        print(folder_name, len(lidar_image_file_list))
    print("---")
    lidar_pcd_list = []
    for lidar_pcd_folder_name in lidar_pcd_folder_list:
        lidar_pcd_file_list = glob.glob(lidar_pcd_folder_name+'/*.pcd')
        lidar_pcd_list.append(lidar_pcd_file_list)
        print(len(lidar_pcd_file_list))
    print("---")
    print("---")


    standard_topic = "/blackfly/image_raw/compressed"
    # 기준 토픽에 대한 index 
    df_color_cam = df[df['topic'] == standard_topic]
    df_color_cam_time_list = df_color_cam.rosbag_time.to_list()
    # Start_Time_list = np.array(df_color_cam_time_list) - 0.1
    # End_Time_list = np.array(df_color_cam_time_list) + 0.1
    Start_Time_list = np.array(df_color_cam_time_list) - 10**8
    End_Time_list = np.array(df_color_cam_time_list) +10**8
    # Start_Time_list = np.array(df_color_cam_time_list) - 6**7
    # End_Time_list = np.array(df_color_cam_time_list) + 6**7
    all_data_count = len(df_color_cam_time_list)

    sync_timestamp = {}
    sync_timeoffset = {}
    sync_index = {}
    for topic_name in topic_name_list:
        sync_timestamp[topic_name] = []
        sync_timeoffset[topic_name] = []
        sync_index[topic_name] = []
    print("")
    for sync_time_count in range(all_data_count):
        # print(f"sync_time_count : {str(sync_time_count+1).zfill(5)}\r",end='')
        ST = Start_Time_list[sync_time_count]
        ET = End_Time_list[sync_time_count]
        sync_index_list = []
        sync_timestamp_list = []
        sync_timeoffset_list = []

        break_check = False
        for topic_name in topic_name_list:
            if break_check == True:
                break

            # if topic_name == "/a65/image_raw":
            df_topic = df[df['topic'] == "/a65/image_raw"]
            df_section = df_topic[(df_topic['rosbag_time'] > ST) & (df_topic['rosbag_time'] < ET)]
            # calculate time offset
            timestamp_list = df_section['rosbag_time'].tolist()
            if len(timestamp_list) == 0:
                break_check = True
                break
            time_offset_list = np.abs(np.array(timestamp_list) - df_color_cam_time_list[sync_time_count])*0.1**9
            time_sync_index = np.argmin(time_offset_list)
            # sync_time                                           # 1713034379 000000000
            sync_timestamp_data = timestamp_list[time_sync_index] # 1710823594.094663922 0.1**9
            sync_timeoffset_data = time_offset_list[time_sync_index]
            sync_index_data = df_section['index'].iloc[time_sync_index]

            if len(sync_timestamp[topic_name]) > 1 and sync_timestamp_data == sync_timestamp[topic_name][-1]:
                    break_check = True
                    print("break***************")
                    break

            sync_timestamp_list.append(sync_timestamp_data)
            sync_timeoffset_list.append(sync_timeoffset_data)
            sync_index_list.append(sync_index_data)

            if topic_name == "/a65/image_raw" and len(sync_timestamp[topic_name]) > 1:
                index_num = len(sync_timestamp[topic_name])
                print(f"{index_num}   {sync_timestamp_data}  {sync_timeoffset_data}  {sync_timestamp[topic_name][-1]}")

            # try:
            #     df_topic = df[df['topic'] == topic_name]
            #     df_section = df_topic[(df_topic['rosbag_time'] > ST) & (df_topic['rosbag_time'] < ET)]
            #     # calculate time offset
            #     timestamp_list = df_section['rosbag_time'].tolist()
            #     time_offset_list = np.abs(np.array(timestamp_list) - df_color_cam_time_list[sync_time_count])*0.1**9
            #     time_sync_index = np.argmin(time_offset_list)
            #     # sync_time                                           # 1713034379 000000000
            #     sync_timestamp_data = timestamp_list[time_sync_index] # 1710823594.094663922 0.1**9
            #     sync_timeoffset_data = time_offset_list[time_sync_index]
            #     sync_index_data = df_section['index'].iloc[time_sync_index]
            #     sync_timestamp_list.append(sync_timestamp_data)
            #     sync_timeoffset_list.append(sync_timeoffset_data)
            #     sync_index_list.append(sync_index_data)
            # except :
            #     pass

        if len(sync_index_list) == len(topic_name_list) and \
            len(sync_timestamp_list) == len(topic_name_list) and \
            len(sync_timeoffset_list) == len(topic_name_list):
            for i, topic_name in enumerate(topic_name_list):
                sync_index[topic_name].append(sync_index_list[i])
                sync_timestamp[topic_name].append(sync_timestamp_list[i])
                sync_timeoffset[topic_name].append(sync_timeoffset_list[i])



### sync time ###
# 73 1710823164515952278
# 74 1710823164615667926
# 75 1710823164754414678
# 76 1710823164817075635
# 77 1710823164922288008
# 78 1710823165011564947
# 79 1710823165053218244
# 80 1710823166323061063
# 81 1710823166323061063
# 82 1710823166422719094
# 83 1710823166523587143
# 84 1710823166626197531
# 85 1710823166736074632

### rosbag time ###
# /a65/image_raw 1710823164615667926 00075.png 75
# /a65/image_raw 1710823164754414678 00076.png 76
# /a65/image_raw 1710823164817075635 00077.png 77
# /a65/image_raw 1710823164922288008 00078.png 78
# /a65/image_raw 1710823165011564947 00079.png 79
# /a65/image_raw 1710823165053218244 00080.png 80
# /a65/image_raw 1710823166323061063 00081.png 81
# ******** /a65/image_raw 1710823166368787068 81
# /a65/image_raw 1710823166422719094 00083.png 83
# /a65/image_raw 1710823166523587143 00084.png 84
# /a65/image_raw 1710823166626197531 00085.png 85
# /a65/image_raw 1710823166736074632 00086.png 86
# /a65/image_raw 1710823166830505029 00087.png 87
# /a65/image_raw 1710823166943044447 00088.png 88
# /a65/image_raw 1710823167040881503 00089.png 89
# /a65/image_raw 1710823167140801512 00090.png 90

if __name__ == "__main__":
	main()

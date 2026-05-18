
# ros odometry
# https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html
# BESTPOS
# https://docs.novatel.com/OEM7/Content/Logs/BESTPOS.htm
# INSPVA
# https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSPVA.htm
# https://docs.ros.org/en/noetic/api/novatel_oem7_driver/html/bestpos__handler_8cpp_source.html

import rosbag
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import sensor_msgs.point_cloud2 as pc2

import cv2
import pcl

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from cv_bridge.boost.cv_bridge_boost import cvtColor2
from cv_bridge import CvBridge
bridge = CvBridge()

def make_dir(bag_file, raw_image_topic, compressed_image_topic, pointcloud_topic):
    bag_file_name = bag_file.split("/")[-1].split(".")[0]
    upper_dir = bag_file.replace(bag_file_name+".bag","")

    ### 폴더 생성 ###
    # csv log file folder
    directory = upper_dir+bag_file_name + "/log/"
    os.makedirs(directory,exist_ok=True)

    raw_image_topic_dir = []
    compressed_image_topic_dir = []
    pointcloud_topic_dir = []
    for topic_name in raw_image_topic:
        if "/image_raw" in topic_name:
            directory = upper_dir+bag_file_name + "/" + topic_name.split("/")[1] + "/"
            raw_image_topic_dir.append(directory)
        else :
            directory = upper_dir+bag_file_name + topic_name + "/"
            raw_image_topic_dir.append(directory)
        print(directory)
        os.makedirs(directory,exist_ok=True)
    for topic_name in compressed_image_topic:
        if "/image_raw/compressed" in topic_name:
            directory = upper_dir+bag_file_name + "/" + topic_name.split("/")[1] + "/"
            compressed_image_topic_dir.append(directory)
        elif "/image_color/compressed" in topic_name:
            directory = upper_dir+bag_file_name + "/" + topic_name.split("/")[1] + "/"
            compressed_image_topic_dir.append(directory)
        elif "/image_mono/compressed" in topic_name:
            directory = upper_dir+bag_file_name + "/" + topic_name.split("/")[1] + "/"
            compressed_image_topic_dir.append(directory)
        else :
            directory = upper_dir+bag_file_name + topic_name + "/"
            compressed_image_topic_dir.append(directory)
        print(directory)
        os.makedirs(directory,exist_ok=True)
    for topic_name in pointcloud_topic:
        directory = upper_dir+bag_file_name + topic_name + "/"
        pointcloud_topic_dir.append(directory)
        print(directory)
        os.makedirs(directory,exist_ok=True)
    return raw_image_topic_dir, compressed_image_topic_dir, pointcloud_topic_dir

def rosbag_read(bag_file_dir):
    bag_file_name = bag_file_dir.split("/")[-1].split(".")[0]
    ### rosbag data read ###
    print(f"rosbag {bag_file_name} read start")
    bag = rosbag.Bag(bag_file_dir)
    # bag_start_time = bag.get_start_time()
    # bag_end_time = bag.get_end_time()
    # bag_info = bag.get_type_and_topic_info()
    # bag_duration = bag_end_time - bag_start_time
    print(f"rosbag {bag_file_name} read end")
    return bag

def timestamp_data_frame_make(bag_file_dir, bag_timestamp_csv_file_dir):
    bag_read_complete = False
    bag = None
    if os.path.exists(bag_timestamp_csv_file_dir) == False :
        bag = rosbag_read(bag_file_dir)
        bag_read_complete = True

        index_list = []
        topic_list = []
        rosbag_time_list = []
        header_time_list = []
        index = 0
        count = -1

        bag_start_time = bag.get_start_time()
        bag_end_time = bag.get_end_time()
        bag_duration = bag_end_time - bag_start_time
        for topic, msg, t in bag.read_messages():
            count += 1

            ### progress bar ###
            progress_number = int((t.to_sec()-bag_start_time)/bag_duration*50)
            progress = "timestamp log make progress : "
            for i in range(50):
                if i <= progress_number:
                    progress += "#"
                else :
                    progress += "_"
            print(progress+'\r',end='')

            header_time = msg.header.stamp.to_sec()
            
            index_list.append(index)
            topic_list.append(topic)
            # rosbag_time_list.append(t.to_sec())
            rosbag_time_list.append(t.to_nsec())
            header_time_list.append(header_time)
            index += 1

        data = {'index': index_list, 'topic': topic_list, 'rosbag_time': rosbag_time_list, 'header_time': header_time_list}
        df = pd.DataFrame(data)
        df.to_csv(bag_timestamp_csv_file_dir, index=False)
    else :
        df = pd.read_csv(bag_timestamp_csv_file_dir)
    return df, bag_read_complete, bag


def sync_index_timeoffset_extract(standard_topic, df, topic_name_list, sync_index_csv_file_dir, 
                                  sync_timestamp_csv_file_dir, sync_timeoffset_csv_file_dir, display=False):
    if os.path.exists(sync_index_csv_file_dir) and os.path.exists(sync_timestamp_csv_file_dir) and os.path.exists(sync_timeoffset_csv_file_dir):
        # sync_index = pd.read_csv(sync_index_csv_file_dir)
        # sync_timeoffset = pd.read_csv(sync_timeoffset_csv_file_dir)
        # sync_timestamp = pd.read_csv(sync_timestamp_csv_file_dir)
        df_sync_index = pd.read_csv(sync_index_csv_file_dir)
        df_sync_timeoffset = pd.read_csv(sync_timeoffset_csv_file_dir)
        df_sync_timestamp = pd.read_csv(sync_timestamp_csv_file_dir)
        sync_index = df_sync_index.to_dict()
        sync_timeoffset = df_sync_index.to_dict()
        sync_timestamp = df_sync_index.to_dict()
        # 기준 토픽에 대한 index 
        df_color_cam = df[df['topic'] == standard_topic]
        df_color_cam_time_list = df_color_cam.rosbag_time.to_list()
        # Start_Time_list = np.array(df_color_cam_time_list) - 0.1
        # End_Time_list = np.array(df_color_cam_time_list) + 0.1
        Start_Time_list = np.array(df_color_cam_time_list) - 10**8
        End_Time_list = np.array(df_color_cam_time_list) + 10**8
        all_data_count = len(df_color_cam_time_list)
    else:
        standard_topic = "/blackfly/image_raw/compressed"
        # 기준 토픽에 대한 index 
        df_color_cam = df[df['topic'] == standard_topic]
        df_color_cam_time_list = df_color_cam.rosbag_time.to_list()
        # Start_Time_list = np.array(df_color_cam_time_list) - 0.1
        # End_Time_list = np.array(df_color_cam_time_list) + 0.1
        Start_Time_list = np.array(df_color_cam_time_list) - 10**8
        End_Time_list = np.array(df_color_cam_time_list) +10**8
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
            print(f"sync_time_count : {str(sync_time_count+1).zfill(5)}\r",end='')
            ST = Start_Time_list[sync_time_count]
            ET = End_Time_list[sync_time_count]
            sync_index_list = []
            sync_timestamp_list = []
            sync_timeoffset_list = []
            for topic_name in topic_name_list:
                try:
                    df_topic = df[df['topic'] == topic_name]
                    df_section = df_topic[(df_topic['rosbag_time'] > ST) & (df_topic['rosbag_time'] < ET)]
                    # calculate time offset
                    timestamp_list = df_section['rosbag_time'].tolist()
                    time_offset_list = np.abs(np.array(timestamp_list) - df_color_cam_time_list[sync_time_count])*0.1**9
                    time_sync_index = np.argmin(time_offset_list)
                    # sync_time                                           # 1713034379 000000000
                    sync_timestamp_data = timestamp_list[time_sync_index] # 1710823594.094663922 0.1**9
                    sync_timeoffset_data = time_offset_list[time_sync_index]
                    sync_index_data = df_section['index'].iloc[time_sync_index]
                    sync_timestamp_list.append(sync_timestamp_data)
                    sync_timeoffset_list.append(sync_timeoffset_data)
                    sync_index_list.append(sync_index_data)
                except :
                    pass

            if len(sync_index_list) == len(topic_name_list) and \
                len(sync_timestamp_list) == len(topic_name_list) and \
                len(sync_timeoffset_list) == len(topic_name_list):
                for i, topic_name in enumerate(topic_name_list):
                    sync_index[topic_name].append(sync_index_list[i])
                    sync_timestamp[topic_name].append(sync_timestamp_list[i])
                    sync_timeoffset[topic_name].append(sync_timeoffset_list[i])

        df_sync_index = pd.DataFrame(sync_index)
        df_sync_timestamp = pd.DataFrame(sync_timestamp)
        df_sync_timeoffset = pd.DataFrame(sync_timeoffset)
        df_sync_index.to_csv(sync_index_csv_file_dir, index=False)
        df_sync_timestamp.to_csv(sync_timestamp_csv_file_dir, index=False)
        df_sync_timeoffset.to_csv(sync_timeoffset_csv_file_dir, index=False)

        df_sync_timestamp.astype(int)

    if display == True:
        print("==================")
        print(f"all data count : {all_data_count}")
        for topic_name in topic_name_list:
            print(f"{topic_name} : {df_sync_index[topic_name].shape[0]} | min:{df_sync_timeoffset[topic_name].min}  max:{df_sync_timeoffset[topic_name].max} ")

        print(df_sync_index)
        print(df_sync_timeoffset)
        
        # # 각 열의 평균과 표준편차 계산
        means = df_sync_timeoffset.mean()
        stds = df_sync_timeoffset.std()

        # # 막대 그래프 그리기
        plt.rc('xtick', labelsize=5)  # x축 눈금 폰트 크기 
        plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기

        plt.bar(df_sync_timeoffset.columns, means, yerr=stds, capsize=5)
        plt.xlabel('col')
        plt.ylabel('mean')
        plt.title('time sync')
        plt.show()

    return df_sync_index, df_sync_timestamp

def navigation_data_extract(navigation_csv_file_dir, bag_read_complete, bag, bag_file_dir, gps_topic, sync_timestamp):
    print("navigation data extract Start")
    if os.path.exists(navigation_csv_file_dir):
        bag_read_complete = False
        bag = None
    else :
        if bag_read_complete == False:
            bag = rosbag_read(bag_file_dir)
            bag_read_complete = True

        df_nav_data = {
            "unixtime":[],
            "datetime":[],
            "latitude":[],
            "longitude":[],
            "height":[],
            "north_velocity":[],
            "east_velocity":[],
            "up_velocity":[],
            "roll":[],
            "pitch":[],
            "azimuth":[],
            "status":[],

            "position_x":[],
            "position_y":[],
            "position_z":[],

            "orientation_x":[],
            "orientation_y":[],
            "orientation_z":[],
            "orientation_w":[],

            "twist_linear_x":[],
            "twist_linear_y":[],
            "twist_linear_z":[]
        }
        index_inspva_list = sync_timestamp["/novatel/oem7/inspva"].tolist()
        index_odom_list = sync_timestamp["/novatel/oem7/odom"].tolist()
        index_inspva_list = list(map(int, index_inspva_list))
        index_odom_list = list(map(int, index_odom_list))
        print(len(index_inspva_list))
        print(len(index_odom_list))
        count = -1

        bag_start_time = bag.get_start_time()
        bag_end_time = bag.get_end_time()
        bag_duration = bag_end_time - bag_start_time
        for topic, msg, t in bag.read_messages():
            count += 1

            ### progress bar ###
            progress_number = int((t.to_sec()-bag_start_time)/bag_duration*50)
            progress = "navigation data extract progress : "
            for i in range(50):
                if i <= progress_number:
                    progress += "#"
                else :
                    progress += "_"
            print(progress+'\r',end='')

            # if topic in gps_topic and count in sync_index[topic]:
            # print(f"{topic}  :  {count}")
            if topic == "/novatel/oem7/inspva" and t.to_nsec() in index_inspva_list:
                # print(f"{topic}  :  {count}")
                unix_time = t.to_sec()
                date_time = datetime.fromtimestamp(unix_time)

                latitude = msg.latitude
                longitude = msg.longitude
                height = msg.height

                north_velocity = msg.north_velocity
                east_velocity = msg.east_velocity
                up_velocity = msg.up_velocity

                roll = msg.roll
                pitch = msg.pitch
                azimuth = msg.azimuth

                status = ""
                msg_status = msg.status.status
                if msg_status == 0:
                    status = "INS_INACTIVE"
                elif msg_status == 1:
                    status = "INS_ALIGNING"
                elif msg_status == 2:
                    status = "INS_HIGH_VARIANCE"
                elif msg_status == 3:
                    status = "INS_SOLUTION_GOOD"
                elif msg_status == 4:
                    status = "INS_SOLUTION_FREE"
                elif msg_status == 5:
                    status = "INS_ALIGNMENT_COMPLETE"
                elif msg_status == 6:
                    status = "DETERMINING_ORIENTATION"
                elif msg_status == 7:
                    status = "WAITING_INITIALPOS"
                elif msg_status == 8:
                    status = "WAITING_AZIMUTH"
                elif msg_status == 9:
                    status = "INITIALIZING_BIASES"
                elif msg_status == 10:
                    status = "MOTION_DETECT"
                elif msg_status == 11:
                    status = "WAITING_ALIGNMENTORIENTATION"

                df_nav_data["unixtime"].append(unix_time)
                df_nav_data["datetime"].append(date_time)

                df_nav_data["latitude"].append(latitude)
                df_nav_data["longitude"].append(longitude)
                df_nav_data["height"].append(height)

                df_nav_data["north_velocity"].append(north_velocity)
                df_nav_data["east_velocity"].append(east_velocity)
                df_nav_data["up_velocity"].append(up_velocity)

                df_nav_data["roll"].append(roll)
                df_nav_data["pitch"].append(pitch)
                df_nav_data["azimuth"].append(azimuth)

                df_nav_data["status"].append(status)
            elif topic == "/novatel/oem7/odom" and t.to_nsec() in index_odom_list:
                # print(f"{topic}  :  {count}")
                position = msg.pose.pose.position
                orientation = msg.pose.pose.orientation
                linear = msg.twist.twist.linear
                df_nav_data["position_x"].append(position.x)
                df_nav_data["position_y"].append(position.y)
                df_nav_data["position_z"].append(position.z)

                df_nav_data["orientation_x"].append(orientation.x)
                df_nav_data["orientation_y"].append(orientation.y)
                df_nav_data["orientation_z"].append(orientation.z)
                df_nav_data["orientation_w"].append(orientation.w)

                df_nav_data["twist_linear_x"].append(linear.x)
                df_nav_data["twist_linear_y"].append(linear.y)
                df_nav_data["twist_linear_z"].append(linear.z)

        print("")
        print(len(df_nav_data["unixtime"]))
        print(len(df_nav_data["twist_linear_z"]))
        # exit()
        df_nav = pd.DataFrame(df_nav_data)
        # print(df_nav)
        df_nav.to_csv(navigation_csv_file_dir, index=False)
    print("navigation data extract End")
    return bag_read_complete, bag

def rosbag_extract(bag_file_dir, display=False):
    # 토픽 목록은 configs/extraction.yaml > extraction.topics 에서 로드
    from src.common import load_config
    _cfg = load_config("default.yaml")
    raw_image_topic = list(_cfg.extraction.topics.raw_image)
    compressed_image_topic = list(_cfg.extraction.topics.compressed_image)
    pointcloud_topic = list(_cfg.extraction.topics.pointcloud)
    gps_topic = list(_cfg.extraction.topics.gps)

    topic_name_list = raw_image_topic + compressed_image_topic + pointcloud_topic + gps_topic

    bag_file_name = bag_file_dir.split("/")[-1].split(".")[0]
    bag_timestamp_csv_file_dir = bag_file_dir.replace('.bag','/log/')+bag_file_name+'_timestamp.csv'
    sync_index_csv_file_dir = bag_file_dir.replace('.bag','/log/')+bag_file_name+'_sync_index.csv'
    sync_timestamp_csv_file_dir = bag_file_dir.replace('.bag','/log/')+bag_file_name+'_sync_timestamp.csv'
    sync_timeoffset_csv_file_dir = bag_file_dir.replace('.bag','/log/')+bag_file_name+'_sync_timeoffset.csv'
    navigation_csv_file_dir = bag_file_dir.replace('.bag','/')+'navigation.csv'

    ### 폴더 생성 ###
    raw_image_topic_dir, compressed_image_topic_dir, pointcloud_topic_dir = \
        make_dir(bag_file_dir, raw_image_topic, compressed_image_topic, pointcloud_topic)
    ### rosbag 타임스탬프 생성 ###
    df, bag_read_complete, bag = timestamp_data_frame_make(bag_file_dir, bag_timestamp_csv_file_dir)
    ### 기준 토픽에 따른 index, timeoffset 계산 ###
    sync_index, sync_timestamp = sync_index_timeoffset_extract("/blackfly/image_raw/compressed", df, topic_name_list, \
                                                         sync_index_csv_file_dir, sync_timestamp_csv_file_dir, sync_timeoffset_csv_file_dir, display=display)
    # exit()
    ### 항법 데이터 취득 ###
    bag_read_complete, bag = navigation_data_extract(navigation_csv_file_dir, bag_read_complete, bag, bag_file_dir, gps_topic, sync_timestamp)
    
    ### rosbag data read ###
    if bag_read_complete == False:
        bag = rosbag_read(bag_file_dir)

    # 타임스템프 int 형으로 바꿔서 저장
    # 판다스에서 형변환을 직접하는건 영향을 안미침...
    time_dict = {}
    for topic_name in topic_name_list:
        time_list = sync_timestamp[topic_name].tolist()
        time_list = list(map(int, time_list))
        time_dict[topic_name] = time_list

    count = -1
    bag_start_time = bag.get_start_time()
    bag_end_time = bag.get_end_time()
    bag_duration = bag_end_time - bag_start_time
    for topic, msg, t in bag.read_messages():
        count += 1

        # ### progress bar ###
        # progress_number = int((t.to_sec()-bag_start_time)/bag_duration*50)
        # progress = "Sensor data extract progress : "
        # for i in range(50):
        #     if i <= progress_number:
        #         progress += "#"
        #     else :
        #         progress += "_"
        # print(progress+'\r',end='')

        # if topic in compressed_image_topic and t.to_nsec() in time_dict[topic]:
        #     print(type(time_dict["/blackfly/image_raw/compressed"]))
        #     print(time_dict["/blackfly/image_raw/compressed"].index(t.to_nsec()))

        if topic in raw_image_topic and t.to_nsec() in time_dict[topic]:
            ### 취득 시간 이름 ###
            # file_name = raw_image_topic_dir[raw_image_topic.index(topic)] + str(t)+'.png'
            ### 싱크 번호 이름 ###
            # file_index = sync_index[topic].tolist().index(count)+1
            file_index = time_dict[topic].index(t.to_nsec())+1
            file_name = raw_image_topic_dir[raw_image_topic.index(topic)] + str(file_index).zfill(5)+'.png'
            if os.path.exists(file_name): print("pass..."); continue
            print(f"{topic} : {t} {msg.encoding}    {file_name}")
            raw_image = bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')
            if topic == "/a65/image_raw":
                # image_data_raw = raw_image.astype(np.uint16) * 0.04 - 273.15
                # aximg = plt.imshow(image_data_raw, cmap='inferno')
                # arr = aximg.make_image(renderer=None, unsampled=True)[0]
                image_data_raw = raw_image.astype(np.uint16)
                cv2.imwrite(file_name,image_data_raw)
            else :
                cv2.imwrite(file_name,raw_image)

        elif topic in compressed_image_topic and t.to_nsec() in time_dict[topic]:
        # if topic in compressed_image_topic and t.to_nsec() in time_list:
            ### 취득 시간 이름 ###
            # file_name = compressed_image_topic_dir[compressed_image_topic.index(topic)] + str(t)+'.png'
            ### 싱크 번호 이름 ###
            # file_index = sync_index[topic].tolist().index(count)+1
            file_index = time_dict[topic].index(t.to_nsec())+1
            file_name = compressed_image_topic_dir[compressed_image_topic.index(topic)] + str(file_index).zfill(5)+'.png'
            if os.path.exists(file_name): print("pass..."); continue
            print(f"{topic} : {t}   {file_name}")
            buf = np.frombuffer(msg.data, np.uint8)
            compressed_image = cvtColor2(cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR), 'bayer_rggb8', 'bgr8')
            cv2.imwrite(file_name,compressed_image)
            
        elif topic in pointcloud_topic and t.to_nsec() in time_dict[topic]:
            ### 취득 시간 이름 ###
            # file_name = pointcloud_topic_dir[pointcloud_topic.index(topic)] + str(t)+'.pcd'
            ### 싱크 번호 이름 ###
            # file_index = sync_index[topic].tolist().index(count)+1
            file_index = time_dict[topic].index(t.to_nsec())+1
            file_name = pointcloud_topic_dir[pointcloud_topic.index(topic)] + str(file_index).zfill(5)+'.pcd'
            if os.path.exists(file_name): print("pass..."); continue
            print(f"{topic} : {t}   {file_name}")
            ### 포인트 클라우드 데이터 추출 ###
            # pcd = list(pc2.read_points(msg, skip_nans=True, field_names=("x","y","z","intensity","ring","time")))
            pcd = list(pc2.read_points(msg, skip_nans=True, field_names=("x","y","z","intensity")))
            points=np.asanyarray(list(pcd))
            ### pcl 변환 ###
            pc = pcl.PointCloud_PointXYZI()
            pc.from_list(points[:,:4].tolist()) # XYZI
            pcl.save(pc, file_name)
    bag.close()


def main():
    """rosbag(.bag) 또는 rosbag을 담은 디렉토리에서 모든 센서 데이터 추출.

    configs/extraction.yaml > extraction.rosbag_path 또는 --bag 인자로 입력 지정.
    """
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="step_pre1: rosbag → folder extraction")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--bag", default=None, help="rosbag(.bag) 또는 디렉토리 (없으면 interactive)")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    rosbag_file = args.bag or cfg.extraction.rosbag_path
    if not rosbag_file:
        rosbag_file = input("Put rosbag file or folder : ").split("'")[1]
    if os.path.isdir(rosbag_file):
        # 모든 데이터 불러오기
        file_list = os.listdir(rosbag_file)
        for name in file_list:
            # 디렉토리 이름 거르기
            if os.path.isdir(rosbag_file+'/'+name):
                continue
            # 폴더 안에 메타데이터, imu 데이터 있으면, 분류 성공한 rosbag 파일이니 넘어가기
            if os.path.exists(rosbag_file+'/'+name.split('.')[0]+'/navigation.csv') :
                continue
            
            # 진행중인 파일 표시
            if os.path.exists(rosbag_file+'/thread/'+name.split('.')[0]+'.txt'):
                continue
            else :
                try:
                    os.makedirs(rosbag_file+'/thread')
                except:
                    pass
                f = open(rosbag_file+'/thread/'+name.split('.')[0]+'.txt', 'w')
                f.close()
            
            # 파일 분류 시작
            print(rosbag_file+'/'+name,"  Start")
            st = time.time()
            rosbag_extract(rosbag_file+"/"+name, display=args.display)
            print(rosbag_file+'/'+name,"  end.  Time Take : ",time.time()-st)
            
            # 진행중인 파일 표시 삭제
            os.system(f"rm {rosbag_file+'/thread/'+name.split('.')[0]+'.txt'}")
    else :
        print(rosbag_file,"  Start")
        rosbag_extract(rosbag_file, display=args.display)


    print("\n\nEND")
if __name__ == "__main__":
	main()


'''
topics:      /a65/camera_info                 1159 msgs    : sensor_msgs/CameraInfo       
             /a65/image_raw                   1159 msgs    : sensor_msgs/Image            
             /blackfly/camera_info             587 msgs    : sensor_msgs/CameraInfo       
             /blackfly/image_raw/compressed    587 msgs    : sensor_msgs/CompressedImage  
             
             /gps/fix                         2940 msgs    : sensor_msgs/NavSatFix        
             /gps/gps                         2940 msgs    : gps_common/GPSFix            
             /gps/imu                         5880 msgs    : sensor_msgs/Imu              
             /imu/data_raw                    5876 msgs    : sensor_msgs/Imu              
             /novatel/oem7/bestgnsspos         588 msgs    : novatel_oem7_msgs/BESTGNSSPOS
             /novatel/oem7/bestpos             588 msgs    : novatel_oem7_msgs/BESTPOS    
             /novatel/oem7/bestutm              58 msgs    : novatel_oem7_msgs/BESTUTM    
             /novatel/oem7/bestvel             588 msgs    : novatel_oem7_msgs/BESTVEL    
             /novatel/oem7/corrimu            5876 msgs    : novatel_oem7_msgs/CORRIMU    
             /novatel/oem7/driver/bond         236 msgs    : bond/Status                   (3 connections)
             /novatel/oem7/heading2             58 msgs    : novatel_oem7_msgs/HEADING2   
             /novatel/oem7/inspva             2938 msgs    : novatel_oem7_msgs/INSPVA     
             /novatel/oem7/inspvax              58 msgs    : novatel_oem7_msgs/INSPVAX    
             /novatel/oem7/insstdev             58 msgs    : novatel_oem7_msgs/INSSTDEV   
             /novatel/oem7/odom               2938 msgs    : nav_msgs/Odometry            
             /novatel/oem7/oem7raw            8099 msgs    : novatel_oem7_msgs/Oem7RawMsg 
             /novatel/oem7/ppppos               58 msgs    : novatel_oem7_msgs/PPPPOS     
             /novatel/oem7/rxstatus             64 msgs    : novatel_oem7_msgs/RXSTATUS   
             /novatel/oem7/time                 58 msgs    : novatel_oem7_msgs/TIME       

             /ouster1/nearir_image             588 msgs    : sensor_msgs/Image            
             /ouster1/points                   588 msgs    : sensor_msgs/PointCloud2      
             /ouster1/range_image              588 msgs    : sensor_msgs/Image            
             /ouster1/reflec_image             588 msgs    : sensor_msgs/Image            
             /ouster1/signal_image             588 msgs    : sensor_msgs/Image            
             /ouster2/nearir_image             588 msgs    : sensor_msgs/Image            
             /ouster2/points                   588 msgs    : sensor_msgs/PointCloud2      
             /ouster2/range_image              588 msgs    : sensor_msgs/Image            
             /ouster2/reflec_image             588 msgs    : sensor_msgs/Image            
             /ouster2/signal_image             588 msgs    : sensor_msgs/Image            
             /ouster3/nearir_image             588 msgs    : sensor_msgs/Image            
             /ouster3/points                   588 msgs    : sensor_msgs/PointCloud2      
             /ouster3/range_image              588 msgs    : sensor_msgs/Image            
             /ouster3/reflec_image             588 msgs    : sensor_msgs/Image            
             /ouster3/signal_image             588 msgs    : sensor_msgs/Image
'''


'''
Topic : /novatel/oem7/odom

header: 
  seq: 85105
  stamp: 
    secs: 1710823833
    nsecs: 794711045
  frame_id: "odom"
child_frame_id: "base_link"
pose: 
  pose: 
    position: 
      x: 427463.4818012593
      y: 4196858.672075394
      z: 667.704937690869
    orientation: 
      x: -0.021062913091783278
      y: 0.03667651355752763
      z: -0.9991051741682078
      w: -0.0001949247743751048
  covariance: [0.00013227050064969215, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00015292474183723273, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00027418022137435175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.655984407731534e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.868299696050466e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.019001152175857294]
twist: 
  twist: 
    linear: 
      x: 6.799817073355493
      y: 0.04149068924446089
      z: -0.06922269664372613
    angular: 
      x: 0.0
      y: 0.0
      z: 0.0
  covariance: [4.762786520509676e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.3195832548117465e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.8586260664293685e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
---
Topic : /novatel/oem7/inspva

header: 
  seq: 83903
  stamp: 
    secs: 1710823809
    nsecs: 795104801
  frame_id: "gps"
nov_header: 
  message_name: "INSPVA"
  message_id: 508
  message_type: 0
  sequence_number: 0
  time_status: 20
  gps_week_number: 2306
  gps_week_milliseconds: 190227780
latitude: 37.916007083585306
longitude: 128.17667404369587
height: 688.1915311058983
north_velocity: 2.0892051512295233
east_velocity: -9.355341927256175
up_velocity: 0.45859983449177055
roll: 0.11511686187802156
pitch: 3.3552213004688123
azimuth: 281.99631921209556
status: 
  status: 3
'''

'''
/a65/image_raw : 1710823588115131178 mono16    /media/ros/T7_Shield/add_rosbag/2024-03-19/2024-03-19-13-39-16_0/a65/04292.png
libpng error: Write Error
/ouster3/range_image : 1710823588191644259 mono16    /media/ros/T7_Shield/add_rosbag/2024-03-19/2024-03-19-13-39-16_0/ouster3/range_image/04293.png
libpng error: Write Error
/ouster3/signal_image : 1710823588191807951 mono16    /media/ros/T7_Shield/add_rosbag/2024-03-19/2024-03-19-13-39-16_0/ouster3/signal_image/04293.png
libpng error: Write Error
/ouster3/reflec_image : 1710823588192069598 mono16    /media/ros/T7_Shield/add_rosbag/2024-03-19/2024-03-19-13-39-16_0/ouster3/reflec_image/04293.png
libpng error: Write Error
'''
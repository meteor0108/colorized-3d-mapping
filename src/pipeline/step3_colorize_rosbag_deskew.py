
import os
import cv2
import numpy as np
import copy
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp 

from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

class PointCloudProcessor:
    """포인트클라우드 처리 관련 기능"""
    
    @staticmethod
    def range_filter(pcd, distance_threshold=30, query_point=[0, 0, 0]):
        points = np.asarray(pcd.points)
        distances = np.linalg.norm(points - query_point, axis=1)
        filtered_indices = np.where(distances <= distance_threshold)[0]
        return pcd.select_by_index(filtered_indices)
    
    @staticmethod
    def undistortion(x, y, distortion, r, fisheye=False):
        if not fisheye and distortion.shape[0] == 5:
            X_Radial_dist = x * (1 + distortion[0] * r**2 + distortion[1] * r**4 + distortion[4] * r**6)
            Y_Radial_dist = y * (1 + distortion[0] * r**2 + distortion[1] * r**4 + distortion[4] * r**6)
            X_Tangential_dist = 2 * distortion[2] * x * y + distortion[3] * (r**2 + 2 * x**2)
            Y_Tangential_dist = distortion[2] * (r**2 + 2 * y**2) + 2 * distortion[3] * x * y
            X_distortion = X_Radial_dist + X_Tangential_dist
            Y_distortion = Y_Radial_dist + Y_Tangential_dist
        elif not fisheye and distortion.shape[0] == 4:
            X_Radial_dist = x * (1 + distortion[0] * r**2 + distortion[1] * r**4)
            Y_Radial_dist = y * (1 + distortion[0] * r**2 + distortion[1] * r**4)
            X_Tangential_dist = 2 * distortion[2] * x * y + distortion[3] * (r**2 + 2 * x**2)
            Y_Tangential_dist = distortion[2] * (r**2 + 2 * y**2) + 2 * distortion[3] * x * y
            X_distortion = X_Radial_dist + X_Tangential_dist
            Y_distortion = Y_Radial_dist + Y_Tangential_dist
        return X_distortion, Y_distortion
    
    @staticmethod
    def project_to_image(pointcloud, image_, extrinsic, intrinsic, distortion, fisheye, 
                        point_size=3, alpha=0.1, max_range=15, mask_params=None, cutoff_y=None):
        
        cmap = plt.cm.get_cmap("jet", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        
        lidar_pts = np.asarray(pointcloud.points)
        image = copy.deepcopy(image_)
        height, width = image.shape[:2]
        length = np.linalg.norm(lidar_pts, 2, axis=1)
        
        homo = extrinsic @ np.vstack((lidar_pts.T, np.ones(lidar_pts.shape[0])))
        x = homo[0, :] / homo[2, :]
        y = homo[1, :] / homo[2, :]
        r = np.sqrt(x**2 + y**2)
        
        X_distortion, Y_distortion = PointCloudProcessor.undistortion(x, y, distortion, r, fisheye)
        pixel = intrinsic @ np.vstack((X_distortion, Y_distortion, np.ones(X_distortion.shape[0])))
        
        valid_mask = (pixel[0, :] > 0) & (pixel[0, :] < width) & \
                     (pixel[1, :] > 0) & (pixel[1, :] < height) & \
                     (homo[2, :] > 0)

        if mask_params is not None:
            mcx, mcy, mr = mask_params
            dist_from_center = np.sqrt((pixel[0, :] - mcx)**2 + (pixel[1, :] - mcy)**2)
            valid_mask &= (dist_from_center < mr)

        if cutoff_y is not None:
            valid_mask &= (pixel[1, :] < cutoff_y)

        index = np.where(valid_mask)
        if len(index[0]) == 0:
            return image, o3d.geometry.PointCloud()
        
        pixels = pixel.T[index, :2].astype(int)[0]
        D = length[index]
        color_index = (255 * D / max_range).astype(int)
        color_index[np.where(color_index > 255)] = 0
        color_map = cmap[color_index]
        
        padding = 5
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        pixel_range = np.arange(-point_size, point_size + 1).tolist()
        
        for i in pixel_range:
            for j in pixel_range:
                try:
                    image[padding + pixels[:, 1] + i, padding + pixels[:, 0] + j, :] = \
                        (image[padding + pixels[:, 1] + i, padding + pixels[:, 0] + j, :] * (1 - alpha) + 
                         color_map * alpha).astype(np.uint8)
                except IndexError:
                    pass
        
        height, width, _ = image.shape
        projection_image = image[padding:height - padding, padding:width - padding]
        projection_color = image_[pixels[:, 1], pixels[:, 0], :]
        
        colorlize_point = o3d.geometry.PointCloud()
        colorlize_point.points = o3d.utility.Vector3dVector(lidar_pts[index])
        color_point = projection_color / 255
        color_point = np.flip(color_point, axis=1) # BGR to RGB
        colorlize_point.colors = o3d.utility.Vector3dVector(color_point)
        
        return projection_image, colorlize_point


class RosLiDARCameraFusion:
    """ROS Bag 기반 LiDAR-Camera 융합 및 Deskewing 클래스 (rosbags 라이브러리 전용)"""
    
    def __init__(self, bag_path, output_dir):
        self.bag_path = bag_path
        self.output_dir = output_dir
        
        # === 설정 파라미터 (환경에 맞게 수정) ===
        self.topic_image = "/blackfly/image_raw/compressed" 
        self.topic_lidar1 = "/ouster1/points"
        self.topic_lidar2 = "/ouster2/points"
        self.topic_lidar3 = "/ouster3/points"
        self.topic_gps = "/novatel/oem7/odom" # nav_msgs/Odometry
        
        self.scan_period = 0.1
        
        # 카메라 파라미터
        self.intrinsic = np.array([
            [848.018213, -0.875069, 970.050807],
            [0.0, 849.002273, 612.744961],
            [0.0, 0.0, 1.0]
        ])
        self.distortion = np.array([-0.022594, 0.030614, -0.001038, -3.5E-05])
        
        # 외부 파라미터
        self.extrinsic_ouster1 = np.array([
            [-4.72864607e-01, -8.76671114e-01, -8.85822813e-02, -5.17475772e-01],
            [1.25955567e-02, 9.37965264e-02, -9.95511709e-01, -6.61842700e-04],
            [8.81045070e-01, -4.71857997e-01, -3.33108970e-02, -6.78935532e-02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.extrinsic_ouster2 = np.array([
            [9.06068473e-03, -9.99958607e-01, -8.29860879e-04, -2.71261603e-02],
            [-1.29545341e-02, 7.12443573e-04, -9.99915833e-01, -1.03037610e-01],
            [9.99875034e-01, 9.07067258e-03, -1.29475426e-02, -8.95660947e-02],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.extrinsic_ouster3 = np.array([
            [0.49316712, -0.86663967, 0.075643, 0.4635232],
            [0.0065567, -0.08324712, -0.99650736, -0.01027599],
            [0.86990988, 0.49194064, -0.03537245, -0.07933007],
            [0., 0., 0., 1.]
        ])
        
        self.extrinsics = {
            self.topic_lidar1: self.extrinsic_ouster1,
            self.topic_lidar2: self.extrinsic_ouster2,
            self.topic_lidar3: self.extrinsic_ouster3
        }

        # GPS-LiDAR Transform
        extrinsic_gps2ouster2 = np.array([
            [-0.986173, 0.0251448, 0.163798, -0.406935],
            [-0.0267946, -0.99961, -0.00787014, 0.179598],
            [0.163536, -0.0121502, 0.986463, 0.221061],
            [0., 0., 0., 1.]
        ])
        R_fix = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        extrinsic_gps2ouster2 = extrinsic_gps2ouster2 @ R_fix
        extrinsic_gps2ouster2[1, 3] = -extrinsic_gps2ouster2[1, 3]

        T_L2_to_Cam = np.linalg.inv(self.extrinsic_ouster2)
        
        self.gps_extrinsics = {
            self.topic_lidar1: extrinsic_gps2ouster2 @ T_L2_to_Cam @ self.extrinsic_ouster1,
            self.topic_lidar2: extrinsic_gps2ouster2,
            self.topic_lidar3: extrinsic_gps2ouster2 @ T_L2_to_Cam @ self.extrinsic_ouster3
        }

        # 카메라 비네팅 처리 파라미터
        self.mask_params = [960, 600, 1022]
        self.vehicle_cutoff_y = 829
        self.voxel_size = 0.1
        self.max_range = 100
        
        # 궤적 및 결과 저장 변수
        self.interp_pos = None
        self.interp_rot = None
        self.gps_times = []
        self.pointcloud_list = []
        self.coordinate_list = []
        self.position_origin = None
        #라이다 타임스탬프 오프셋
        self.lidar_time_offset = 0.0
        self.is_offset_checked = False

    def get_msg_time(self, msg):
        """rosbags 메시지 객체에서 시간(sec, float) 추출"""
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def parse_pointcloud_rosbags(self, msg):
        """
        rosbags 라이브러리의 PointCloud2 메시지 객체를 numpy array로 변환
        Returns: points (N, 3), rings (N,), timestamp (float)
        """
        width = msg.width
        height = msg.height
        point_step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # 구조: x(0), y(4), z(8), intensity(16), ring(20) 가정
        # (사용자 제공 정보: PointField(name='ring', offset=20, datatype=4, count=1))
        
        points_num = width * height
        if points_num == 0:
            return np.empty((0, 3)), np.empty(0), 0.0
            
        raw_points = data.reshape(-1, point_step)
        
        x = raw_points[:, 0:4].view(np.float32)
        y = raw_points[:, 4:8].view(np.float32)
        z = raw_points[:, 8:12].view(np.float32)
        
        # Ring: offset 20, datatype 4 (uint16)
        ring_offset = 20
        rings = raw_points[:, ring_offset:ring_offset+2].view(np.uint16)
        
        points = np.hstack((x, y, z))
        timestamp = self.get_msg_time(msg)
        
        return points, rings.flatten(), timestamp

    def apply_vignetting_correction(self, image, strength=0.5):
        """비네팅 보정"""
        height, width = image.shape[:2]
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        center_x, center_y = width / 2, height / 2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist
        vignette_mask = 1 + strength * (dist_norm ** 2)
        
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] *= vignette_mask
        return np.clip(result, 0, 255).astype(np.uint8)

    def prepare_trajectory(self):
        """rosbags Reader를 사용하여 GPS 궤적 생성"""
        print("GPS 궤적 생성 중 (rosbags)...")
        positions = []
        quaternions = []
        timestamps = []

        with Reader(self.bag_path) as reader:
            connections = [x for x in reader.connections if x.topic == self.topic_gps]
            if not connections:
                print(f"[Error] GPS 토픽({self.topic_gps})이 bag 파일에 없습니다.")
                return False
                
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # [수정됨] rawdata를 CDR 포맷으로 변환 후 디시리얼라이즈
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                
                ts = self.get_msg_time(msg)
                
                # Odometry 메시지 파싱
                pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                
                timestamps.append(ts)
                positions.append(pos)
                quaternions.append(quat)

        if not timestamps:
            print("[Error] GPS 데이터를 찾을 수 없습니다.")
            return False

        self.gps_times = np.array(timestamps)
        positions = np.array(positions)
        quaternions = np.array(quaternions)

        self.position_origin = positions[0]
        self.interp_pos = interp1d(self.gps_times, positions, axis=0, kind='linear', fill_value="extrapolate")
        self.interp_rot = Slerp(self.gps_times, R.from_quat(quaternions))
        
        print(f"궤적 생성 완료: {len(timestamps)} points")
        return True

    def get_interpolated_pose(self, query_time):
        if self.interp_pos is None:
            return np.eye(4)
        
        pos = self.interp_pos(query_time)
        pos -= self.position_origin
        
        t_clamped = np.clip(query_time, self.gps_times[0], self.gps_times[-1])
        rot_matrix = self.interp_rot(t_clamped).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = pos
        return T

    def deskew_pointcloud(self, points, rings, start_time):
        """LiDAR Deskewing"""
        if len(points) == 0:
            return points

        max_ring = np.max(rings)
        if max_ring == 0: max_ring = 1
        
        T_ref = self.get_interpolated_pose(start_time)
        T_ref_inv = np.linalg.inv(T_ref)

        deskewed_points = np.zeros_like(points)
        unique_rings = np.unique(rings)
        
        for r in unique_rings:
            mask = (rings == r)
            pts_r = points[mask]
            
            t_curr = start_time + (r / max_ring) * self.scan_period
            T_curr = self.get_interpolated_pose(t_curr)
            
            # P_corrected = T_ref^-1 * T_curr * P_raw
            T_correction = T_ref_inv @ T_curr
            
            pts_homo = np.hstack((pts_r, np.ones((pts_r.shape[0], 1))))
            pts_corrected = (T_correction @ pts_homo.T).T
            
            deskewed_points[mask] = pts_corrected[:, :3]
            
        return deskewed_points

    def process_bag(self):
        """rosbags Reader를 사용하는 Bag 처리 메인 루프"""
        if not self.prepare_trajectory():
            return

        print("Bag 파일 처리 시작 (rosbags)...")
        gps_start_time = self.gps_times[0] if len(self.gps_times) > 0 else 0
        latest_lidars = {
            self.topic_lidar1: None,
            self.topic_lidar2: None,
            self.topic_lidar3: None
        }
        
        target_topics = [self.topic_image, self.topic_lidar1, self.topic_lidar2, self.topic_lidar3]
        
        with Reader(self.bag_path) as reader:
            connections = [x for x in reader.connections if x.topic in target_topics]
            
            count = 0
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                topic = connection.topic
                
                # CDR 포맷으로 변환 후 디시리얼라이즈
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                
                if topic in [self.topic_lidar1, self.topic_lidar2, self.topic_lidar3]:
                    points, rings, msg_time = self.parse_pointcloud_rosbags(msg)
                    
                    if not self.is_offset_checked:
                        if gps_start_time > 0:
                            # GPS 시간과 현재 LiDAR 시간의 차이 계산
                            time_diff = gps_start_time - msg_time
                            
                            # 차이가 30초 ~ 40초 사이라면 (약 35초 차이)
                            if 30.0 < abs(time_diff) < 40.0:
                                print(f"[Auto Sync] 시간 차이 감지됨 ({time_diff:.2f}s). 37초 오프셋을 적용합니다.")
                                self.lidar_time_offset = 37.0
                            else:
                                print(f"[Auto Sync] 시간 차이 정상 범위 ({time_diff:.2f}s). 오프셋 미적용.")
                                
                        self.is_offset_checked = True
                    
                    msg_time += self.lidar_time_offset

                    if len(points) > 0:
                        latest_lidars[topic] = {
                            'points': points,
                            'rings': rings,
                            'timestamp': msg_time
                        }
                        
                elif topic == self.topic_image:
                    count += 1
                    if count % 10 == 0:
                        print(f"이미지 처리 중... {count}")

                    try:
                        # === [수정됨] CompressedImage와 Raw Image 분기 처리 ===
                        cv_image = None
                        
                        # 1. CompressedImage 인 경우 ('format' 속성이 있음)
                        if hasattr(msg, 'format'):
                            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                            decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                            
                            # 이미지가 흑백(2차원)이거나 채널이 1개라면 Bayer 변환 시도
                            if len(decoded_img.shape) == 2 or decoded_img.shape[2] == 1:
                                cv_image = cv2.cvtColor(decoded_img, cv2.COLOR_BayerBG2BGR) 
                            else:
                                cv_image = decoded_img
                        
                        # 2. Raw Image 인 경우 ('encoding' 속성이 있음)
                        elif hasattr(msg, 'encoding'):
                            if '8' in msg.encoding:
                                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                            else:
                                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                                
                            cv_image = np_arr.reshape((msg.height, msg.width, -1))
                            
                            if msg.encoding == 'rgb8':
                                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                            elif msg.encoding == 'bayer_rggb8':
                                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BayerBG2BGR)
                        
                        if cv_image is None:
                            print("이미지 디코딩 실패: 알 수 없는 포맷")
                            continue
                            
                    except Exception as e:
                        print(f"이미지 변환 에러: {e}")
                        continue
                    # ========================================================
                        
                    cv_image = self.apply_vignetting_correction(cv_image, strength=0.4)
                    query_time = self.get_msg_time(msg)
                    
                    # 좌표계 (시각화용)
                    pose_matrix = self.get_interpolated_pose(query_time)
                    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    coordinate.transform(pose_matrix)
                    if not self.coordinate_list:
                        coordinate.scale(5, center=coordinate.get_center())
                    self.coordinate_list.append(coordinate)
                    
                    for lidar_topic, lidar_data in latest_lidars.items():
                        if lidar_data is None:
                            continue
                        
                        if abs(lidar_data['timestamp'] - query_time) > 0.5:
                            continue
                            
                        # 1. Deskewing
                        points_corrected = self.deskew_pointcloud(
                            lidar_data['points'], 
                            lidar_data['rings'], 
                            lidar_data['timestamp']
                        )
                        
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points_corrected)
                        
                        # 2. Filtering
                        pcd = PointCloudProcessor.range_filter(pcd, distance_threshold=50)
                        
                        # 3. Projection
                        ext_lidar = self.extrinsics[lidar_topic]
                        _, pcd_colored = PointCloudProcessor.project_to_image(
                            pcd, cv_image, ext_lidar, self.intrinsic, self.distortion, 
                            fisheye=False, max_range=self.max_range,
                            mask_params=self.mask_params, cutoff_y=self.vehicle_cutoff_y
                        )
                        
                        # 4. To Global
                        ext_gps = self.gps_extrinsics[lidar_topic]
                        lidar_time_pose = self.get_interpolated_pose(lidar_data['timestamp'])
                        global_transform = lidar_time_pose @ ext_gps
                        pcd_colored.transform(global_transform)
                        
                        # 5. Save
                        pcd_colored = pcd_colored.voxel_down_sample(self.voxel_size)
                        self.pointcloud_list.append(pcd_colored)

        print("Bag 처리 완료.")

    def merge_and_save(self):
        if not self.pointcloud_list:
            print("저장할 포인트클라우드가 없습니다.")
            return

        print("전체 포인트클라우드 병합 중...")
        merged = o3d.geometry.PointCloud()
        for pcd in self.pointcloud_list:
            merged += pcd
        
        merged = merged.voxel_down_sample(self.voxel_size)
        
        save_path = os.path.join(self.output_dir, "2024-03-19-13-39-16_0_lidar_deskew.pcd")
        o3d.io.write_point_cloud(save_path, merged, write_ascii=False, print_progress=True)
        print(f"저장 완료: {save_path}")
        
        o3d.visualization.draw_geometries(self.coordinate_list + [merged])

def main():
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step3: Rosbag-based colorize + deskew")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--bag", required=True, help="입력 rosbag 파일 또는 디렉토리")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.outputs.pcd
    os.makedirs(output_dir, exist_ok=True)

    processor = RosLiDARCameraFusion(args.bag, output_dir)
    processor.process_bag()
    processor.merge_and_save()


if __name__ == "__main__":
    main()

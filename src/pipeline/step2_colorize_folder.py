import os
import cv2
import natsort
import numpy as np
import copy
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime


class TimestampParser:
    """타임스탬프 파싱을 담당하는 클래스"""
    
    FORMATS = [
        "%Y-%m-%d_%H-%M-%S.%f",
        "%Y-%m-%d-%H-%M-%S_%f",
        "%Y-%m-%d-%H-%M-%S.%f",
        "%Y-%m-%d_%H-%M-%S",
        "%Y-%m-%d-%H-%M-%S"
    ]
    
    @staticmethod
    def parse(filename):
        """파일명에서 타임스탬프 추출"""
        name_body = os.path.splitext(filename)[0]
        
        # Unix Timestamp 시도
        try:
            return float(name_body)
        except ValueError:
            pass
        
        # 날짜 형식 파싱
        for fmt in TimestampParser.FORMATS:
            try:
                dt = datetime.strptime(name_body, fmt)
                return dt.timestamp()
            except ValueError:
                continue
        
        return None


class PointCloudProcessor:
    """포인트클라우드 처리 관련 기능"""
    
    @staticmethod
    def range_filter(pcd, distance_threshold=30, query_point=[0, 0, 0]):
        """거리 기반 필터링"""
        points = np.asarray(pcd.points)
        distances = np.linalg.norm(points - query_point, axis=1)
        filtered_indices = np.where(distances <= distance_threshold)[0]
        return pcd.select_by_index(filtered_indices)
    
    @staticmethod
    def undistortion(x, y, distortion, r, fisheye=False):
        """왜곡 보정"""
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
                        point_size=3, alpha=0.1, max_range=15,mask_params=None, cutoff_y=None):
        """포인트클라우드를 이미지에 투영"""
        cmap = plt.cm.get_cmap("jet", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        
        # lidar_pts = np.asarray(pointcloud.points)
        # image = copy.deepcopy(image_)
        # length = np.linalg.norm(lidar_pts, 2, axis=1)
        
        # homo = extrinsic @ np.vstack((lidar_pts.T, np.ones(lidar_pts.shape[0])))
        # x = homo[0, :] / homo[2, :]
        # y = homo[1, :] / homo[2, :]
        # r = np.sqrt(x**2 + y**2)
        
        # X_distortion, Y_distortion = PointCloudProcessor.undistortion(x, y, distortion, r, fisheye)
        # pixel = intrinsic @ np.vstack((X_distortion, Y_distortion, np.ones(X_distortion.shape[0])))
        
        # min_range = 0
        # index = np.where((pixel[0, :] > 0) & (pixel[0, :] < image.shape[1]) & 
        #                 (pixel[1, :] > 0) & (pixel[1, :] < image.shape[0]) & 
        #                 (homo[2, :] > min_range))

        #카메라 비네팅 보정 코드 추가
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
        
        # --- 필터링 조건 계산 ---
        # 1. 기본 이미지 영역 및 전방 포인트 조건
        valid_mask = (pixel[0, :] > 0) & (pixel[0, :] < width) & \
                     (pixel[1, :] > 0) & (pixel[1, :] < height) & \
                     (homo[2, :] > 0)

        # 2. 원형 마스크 조건 적용 (MASK_CX, MASK_CY, MASK_R)
        if mask_params is not None:
            mcx, mcy, mr = mask_params
            dist_from_center = np.sqrt((pixel[0, :] - mcx)**2 + (pixel[1, :] - mcy)**2)
            valid_mask &= (dist_from_center < mr)

        # 3. 차량 하단 컷오프 적용 (VEHICLE_CUTOFF_Y)
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
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, 
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image[padding + pixels[:, 1], padding + pixels[:, 0], :] = color_map
        
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
        color_point = np.flip(color_point, axis=1)
        colorlize_point.colors = o3d.utility.Vector3dVector(color_point)
        
        return projection_image, colorlize_point


class FileManager:
    """파일 관리 클래스"""
    
    @staticmethod
    def get_files_and_times(folder):
        """폴더에서 파일 및 타임스탬프 추출"""
        if not os.path.exists(folder):
            print(f"[Error] 폴더가 없습니다: {folder}")
            return [], np.array([])
        
        files = natsort.natsorted(os.listdir(folder))
        times = []
        valid_files = []
        
        for f in files:
            if not f.endswith('.pcd'):
                continue
            
            ts = TimestampParser.parse(f)
            if ts is not None:
                times.append(ts)
                valid_files.append(f)
        
        return valid_files, np.array(times)


class LiDARCameraFusion:
    """LiDAR-Camera 융합 메인 클래스"""
    
    def __init__(self, config_path):
        """초기화 및 설정"""
        # 카메라 내부 파라미터
        self.intrinsic = np.array([
            [848.018213, -0.875069, 970.050807],
            [0.0, 849.002273, 612.744961],
            [0.0, 0.0, 1.0]
        ])
        self.distortion = np.array([-0.022594, 0.030614, -0.001038, -3.5E-05])
        
        # LiDAR-Camera 외부 파라미터
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
        
        self.extrinsic = {
            "ouster1": self.extrinsic_ouster1,
            "ouster2": self.extrinsic_ouster2,
            "ouster3": self.extrinsic_ouster3
        }
        
        # GPS-LiDAR 변환 행렬
        extrinsic_gps2ouster2 = np.array([
            [-0.986173, 0.0251448, 0.163798, -0.406935],
            [-0.0267946, -0.99961, -0.00787014, 0.179598],
            [0.163536, -0.0121502, 0.986463, 0.221061],
            [0., 0., 0., 1.]
        ])
        
        R_fix = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        extrinsic_gps2ouster2 = extrinsic_gps2ouster2 @ R_fix
        extrinsic_gps2ouster2[1, 3] = -extrinsic_gps2ouster2[1, 3]
        
        # 센서 변환 행렬 계산
        T_L2_to_Cam = np.linalg.inv(self.extrinsic['ouster2'])
        self.extrinsic_gps2ouster1 = extrinsic_gps2ouster2 @ T_L2_to_Cam @ self.extrinsic['ouster1']
        self.extrinsic_gps2ouster2 = extrinsic_gps2ouster2
        self.extrinsic_gps2ouster3 = extrinsic_gps2ouster2 @ T_L2_to_Cam @ self.extrinsic['ouster3']
        
        # 처리 파라미터
        self.fisheye = False
        self.projection_distance = 100
        self.distance_threshold = 50
        self.voxel_size = 0.1
        self.time_tolerance = 0.2
        
        # 경로 설정
        self.file_directory = config_path
        self.pointcloud_folder1 = os.path.join(config_path, "ouster1/points/")
        self.pointcloud_folder2 = os.path.join(config_path, "ouster2/points/")
        self.pointcloud_folder3 = os.path.join(config_path, "ouster3/points/")
        self.image_folder = os.path.join(config_path, "blackfly/")
        self.navigation_file = os.path.join(config_path, "navigation.csv")
        self.output_folder = os.path.join(config_path, "pointcloud_color/")
        
        # 데이터 저장 변수
        self.df_nav = None
        self.nav_timestamps = None
        self.files_1 = []
        self.files_2 = []
        self.files_3 = []
        self.times_1 = np.array([])
        self.times_2 = np.array([])
        self.times_3 = np.array([])
        self.image_files = []
        
        # 결과 저장 변수
        self.coordinate_list = []
        self.pointcloud_list = []
        self.is_origin_set = False
        self.position_x_origin = 0
        self.position_y_origin = 0
        self.position_z_origin = 0
        
        # 좌표계 객체
        self.coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()

        #카메라 비네팅 보정
        self.mask_params = [960, 600, 1022]  # [CX, CY, R]
        self.vehicle_cutoff_y = 829
    
    def validate_paths(self):
        """경로 유효성 검사"""
        required_paths = [
            self.pointcloud_folder2,
            self.image_folder,
            self.navigation_file
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                print(f"[Critical Error] 경로 없음: {path}")
                return False
        
        os.makedirs(self.output_folder, exist_ok=True)
        return True
    
    def load_navigation_data(self):
        """GPS 네비게이션 데이터 로드"""
        try:
            self.df_nav = pd.read_csv(self.navigation_file)
            self.df_nav.columns = self.df_nav.columns.str.strip()
            
            time_cols = [c for c in self.df_nav.columns if 'time' in c.lower() or 'stamp' in c.lower()]
            if not time_cols:
                print("[Error] CSV에 타임스탬프 컬럼이 없습니다")
                return False
            
            self.nav_timestamps = self.df_nav[time_cols[0]].values
            print(f"Navigation 데이터 로드 완료: {len(self.nav_timestamps)} rows")
            return True
        except Exception as e:
            print(f"[Error] Navigation 로드 실패: {e}")
            return False
    
    def load_sensor_files(self):
        """모든 센서 파일 및 타임스탬프 로드"""
        print("센서 파일 타임스탬프 계산 중...")
        
        self.files_1, self.times_1 = FileManager.get_files_and_times(self.pointcloud_folder1)
        self.files_2, self.times_2 = FileManager.get_files_and_times(self.pointcloud_folder2)
        self.files_3, self.times_3 = FileManager.get_files_and_times(self.pointcloud_folder3)
        
        print(f"LiDAR 파일 로드 -> L1: {len(self.files_1)}, L2: {len(self.files_2)}, L3: {len(self.files_3)}")
        
        if len(self.files_2) == 0:
            print("[Critical Error] Ouster2 파일이 없습니다!")
            return False
        
        self.image_files = natsort.natsorted(os.listdir(self.image_folder))
        print(f"이미지 파일 로드 -> {len(self.image_files)}")
        
        return True
    
    def match_lidar_file(self, query_time, files, times):
        """시간 기반 LiDAR 파일 매칭"""
        if len(times) == 0:
            return None
        
        idx = (np.abs(times - query_time)).argmin()
        if abs(times[idx] - query_time) < self.time_tolerance:
            return files[idx]
        return None
    
    def get_gps_pose(self, query_time):
        """GPS 포즈 추출"""
        closest_idx = (np.abs(self.nav_timestamps - query_time)).argmin()
        
        try:
            position_x = self.df_nav['position_x'].loc[closest_idx]
            position_y = self.df_nav['position_y'].loc[closest_idx]
            position_z = self.df_nav['position_z'].loc[closest_idx]
            orientation_x = self.df_nav['orientation_x'].loc[closest_idx]
            orientation_y = self.df_nav['orientation_y'].loc[closest_idx]
            orientation_z = self.df_nav['orientation_z'].loc[closest_idx]
            orientation_w = self.df_nav['orientation_w'].loc[closest_idx]
            
            return {
                'position': [position_x, position_y, position_z],
                'orientation': [orientation_x, orientation_y, orientation_z, orientation_w]
            }
        except KeyError:
            return None
    
    def set_origin(self, position):
        """원점 설정"""
        if not self.is_origin_set:
            self.position_x_origin = position[0]
            self.position_y_origin = position[1]
            self.position_z_origin = position[2]
            self.is_origin_set = True
    
    def compute_trajectory(self, pose):
        """궤적 행렬 계산"""
        pos_x = pose['position'][0] - self.position_x_origin
        pos_y = pose['position'][1] - self.position_y_origin
        pos_z = pose['position'][2] - self.position_z_origin
        
        trajectory = np.identity(4)
        r = R.from_quat(pose['orientation'])
        trajectory[:3, :3] = r.as_matrix()
        trajectory[:3, 3] = [pos_x, pos_y, pos_z]
        
        return trajectory
    
    def process_lidar_sensor(self, lidar_path, image, ext_lidar, ext_gps, trajectory):
        """개별 LiDAR 센서 처리"""
        if lidar_path is None or not os.path.exists(lidar_path):
            return None
        
        pcd = o3d.io.read_point_cloud(lidar_path)
        if pcd.is_empty():
            return None
        
        # 거리 필터링
        pcd = PointCloudProcessor.range_filter(pcd, distance_threshold=self.distance_threshold)
        
        # 이미지 투영 및 색상 할당
        _, pcd = PointCloudProcessor.project_to_image(
            pcd, image, ext_lidar, self.intrinsic, self.distortion, 
            self.fisheye, max_range=self.projection_distance,
            mask_params=self.mask_params,
            cutoff_y=self.vehicle_cutoff_y
        )
        
        # 글로벌 좌표계 변환
        translate_matrix = trajectory @ ext_gps
        pcd_global = pcd.transform(translate_matrix)
        
        # 다운샘플링
        pcd_global = pcd_global.voxel_down_sample(voxel_size=self.voxel_size)
        
        return pcd_global
    
    def apply_vignetting_correction(self, image, strength=0.5):
        """
        간이 다항식 모델을 이용한 비네팅 보정
        strength: 보정 강도 (0: 보정 없음, 수치가 클수록 외곽을 더 밝게 함)
        """
        height, width = image.shape[:2]
        
        # 1. 이미지 중심으로부터의 거리 맵 생성
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        center_x, center_y = width / 2, height / 2
        
        # 중심으로부터의 거리 계산 및 정규화 (0~1)
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist
        
        # 2. 보정 마스크 생성 (중심은 1, 외곽으로 갈수록 커지는 계수)
        # Gain = 1 + strength * (dist_norm^2) 등의 모델 사용
        vignette_mask = 1 + strength * (dist_norm ** 2)
        
        # 3. 채널별 적용 (BGR)
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] *= vignette_mask
            
        # 4. 픽셀 값 제한 (0~255) 및 타입 변환
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def process_single_frame(self, image_file):
        """단일 프레임 처리"""
        # 이미지 로드
        image_path = os.path.join(self.image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            return False
        #비네팅 보정
        image = self.apply_vignetting_correction(image, strength=0.4)
        
        # 타임스탬프 파싱
        query_time = TimestampParser.parse(image_file)
        if query_time is None:
            return False
        
        # LiDAR 파일 매칭
        lidar1_file = self.match_lidar_file(query_time, self.files_1, self.times_1)
        lidar2_file = self.match_lidar_file(query_time, self.files_2, self.times_2)
        lidar3_file = self.match_lidar_file(query_time, self.files_3, self.times_3)
        
        lidar1_path = os.path.join(self.pointcloud_folder1, lidar1_file) if lidar1_file else None
        lidar2_path = os.path.join(self.pointcloud_folder2, lidar2_file) if lidar2_file else None
        lidar3_path = os.path.join(self.pointcloud_folder3, lidar3_file) if lidar3_file else None
        
        # GPS 포즈 추출
        pose = self.get_gps_pose(query_time)
        if pose is None:
            return False
        
        # 원점 설정
        self.set_origin(pose['position'])
        
        # 궤적 계산
        trajectory = self.compute_trajectory(pose)
        
        # 좌표계 시각화
        coordinate_translate = copy.deepcopy(self.coordinate).transform(trajectory)
        if not self.coordinate_list:
            coordinate_translate = coordinate_translate.scale(5, center=coordinate_translate.get_center())
        self.coordinate_list.append(coordinate_translate)
        
        # 각 LiDAR 센서 처리
        lidar_configs = [
            {'path': lidar1_path, 'ext_lidar': self.extrinsic['ouster1'], 'ext_gps': self.extrinsic_gps2ouster1},
            {'path': lidar2_path, 'ext_lidar': self.extrinsic['ouster2'], 'ext_gps': self.extrinsic_gps2ouster2},
            {'path': lidar3_path, 'ext_lidar': self.extrinsic['ouster3'], 'ext_gps': self.extrinsic_gps2ouster3}
        ]
        
        for config in lidar_configs:
            pcd_global = self.process_lidar_sensor(
                config['path'], image, config['ext_lidar'], 
                config['ext_gps'], trajectory
            )
            if pcd_global is not None:
                self.pointcloud_list.append(pcd_global)
        
        return True
    
    def merge_pointclouds(self):
        """포인트클라우드 병합"""
        if not self.pointcloud_list:
            print("[Error] 병합할 포인트클라우드가 없습니다")
            return None
        
        print("\n포인트클라우드 병합 중...")
        merged = copy.deepcopy(self.pointcloud_list[0])
        
        for i in range(1, len(self.pointcloud_list)):
            merged += self.pointcloud_list[i]
        
        print("최종 다운샘플링...")
        merged = merged.voxel_down_sample(voxel_size=self.voxel_size)
        
        return merged
    
    def save_result(self, pointcloud):
        """결과 저장"""
        save_filename = f"{self.file_directory.split('/')[-1]}_full3_corrected.pcd"
        print(f"저장 중: {save_filename}...")
        o3d.io.write_point_cloud(save_filename, pointcloud, 
                                 write_ascii=False, print_progress=True)
        return save_filename
    
    def visualize(self, pointcloud):
        """시각화"""
        print("시각화 중...")
        o3d.visualization.draw_geometries(self.coordinate_list + [pointcloud])
    
    def run(self):
        """전체 프로세스 실행"""
        # 1. 경로 검증
        if not self.validate_paths():
            return
        
        # 2. 데이터 로드
        if not self.load_navigation_data():
            return
        
        if not self.load_sensor_files():
            return
        
        # 3. 메인 루프
        print("\n메인 루프 시작...")
        for i, image_file in enumerate(self.image_files):
            if i % 10 == 0:
                print(f"처리 중 {i}/{len(self.image_files)}: {image_file}")
            
            self.process_single_frame(image_file)
        
        # 4. 병합
        merged_pointcloud = self.merge_pointclouds()
        if merged_pointcloud is None:
            return
        
        # 5. 저장
        self.save_result(merged_pointcloud)
        
        # 6. 시각화
        self.visualize(merged_pointcloud)


def main():
    """메인 함수 - configs/default.yaml에서 입력 폴더 경로를 읽음."""
    import argparse
    from src.common import load_config

    parser = argparse.ArgumentParser(description="Step2: Single-folder colorize")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--folder", default=None, help="처리할 데이터 폴더 (없으면 projection.input_folder 사용)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    folder = args.folder or cfg.projection.input_folder
    processor = LiDARCameraFusion(folder)
    processor.run()


if __name__ == "__main__":
    main()
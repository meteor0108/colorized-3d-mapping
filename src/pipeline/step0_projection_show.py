import os
import cv2
import numpy as np
import copy
import open3d as o3d

import matplotlib.pyplot as plt


def pointcloud_range_filter(pcd, distance_threshold=30, query_point = [0, 0, 0]):
    # 거리 기준으로 필터링
    # query_point = [0, 0, 0]  # 거리를 측정할 기준점 좌표
    # distance_threshold = 100  # 거리 임계값 (미터)

    # 기준점과의 거리를 계산하여 거리 임계값 이내의 포인트만 남깁니다.
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points - query_point, axis=1)
    filtered_indices = np.where(distances <= distance_threshold)[0]

    # 거리 임계값 이내의 포인트 클라우드 생성
    filtered_pcd = pcd.select_by_index(filtered_indices)
    return filtered_pcd

### projection ###
def undistortion(x,y,distortion,r,fisheye=False):
    if fisheye == False and distortion.shape[0] == 5:
        X_Radial_dist = x*(1+distortion[0]*r**2+distortion[1]*r**4+distortion[4]*r**6)
        Y_Radial_dist = y*(1+distortion[0]*r**2+distortion[1]*r**4+distortion[4]*r**6)

        X_Tangential_dist = 2*distortion[2]*x*y+distortion[3]*(r**2+2*x**2)
        Y_Tangential_dist = distortion[2]*(r**2+2*y**2)+2*distortion[3]*x*y

        X_distortion = X_Radial_dist + X_Tangential_dist
        Y_distortion = Y_Radial_dist + Y_Tangential_dist

    elif fisheye == False and distortion.shape[0] == 4:
        X_Radial_dist = x*(1+distortion[0]*r**2+distortion[1]*r**4)
        Y_Radial_dist = y*(1+distortion[0]*r**2+distortion[1]*r**4)

        X_Tangential_dist = 2*distortion[2]*x*y+distortion[3]*(r**2+2*x**2)
        Y_Tangential_dist = distortion[2]*(r**2+2*y**2)+2*distortion[3]*x*y

        X_distortion = X_Radial_dist + X_Tangential_dist
        Y_distortion = Y_Radial_dist + Y_Tangential_dist

    elif fisheye == True:
        theat = np.arctan(r)
        theatD = theat*(1+distortion[0]*theat**2+distortion[1]*theat**4+\
                          distortion[2]*theat**6+distortion[3]*theat**8)
        X_distortion = (theatD/r)*x
        Y_distortion = (theatD/r)*y
    return X_distortion, Y_distortion

def projection(pointcloud, image_, extrinsic, intrinsic, distortion, fisheye, point_size=3, alpha=0.1, max_range=15):
    # matplot colormap
    cmap = plt.cm.get_cmap("jet", 256)                          # 256개의 색상 리스트 반환
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255 # 256개의 색상 리스트 반환
    # 포인트클라우드 값 추출
    lidar_pts = np.asarray(pointcloud.points)

    init_projection_result = None
    cal_projection_result = None

    
    image =  copy.deepcopy(image_)
    length = np.linalg.norm(lidar_pts,2,axis=1)
    # homo = extrinsic @ np.vstack(( lidar_pts.T, np.ones(lidar_pts.shape[0]) ))
    # x = homo[0,:]/homo[2,:]; y = homo[1,:]/homo[2,:]
    homo = extrinsic @ np.vstack(( lidar_pts.T, np.ones(lidar_pts.shape[0]) ))
    # [수정] Z값(깊이) 체크: 점들이 카메라 앞에 있는지 확인
    if homo[2, :].max() < 0:
        print("[Warning] 모든 포인트가 카메라 뒤쪽(음수)에 있습니다. Extrinsic 역행렬 여부를 확인하세요.")
    x = homo[0,:]/homo[2,:]; y = homo[1,:]/homo[2,:]

    r = np.sqrt(x**2 + y**2)
    X_distortion, Y_distortion = undistortion(x,y,distortion,r,fisheye=fisheye)
    pixel = intrinsic @ np.vstack(( X_distortion, Y_distortion, np.ones(X_distortion.shape[0]) ))

    # min_range = 3 # 디스토션 파라미터 핀홀 5개 사용 할때
    min_range = 0 # 디스토션 파라미터 핀홀 4개
    # max_range = 15
    index = np.where((pixel[0,:]>0) & (pixel[0,:]<image.shape[1]) & \
                    (pixel[1,:]>0) & (pixel[1,:]<image.shape[0]) & \
                    (homo[2,:] >min_range) )
    valid_count = len(index[0])
    print(f"[Debug] 이미지 내 매칭된 포인트 개수: {valid_count}")
    if valid_count == 0:
        return image # 매칭된 게 없으면 원본 이미지 반환

    pixels = pixel.T[index,:2].astype(int)[0]
    color_index = (length[index]/max_range).astype(int)*255
    D = length[index]


    color_index = (255*D/max_range).astype(int)
    color_index[np.where(color_index > 255)] = 0
    color_map =  cmap[color_index]
    padding = 5
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # 포인트 클라우드 센터
    image[padding+pixels[:,1],padding+pixels[:,0],:] = color_map

    pixel_range = np.arange(-point_size,point_size+1).tolist()
    for i in pixel_range:
        for j in pixel_range:
            image[padding+pixels[:,1]+i,padding+pixels[:,0]+j,:] = \
                (image[padding+pixels[:,1]+i,padding+pixels[:,0]+j,:]*(1-alpha) + (color_map)*alpha ).astype(np.uint8)

    # 이미지의 네 가장자리에서 5픽셀을 제외한 부분 선택
    height, width, _ = image.shape
    projection_image = image[padding:height-padding, padding:width-padding]
    return projection_image



def main():

    intrinsic = np.array([[848.018213, -0.875069, 970.050807],
                          [       0.0,849.002273, 612.744961],
                          [       0.0,       0.0,        1.0]])
    distortion = np.array([-0.022594, 0.030614, -0.001038, -3.5E-05])

    # extrinsic_ouster1 =np.array([\
    #     [ 0.45977578, -0.01761806, -0.88786026,  0.19366024],\
    #     [ 0.88310561, -0.0961274 ,  0.45922108,  0.49972864],\
    #     [-0.09343829, -0.99521311, -0.02863844, -0.1261248 ],\
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]
    # ])

    # extrinsic_ouster2 =np.array([\
    #     [-0.01636717,  0.01519758, -0.99975054, -0.10317535],\
    #     [ 0.99985623, -0.00418248, -0.01643248,  0.04378189],\
    #     [-0.00443117, -0.99987576, -0.01512694, -0.14911367],\
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]
    # ])

    # extrinsic_ouster3 =np.array([\
    #     [-0.50011948, -0.00159615, -0.86595494,  0.14161749],\
    #     [ 0.86322401,  0.07845827, -0.49868689, -0.43618848],\
    #     [ 0.0687373 , -0.99691612, -0.03786068, -0.09142943],\
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]
    # ])

    extrinsic_ouster1 =np.array([\
        [-4.72864607e-01, -8.76671114e-01 ,-8.85822813e-02, -5.17475772e-01],
        [ 1.25955567e-02, 9.37965264e-02, -9.95511709e-01, -6.61842700e-04,],
        [ 8.81045070e-01, -4.71857997e-01, -3.33108970e-02, -6.78935532e-02],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    extrinsic_ouster2 =np.array([\
        [ 9.06068473e-03, -9.99958607e-01, -8.29860879e-04, -2.71261603e-02],
        [-1.29545341e-02, 7.12443573e-04, -9.99915833e-01, -1.03037610e-01],
        [ 9.99875034e-01, 9.07067258e-03, -1.29475426e-02, -8.95660947e-02],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    extrinsic_ouster3 =np.array([\
        [ 0.49316712, -0.86663967, 0.075643, 0.4635232 ],
        [ 0.0065567, -0.08324712, -0.99650736, -0.01027599],
        [ 0.86990988, 0.49194064, -0.03537245, -0.07933007],
        [ 0., 0., 0., 1. ]
    ])

    extrinsic = {
        "ouster1":extrinsic_ouster1,
        "ouster2":extrinsic_ouster2,
        "ouster3":extrinsic_ouster3
    }



    from src.common import load_config
    _cfg = load_config("default.yaml")
    fisheye = bool(_cfg.camera.fisheye)
    projection_disturnce = _cfg.projection.max_range
    lidar_name = "ouster2"

    file_diractory = _cfg.projection.input_folder
    pointcloud_folder1 = file_diractory+f"/ouster1/points/"
    pointcloud_folder2 = file_diractory+f"/ouster2/points/"
    pointcloud_folder3 = file_diractory+f"/ouster3/points/"
    image_folder = file_diractory+f"/blackfly/"

    file_list = os.listdir(image_folder)
    file_list.sort()
    for file_name in file_list:
        if not file_name.endswith(".png"):
            continue
        ###추가된 부분 ###
        base_name = os.path.splitext(file_name)[0]

        pcd_path1 = os.path.join(pointcloud_folder1, base_name + ".pcd")
        pcd_path2 = os.path.join(pointcloud_folder2, base_name + ".pcd")
        pcd_path3 = os.path.join(pointcloud_folder3, base_name + ".pcd")

        img_path = os.path.join(image_folder, file_name)

        # if not (os.path.exists(pcd_path1) and os.path.exists(pcd_path2) and os.path.exists(pcd_path3)):
        #     print(f"[Skip] PCD 파일이 누락되었습니다: {base_name}")
        #     continue
            
        pointcloud1 = o3d.io.read_point_cloud(pcd_path1)
        pointcloud2 = o3d.io.read_point_cloud(pcd_path2)
        pointcloud3 = o3d.io.read_point_cloud(pcd_path3)

        image = cv2.imread(img_path)
        if image is None:
            print(f"[Error] 이미지를 불러오지 못했습니다: {img_path}")
            continue
        print(f"[Info] 이미지 로드 성공: {image.shape}")
        ###추가완료###

        # file_name = file_name.split('.')[0]
        # print(file_name)
        # pointcloud1 = o3d.io.read_point_cloud(pointcloud_folder1 + f"{file_name}.pcd")
        # pointcloud2 = o3d.io.read_point_cloud(pointcloud_folder2 + f"{file_name}.pcd")
        # pointcloud3 = o3d.io.read_point_cloud(pointcloud_folder3 + f"{file_name}.pcd")
        # pointcloud = pointcloud_range_filter(pointcloud, distance_threshold=50)
        # image = cv2.imread(image_folder + f"{file_name}.png")
        # projection_image1 = projection(pointcloud1, image, np.linalg.inv(extrinsic['ouster1']), intrinsic, distortion, fisheye, max_range=projection_disturnce)
        # projection_image2 = projection(pointcloud2, projection_image1, np.linalg.inv(extrinsic['ouster2']), intrinsic, distortion, fisheye, max_range=projection_disturnce)
        # projection_image3 = projection(pointcloud3, projection_image2, np.linalg.inv(extrinsic['ouster3']), intrinsic, distortion, fisheye, max_range=projection_disturnce)
        projection_image1 = projection(pointcloud1, image, extrinsic['ouster1'], intrinsic, distortion, fisheye, max_range=projection_disturnce)
        projection_image2 = projection(pointcloud2, projection_image1, extrinsic['ouster2'], intrinsic, distortion, fisheye, max_range=projection_disturnce)
        projection_image3 = projection(pointcloud3, projection_image2, extrinsic['ouster3'], intrinsic, distortion, fisheye, max_range=projection_disturnce)

        # cv2.imwrite("calibration_projection.png",projection_image2)
        cv2.imwrite("calibration_projection.png",projection_image3)

        cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('image',projection_image3)
        key = cv2.waitKey(0)
        if key == ord("q"):
            exit()

if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np
import copy
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def main():

    intrinsic = np.array([[848.018213, -0.875069, 970.050807],
                          [       0.0,849.002273, 612.744961],
                          [       0.0,       0.0,        1.0]])
    distortion = np.array([-0.022594, 0.030614, -0.001038, -3.5E-05])

    extrinsic_ouster1 =np.array([\
        [ 0.45977578, -0.01761806, -0.88786026,  0.19366024],\
        [ 0.88310561, -0.0961274 ,  0.45922108,  0.49972864],\
        [-0.09343829, -0.99521311, -0.02863844, -0.1261248 ],\
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])

    extrinsic_ouster2 =np.array([\
        [-0.01636717,  0.01519758, -0.99975054, -0.10317535],\
        [ 0.99985623, -0.00418248, -0.01643248,  0.04378189],\
        [-0.00443117, -0.99987576, -0.01512694, -0.14911367],\
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])

    extrinsic_ouster3 =np.array([\
        [-0.50011948, -0.00159615, -0.86595494,  0.14161749],\
        [ 0.86322401,  0.07845827, -0.49868689, -0.43618848],\
        [ 0.0687373 , -0.99691612, -0.03786068, -0.09142943],\
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])

    extrinsic = {
        "ouster1":extrinsic_ouster1,
        "ouster2":extrinsic_ouster2,
        "ouster3":extrinsic_ouster3
    }
    extrinsic_gps2ouster2 = np.array([
        [-0.986173 ,  0.0251448, 0.163798  ,-0.406935],\
        [-0.0267946, -0.99961  ,-0.00787014, 0.179598],\
        [ 0.163536 , -0.0121502, 0.986463  , 0.221061],\
        [ 0.       ,  0.       , 0.        , 1.      ]
    ])


    from src.common import load_config
    _cfg = load_config("default.yaml")
    fisheye = bool(_cfg.camera.fisheye)
    projection_disturnce = _cfg.projection.max_range
    lidar_name = "ouster2"

    file_diractory = _cfg.projection.input_folder
    pointcloud_folder1 = file_diractory+f"/ouster1/points/"
    pointcloud_folder2 = file_diractory+f"/ouster2/points/"
    pointcloud_folder3 = file_diractory+f"/ouster3/points/"

    navigation_file = file_diractory + "/navigation.csv"
    df_nav = pd.read_csv(navigation_file)
    df_nav_count = df_nav.shape[0]
    position_x_origin = 0
    position_y_origin = 0
    position_z_origin = 0
    trajectory_list = []
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_list = []
    pointcloud_list = []

    for i in range(df_nav_count):
        position_x = df_nav['position_x'].loc[i]
        position_y = df_nav['position_y'].loc[i]
        position_z = df_nav['position_z'].loc[i]

        orientation_x = df_nav['orientation_x'].loc[i]
        orientation_y = df_nav['orientation_y'].loc[i]
        orientation_z = df_nav['orientation_z'].loc[i]
        orientation_w = df_nav['orientation_w'].loc[i]
        print(position_x, position_y, position_z)
        if i == 0:
            position_x_origin = position_x
            position_y_origin = position_y
            position_z_origin = position_z
        pos_x = position_x - position_x_origin
        pos_y = position_y - position_y_origin
        pos_z = position_z - position_z_origin
        trajectory = np.identity(4)

        r = R.from_quat([orientation_x, orientation_y, orientation_z, orientation_w])
        rot = r.as_matrix()
        trajectory[:3,:3] = rot
        trajectory[:3,3] = [pos_x, pos_y, pos_z]

        coordinate_translate = copy.deepcopy(coordinate).transform(trajectory)
        if i == 0:
            coordinate_translate = coordinate_translate.scale(5, center=coordinate_translate.get_center())
        coordinate_list.append(coordinate_translate)

        translate = trajectory @ extrinsic_gps2ouster2
        # pointcloud1 = o3d.io.read_point_cloud(pointcloud_folder1 + f"{str(i).zfill(5)}.pcd")
        pointcloud2 = o3d.io.read_point_cloud(pointcloud_folder2 + f"{str(i).zfill(5)}.pcd").transform(translate)
        # pointcloud3 = o3d.io.read_point_cloud(pointcloud_folder3 + f"{str(i).zfill(5)}.pcd")

        pointcloud_list.append(pointcloud2)

    feature_list = coordinate_list + pointcloud_list
    o3d.visualization.draw_geometries(feature_list)

if __name__ == "__main__":
    main()

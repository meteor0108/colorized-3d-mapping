import os
import open3d as o3d
import cv2
import numpy as np


def pcd_show(path):
    pcd = o3d.io.read_point_cloud(path)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd, mesh])

def image_show(path):
    image = cv2.imread(path)
    csv_file_path = path.replace(path.split(".")[1], 'csv')
    if os.path.exists(csv_file_path):
        corver = np.loadtxt(csv_file_path,delimiter=',', usecols=range(4))
        for i in range(corver.shape[0]):
            center = corver[i,2:].astype(int).tolist()
            cv2.circle(image, center, 10, (0,0,255), 2)
    cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    key = cv2.waitKey(0)
    #if key == 'q':
    #    continue

while True:
    path = input("Put in Data Here : ").split("'")[1]
    if os.path.isdir(path) == True:
        pcd_folder = False
        image_folder = False
        for name in os.listdir(path):
            extenstion = name.split(".")[1]
            if extenstion == 'pcd':
                pcd_folder = True
            elif extenstion in ['jpg', 'png', 'jpeg']:
                image_folder = True
        
        for name in os.listdir(path):
            file_path = os.path.join(path,name)
            extenstion = name.split(".")[1]
            if pcd_folder == True:
                pcd_show(file_path)
            elif image_folder == True and extenstion != 'csv':
                image_show(file_path)
    elif path.split(".")[1] == 'pcd':
        pcd_show(path)
    elif path.split(".")[1] in ['jpg', 'png', 'jpeg']:
        image_show(path)
        

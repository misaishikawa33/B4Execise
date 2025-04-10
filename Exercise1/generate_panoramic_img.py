#2枚の画像からパノラマ画像を生成するプログラム
#左から順番に画像を連結していくようなプログラム
#src_points = ２枚目画像の四隅
#dst_points = １枚目の画像の対応点 

import cv2
import numpy as np
import pdb

from compute_homography_matrix import compute_homography_matrix
from get_points import get_points
from mark_points import mark_points

def get_new_img_size(M, img1, img2):
    # 画像1と画像2のサイズを取得
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # 画像1の四隅の点
    corners_img1 = np.array([
        [0, 0, 1],          # 左上
        [width1, 0, 1],     # 右上
        [0, height1, 1],    # 左下
        [width1, height1, 1]  # 右下
    ])

    # 画像2の四隅の点を射影変換
    corners_img2 = np.array([
        [0, 0, 1],          # 左上
        [width2, 0, 1],     # 右上
        [0, height2, 1],    # 左下
        [width2, height2, 1]  # 右下
    ])
    transformed_corners_img2 = np.dot(M, corners_img2.T).T
    transformed_corners_img2 /= transformed_corners_img2[:, 2][:, np.newaxis]  # 正規化

    # すべての点（画像1の四隅 + 変換された画像2の四隅）をまとめる
    all_corners = np.vstack((corners_img1[:, :2], transformed_corners_img2[:, :2]))

    # 新しい範囲を計算
    min_x, min_y = all_corners.min(axis=0).astype(int)
    max_x, max_y = all_corners.max(axis=0).astype(int)

    print('min_points  ', [min_x, min_y])
    print('max_points  ', [max_x, max_y])

    return [min_x, min_y], [max_x, max_y]

# パノラマ画像を生成
def generate_panoramic_img(img1, img2, M, min_points, max_points):
    #変換後画像のための行列を作成。
    new_width = abs(min_points[0]) + abs(max_points[0]) 
    new_height = abs(min_points[1]) + abs(max_points[1])
    print(new_width, new_height)
    new_img = np.zeros((new_height, new_width, 3))
    # print(new_img.shape)

    # 画像１を変換後画像にコピー
    x1 = abs(min_points[0])
    x2 = abs(min_points[0])+img1.shape[1]
    y1 = abs(min_points[1])
    y2 = abs(min_points[1])+img1.shape[0]
    new_img[y1:y2, x1:x2, :] = img1

    #射影変換行列の逆行列
    M_inv = np.linalg.inv(M)
    #変換（変換行列に基づいた画素の移動）
    offset_x = abs(min_points[0])
    offset_y = abs(min_points[1])

    for i in range(-(offset_y), new_img.shape[0] - offset_y):      #y軸
        for j in range(-(offset_x), new_img.shape[1] - offset_x):     #x軸
            dst_pixel = np.array([j, i, 1])
            x, y, w = np.dot(M_inv, dst_pixel)
            src_pixel = np.array([x/w, y/w]).astype(np.int32)
            # print(type(dst_pixel[0]))
            if(0 <= src_pixel[0] < img2.shape[1] and 0 <= src_pixel[1] < img2.shape[0]):
                new_img[i + offset_y][j + offset_x] = img2[src_pixel[1]][src_pixel[0]]

    cv2.imwrite('homography.jpg',new_img)

    return 0

# -------------------------------------------------------------------

# 画像を読み込み
img_name1 = input("画像のファイル名を入力：")
img1 = cv2.imread(img_name1)
# 画像を読み込み
img_name2 = input("画像のファイル名を入力：")
img2 = cv2.imread(img_name2)

# img1 = cv2.imread("/home/misa/katuda/B4Execise/Exercise1/source.jpg")
# img2 = cv2.imread("/home/misa/katuda/B4Execise/Exercise1/target.jpg")



# 画像２の対応点を入力
src_points = get_points(img2)
# 画像１の対応点を入力
dst_points = get_points(img1)
# 射影変換行列の計算
M = compute_homography_matrix(src_points,dst_points)
# 変換後の左上、右下の点を取得する
min_points, max_points = get_new_img_size(M, img1, img2)
# パノラマ画像を生成
generate_panoramic_img(img1, img2, M, min_points, max_points)

#特徴点を保存
mark_points(img_name1, img1, dst_points)
mark_points(img_name2, img2, src_points)
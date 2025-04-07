#射影変換を使用するためのプログラム
#順変換、逆変換に対応

import cv2
import numpy as np
import pdb

#-------------------------------------------------------------------------------------------

def compute_homography_matrix(src_points, dst_points):

    x0, y0, x1, y1, x2, y2, x3, y3 = [element for row in src_points for element in row]
    print(x0, y0, x1, y1, x2, y2, x3, y3)
    X0, Y0, X1, Y1, X2, Y2, X3, Y3 = [element for row in dst_points for element in row]
    print(X0, Y0, X1, Y1, X2, Y2, X3, Y3)

    tmp_array1 = np.array([[x0, y0, 1, 0, 0, 0, (-1)*x0*X0, (-1)*y0*X0],
                         [0, 0, 0, x0, y0, 1, (-1)*x0*Y0, (-1)*y0*Y0],
                         [x1, y1, 1, 0, 0, 0, (-1)*x1*X1, (-1)*y1*X1],
                         [0, 0, 0, x1, y1, 1, (-1)*x1*Y1, (-1)*y1*Y1],
                         [x2, y2, 1, 0, 0, 0, (-1)*x2*X2, (-1)*y2*X2],
                         [0, 0, 0, x2, y2, 1, (-1)*x2*Y2, (-1)*y2*Y2],
                         [x3, y3, 1, 0, 0, 0, (-1)*x3*X3, (-1)*y3*X3],
                         [0, 0, 0, x3, y3, 1, (-1)*x3*Y3, (-1)*y3*Y3]])    

    tmp_array_inv = np.linalg.inv(tmp_array1)

    tmp_array2 = np.array([X0, Y0, X1, Y1, X2, Y2, X3, Y3])

    M = np.dot(tmp_array_inv, tmp_array2)
    M = np.append(M, 1)
    M = M.reshape(3, 3)
    print(M)

    return M

def move_pixel_foward(img, M):
    print(img.shape)
    new_img = np.zeros((height, width, 3))
    print(new_img.shape)

    for i in range(height):
        for j in range(width):
            src_pixel = np.array([j, i, 1])
            x, y, w = np.dot(M, src_pixel)
            dst_pixel = np.array([x/w, y/w]).astype(np.int32)
            
            if(0 < dst_pixel[0] < width and 0 < dst_pixel[1] < height):
                # print(dst_pixel)
                new_img[dst_pixel[1]][dst_pixel[0]] = img[i][j]

    cv2.imwrite('homography.jpg',new_img)

    return 0

def move_pixel_reverse(img, M):
    #変換後画像のための行列を作成
    new_img = np.zeros((height, width, 3))
    print(new_img.shape)
    #射影変換行列の逆行列
    M_inv = np.linalg.inv(M)
    #変換（変換行列に基づいた画素の移動）
    for i in range(height):
        for j in range(width):
            dst_pixel = np.array([j, i, 1])
            x, y, w = np.dot(M_inv, dst_pixel)
            src_pixel = np.array([x/w, y/w]).astype(np.int32)
            # print(type(dst_pixel[0]))
            if(0 < src_pixel[0] < width and 0 < src_pixel[1] < height):
                new_img[i][j] = img[src_pixel[1]][src_pixel[0]]

    cv2.imwrite('homography.jpg',new_img)

    return 0

# 画像を読み込み
img_name = input("画像のファイル名を入力：")
img = cv2.imread(img_name)

height = img.shape[0]
width = img.shape[1]

print(height, width)

src_points = [[0, 0],
              [0, height],
              [width, height],
              [width, 0]]

dst_points = [[40,40],
              [40,height-40],
              [width, height],
              [width, 0]]

M = compute_homography_matrix(src_points,dst_points)
# move_pixel_foward(img, M)
move_pixel_reverse(img, M)
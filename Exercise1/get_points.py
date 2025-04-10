# 画像の座標を4点取得するプログラム

import cv2
# 縮小率
SCALE = 5
# 座標の配列
points = []

count = 0

def onMouse(event, x, y, flags, params):
    global count, points
    if event == cv2.EVENT_LBUTTONDOWN and count < 4:
        count += 1
        print(count ,[x*SCALE, y*SCALE])
        points.append([x*SCALE, y*SCALE])
        


def get_points(img):
    global count, points
    count = 0
    points = []
    width, height = img.shape[1] // SCALE, img.shape[0] // SCALE
    img = cv2.resize(img, (width,height))
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', onMouse)
    cv2.waitKey(20000)
    print(points)
    return points

# src_points = get_point()
# print(src_points)
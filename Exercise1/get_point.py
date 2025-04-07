# 画像の座標を取得するプログラム

import cv2
# 縮小率
SCALE = 1

def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x*SCALE, y*SCALE)


def get_point():
    img_name = input("画像のファイル名を入力：")
    img = cv2.imread(img_name)
    # print('x = ', img.shape[1] ,'\t' , 'y = ',img.shape[0])
    width, height = img.shape[1] // SCALE, img.shape[0] // SCALE
    img = cv2.resize(img, (width,height))
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', onMouse)
    cv2.waitKey(30000)

get_point()
# 与えられた座標に印をつけるプログラム


import cv2
import colorsys

from get_points import get_points

CONTRAST = 0.03

# 画像，座標が引数
def mark_points(img_name, img, points):

    radius = 3
    thickness = -1
    thickness_font = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5


    for i, center in enumerate(points):
        h = CONTRAST*i % 1
        color  = colorsys.hls_to_rgb(h, 0.5, 0.5)
        color = tuple(value * 255 for value in color)
        color = tuple(reversed(color))
        text = 'p' + str(i)

        cv2.circle(img,center,radius,color,thickness)
        center[0] = center[0]+5
        cv2.putText(img, text, center, font, font_scale, color, thickness_font)

    cv2.imwrite('marked_' + img_name , img)


# # # 画像を読み込み
# img_name = input("画像のファイル名を入力：")
# img = cv2.imread(img_name)
# points = get_points(img)
# mark_points(img_name, img, points)

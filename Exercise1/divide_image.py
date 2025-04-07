#1枚の画像を縦方向に分割する．任意の分割数，重なりを指定可能


import cv2

# 画像を縦方向に分割
def img_divide_vertical():

    img_name = input("画像のファイル名を入力：")
    img = cv2.imread(img_name)
    height, width = img.shape[0], img.shape[1]
    division_num = int(input("分割数を入力："))
    overlap = int(input("重複の幅を入力："))

    for i in range(division_num):
        x1 = int(width/division_num * i)
        x2 = int(width/division_num * (i+1) + overlap)
        sliced_img = img[:, x1:x2  , :]
        cv2.imwrite( str(i+1) + '_' + img_name,sliced_img)

# 画像を縦方向に分割
img_divide_vertical()

# 全方位画像を生成
# コマンドライン引数として，画像名，出力画像サイズ（高さ，幅）を入力

import cv2
import numpy as np
import sys
import math

def main():
    # コマンドライン引数の取得
    args = sys.argv
    img_name = args[1]  #画像の名前
    new_img_size = np.array([args[2], args[3]]).astype(int) #生成画像のサイズ
    # print(img_name)
    # print(new_img_size)

    # 画像を読み込み
    img = cv2.imread(img_name)
    # print(img.shape)

    # 透視投影画像を生成
    new_img = omnidirectional_img(img, new_img_size)

    # 画像の書き出し
    cv2.imwrite('projection_img.jpg', new_img)



# 透視投影画像を生成
def omnidirectional_img(img, new_img_size):

    # 画像のサイズを取得
    img_size = np.array([img.shape[0], img.shape[1]]).astype(int)

    # 画角を計算
    new_img_theta = 2 * math.atan( (math.pi * new_img_size[1]) / img_size[1])
    new_img_phi = 2 * math.atan( (math.pi * new_img_size[0]) / (2 * img_size[0]))
    # print(new_img_theta,new_img_phi)

    #画素間の長さを計算
    angle_degree = new_img_theta/2
    delta_x = 2 * math.tan(angle_degree) / new_img_size[1]

    angle_degree = new_img_phi/2
    delta_y = 2 * math.tan(angle_degree) / new_img_size[0]
    # print(delta_x, delta_y)

    # 透視投影画像の領域を確保
    new_img = np.zeros((new_img_size[0], new_img_size[1], 3))

    # 透視投影画像の中心点を求める（W/2, H/2のこと）
    center_u = new_img_size[1]/2
    center_v = new_img_size[0]/2

    # 透視投影画像の画素からベクトルを計算．ベクトルから全方位画像の角度を求め，画素を移動する
    for v_p in range(new_img_size[0]):
        for u_p in range (new_img_size[1]):
            x, y, z = (u_p - center_u)*delta_x, (v_p - center_v)*delta_y, 1
            # print(x, y, z)
            img_theta = math.atan(x/z)
            img_phi = -math.atan(y/(math.sqrt(x ** 2 + z ** 2)))
            # print(img_theta, img_phi)

            u_e = ( (img_theta+math.pi) * (new_img_size[1]/(2*math.pi)) ).astype(int)
            v_e = ( ((math.pi/2)-img_phi) * (new_img_size[0]/math.pi) ).astype(int)
            # print(type(u_e),type(v_e))
            # print('org:',u_e, v_e, 'new', u_p, v_p)
            if(0 < u_e < img.shape[1] and 0 < v_e < img.shape[0]):
                new_img[v_p][u_p] = img[v_e][u_e]
    
    return new_img


# __name__はモジュール名の文字列が格納されている変数．main関数があるときのみmain関数を実行する
if __name__ == "__main__":
    main()
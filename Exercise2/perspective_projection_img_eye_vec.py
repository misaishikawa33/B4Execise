# 指定する2次元座標を画像の中心点として透視投影画像を生成，
# コマンドライン引数として，画像名，視線ベクトル(v,u)，視線角度（光軸），（angleを入力してから）画角（水平，垂直），（sizeを入力してから）画像サイズ（高さ，幅）

import cv2
import numpy as np
import sys
import math

def main():
    # コマンドライン引数の取得
    args = sys.argv
    img_name = args[1]  #画像の名前
    vec_2d_eye = np.array( [args[2], args[3]] ).astype(int)
    # print(vec_2d_eye)
    psi_eye = int(args[4]) #視線角度

    # 標準入力に情報が含まれている場合のみ取り出す
    new_img_theta, new_img_phi = 0, 0
    new_img_size = np.zeros(2)

    for i, element in enumerate(args):
        if element == 'angle':
            new_img_theta, new_img_phi = int(args[i+1]), int(args[i+2])
        if element == 'size':
            new_img_size = np.array( [args[i+1], args[i+2] ]).astype(int)
    # print(img_name, type(theta_eye), phi_eye, psi_eye)
    # print(new_img_theta, new_img_phi)
    # print(new_img_size)

    # 画像を読み込み
    img = cv2.imread(img_name)
    # print(img.shape)

    # 透視投影画像を生成
    new_img = perspective_projection_img(img, vec_2d_eye, psi_eye, new_img_theta, new_img_phi, new_img_size)

    # 画像の書き出し
    cv2.imwrite('projection_img.jpg', new_img)



# 透視投影画像を生成
def perspective_projection_img(img, vec_2d_eye, psi_eye, new_img_theta = 0, new_img_phi = 0, new_img_size = np.zeros(2)):
   
    # 画像のサイズを取得
    img_size = np.array([img.shape[0], img.shape[1]]).astype(int)
    # print(img_size)

    # 画角，画像サイズを取得
    # 画角，画像サイズを取得できるか確認，どちらも指定されていない場合プログラムを終了
    if new_img_theta == 0  and new_img_phi == 0 and np.all(new_img_size == 0):
        print("画角あるいは画像サイズを少なくとも1つ指定してください")
        sys.exit()
    # 画像サイズのみ指定されている場合，画角を計算
    elif new_img_theta == 0 and new_img_phi == 0: 
        new_img_theta = 2 * math.atan( (math.pi * new_img_size[1]) / img_size[1])
        new_img_phi = 2 * math.atan( (math.pi * new_img_size[0]) / (2 * img_size[0]))
    # 画角のみ指定されている場合，画像サイズを計算
    elif np.all(new_img_size == 0):
        new_img_theta = math.radians(new_img_theta)
        new_img_phi = math.radians(new_img_phi)
        new_img_size[1] = 2 * math.tan( new_img_theta/2 ) * ( img_size[1] / (2*math.pi) ) 
        new_img_size[0] = 2 * math.tan( new_img_phi/2 ) * ( img_size[0] / math.pi )
        new_img_size = new_img_size.astype(int)
    # どちらも指定されている場合
    else:   
        new_img_theta = math.radians(new_img_theta)
        new_img_phi = math.radians(new_img_phi)
    print('画角：', math.degrees(new_img_theta), math.degrees(new_img_phi))
    print('画像サイズ', new_img_size)


    #画素間の長さを計算
    delta_x = ( 2 * math.tan(new_img_theta/2) ) / new_img_size[1]

    delta_y = ( 2 * math.tan(new_img_phi/2) ) / new_img_size[0]
    # print(delta_x, delta_y)


    # 回転行列の計算
    
    theta_eye = (vec_2d_eye[1]-(img_size[1]/2)) * ((2*math.pi)/img_size[1])
    phi_eye = ((img_size[0]/2)-vec_2d_eye[0]) * (math.pi/img_size[0])
    # 視点角度をラジアンに変換
    psi_eye = math.radians(psi_eye)

    # y軸周りの回転行列
    y_rot_mx = np.array([[math.cos(theta_eye), 0, math.sin(theta_eye)],
                         [0, 1, 0],
                         [(-1)*math.sin(theta_eye), 0, math.cos(theta_eye)]])
    # x軸周りの回転行列
    x_rot_mx = np.array([[1, 0, 0],
                         [0, math.cos(phi_eye), (-1)*math.sin(phi_eye)],
                         [0, math.sin(phi_eye), math.cos(phi_eye)]])

    # xy軸周りの回転行列
    xy_rot_mx = np.dot(y_rot_mx, x_rot_mx)
    # print(xy_rot_mx)

    # 回転軸（z軸の単位ベクトル）を取得
    z_unit_vec = np.array([[0], [0], [1]])
    l_x, l_y, l_z = np.squeeze( np.dot(xy_rot_mx, z_unit_vec) )
    # print(l_x, l_y, l_z)

    z_rot_mx = np.array([[pow(l_x,2) * (1-math.cos(psi_eye)) + math.cos(psi_eye), l_x * l_y * (1-math.cos(psi_eye)) - (l_z*math.sin(psi_eye)), l_z * l_x * (1-math.cos(psi_eye)) + (l_y*math.sin(psi_eye))],
                         [l_x * l_y * (1-math.cos(psi_eye)) + l_z * math.sin(psi_eye), pow(l_y, 2) * (1-math.cos(psi_eye)) + math.cos(psi_eye), l_y * l_z * (1-math.cos(psi_eye)) - (l_x*math.sin(psi_eye))],
                         [ l_z * l_x * (1-math.cos(psi_eye)) - (l_y*math.sin(psi_eye)), l_y * l_z * (1-math.cos(psi_eye)) + (l_x*math.sin(psi_eye)), pow(l_z, 2) * (1-math.cos(psi_eye)) + math.cos(psi_eye)]])

    rot_mx = np.dot(z_rot_mx, xy_rot_mx)

    # print(y_rot_mx)
    # print(x_rot_mx)
    # print(xy_rot_mx)
    # print(l_x, l_y, l_z)
    # print(z_rot_mx)
    # print(rot_mx)


    # 透視投影画像の領域を確保
    new_img = np.zeros((new_img_size[0], new_img_size[1], 3))

    # 透視投影画像の中心点を求める（W/2, H/2のこと）
    center_u = new_img_size[1]/2
    center_v = new_img_size[0]/2

    # 透視投影画像の画素からベクトルを計算．ベクトルから全方位画像の角度を求め，画素を移動する
    for v_p in range(new_img_size[0]):
        for u_p in range (new_img_size[1]):
            vec_eye = np.array( [(u_p - center_u)*delta_x, (v_p - center_v)*delta_y, 1] )
            x, y, z =np.dot(rot_mx, vec_eye)
            # print(x, y, z)
            # print(math.atan2(z, x))
            img_theta = math.atan2(x, z)
            # img_theta = math.atan(x/z)
            img_phi = -math.atan(y/(math.sqrt(x ** 2 + z ** 2)))
            # print(img_theta, img_phi)

            u_e = ( (img_theta+math.pi) * (img_size[1]/(2*math.pi)) ).astype(int)
            v_e = ( ((math.pi/2)-img_phi) * (img_size[0]/math.pi) ).astype(int)
            # print(u_e, v_e)
            # print('org:',u_e, v_e, 'new', u_p, v_p)
            # if(0 < u_e < img_size[1] and 0 < v_e < img_size[0]):
            new_img[v_p][u_p] = img[v_e-1][u_e-1]
    
    return new_img


# __name__はモジュール名の文字列が格納されている変数．main関数があるときのみmain関数を実行する
if __name__ == "__main__":
    main()
# 射影変換行列を計算する
import numpy as np

def compute_homography_matrix(src_points, dst_points):

    x0, y0, x1, y1, x2, y2, x3, y3 = [element for row in src_points for element in row]
    # print(x0, y0, x1, y1, x2, y2, x3, y3)
    X0, Y0, X1, Y1, X2, Y2, X3, Y3 = [element for row in dst_points for element in row]
    # print(X0, Y0, X1, Y1, X2, Y2, X3, Y3)

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

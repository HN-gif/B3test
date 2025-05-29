from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import os

# ----------------------------
# 引数処理（最大10枚まで）
# ----------------------------
parser = argparse.ArgumentParser(description='Image stitching with AKAZE (overwrite version).')
for i in range(1, 11):
    parser.add_argument(f'--input{i}', help=f'Path to input image {i}.', default='')
args = parser.parse_args()

# 画像読み込み
image_paths = [getattr(args, f'input{i}') for i in range(1, 11) if getattr(args, f'input{i}')]
images = [cv.imread(p) for p in image_paths if os.path.exists(p)]

if len(images) < 2:
    print('2枚以上の画像を指定してください。')
    exit(1)

# ----------------------------
# 特徴点検出・記述・マッチング
# ----------------------------
akaze = cv.AKAZE_create()
keypoints = []
descriptors = []

for img in images:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kpts, desc = akaze.detectAndCompute(gray, None)
    keypoints.append(kpts)
    descriptors.append(desc)

# ...（前処理と特徴点抽出までは同じ）...

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
homographies = []

for i in range(len(images) - 1):
    matches = matcher.knnMatch(descriptors[i], descriptors[i+1], 2)
    good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]
    
    if len(good_matches) < 10:
        print(f'画像 {i} と {i+1} の間に十分なマッチがありません。')
        exit(1)

    # ✅ マッチングの可視化と保存
    img_matches = cv.drawMatches(
        images[i], keypoints[i],
        images[i+1], keypoints[i+1],
        good_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv.imwrite(f'matches_{i}_{i+1}.png', img_matches)

    # ホモグラフィ推定
    # ４点に絞る、あるいは……

    # しかし一般には、N>=4 点で頑健推定できる findHomography が適切
    src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[i+1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    if len(src_pts) < 4 or len(dst_pts) < 4:
        print(f'画像 {i} と {i+1} の間に十分なマッチがありません。')
        exit(1)
    src_pts = src_pts.reshape(-1, 2)
    dst_pts = dst_pts.reshape(-1, 2)

    # H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    # if H is None:
    #     print(f'画像 {i} と {i+1} のホモグラフィ推定に失敗')
    #     exit(1)
    # homographies.append(H)
    
    # 1. translation推定
    # 単純に差をとる（最も単純）
    # reshape で (N, 2) に変換
    # delta = dst_pts - src_pts  # shape: (N, 2)
    # tx, ty = np.mean(delta, axis=0)  # 正常に unpack 可能
    # T = np.array([[1, 0, tx],
    #             [0, 1, ty],
    #             [0, 0, 1]], dtype=np.float32)
    # homographies.append(T)

    # 2. similarity推定
    # 最小二乗法で (1+a, b, tx, ty) を推定
    A = []
    b = []

    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, -y, 1, 0])
        A.append([y,  x, 0, 1])
        b.append(xp)
        b.append(yp)

    A = np.array(A)
    b = np.array(b)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # shape: (4,)
    a, b_, tx, ty = params
    S = np.array([
        [1+a, -b_, tx],
        [b_, 1+a, ty],
        [0, 0, 1]
    ], dtype=np.float32)
    homographies.append(S)
    # 3. affine推定
    # 最小二乗法で (a, b, c, d, tx, ty) を推定
    # A = []
    # b = []
    # for (x, y), (xp, yp) in zip(src_pts, dst_pts):
    #     A.append([x, y, 1, 0, 0, 0])
    #     A.append([0, 0, 0, x, y, 1])
    #     b.append(xp)
    #     b.append(yp)
    # A = np.array(A)
    # b = np.array(b)
    # params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # shape: (6,)
    # a00, a01, tx, a10, a11, ty = params
    # H = np.array([
    #     [a00, a01, tx],
    #     [a10, a11, ty],
    #     [0, 0, 1]
    # ], dtype=np.float32)
    # homographies.append(H)






# ----------------------------
# 累積ホモグラフィの作成
# ----------------------------
cumulative_H = [np.eye(3)]
for H in homographies:
    cumulative_H.append(cumulative_H[-1] @ H)

# ----------------------------
# キャンバスサイズの推定
# ----------------------------
all_corners = []
for i, img in enumerate(images):
    h, w = img.shape[:2]
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv.perspectiveTransform(corners, cumulative_H[i])
    all_corners.append(warped_corners)

all_corners = np.concatenate(all_corners, axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
canvas_width = x_max - x_min
canvas_height = y_max - y_min
offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

# ----------------------------
# ワーピング
# ----------------------------
# cv.warpPerspective(img, M, (600, 800))を使用して、各画像をキャンバスにワーピング
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

for i, img in enumerate(images):
    M = offset @ cumulative_H[i]
    warp = cv.warpPerspective(img, M, (canvas_width, canvas_height))
    mask = (warp.sum(axis=2) > 0)
    canvas[mask] = warp[mask]


    
# ----------------------------
# 結果保存と表示
# ----------------------------
cv.imwrite('Result.png', canvas)
cv.waitKey(0)
cv.destroyAllWindows()

# Usage:
# python Akaze.py --input1 IMG_8734.JPG --input2 IMG_8735.JPG --input3 IMG_8736.JPG --input4 IMG_8737.JPG --input5 IMG_8738.JPG
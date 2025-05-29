import cv2
import numpy as np
import glob

def stitch_images(image_paths, detector_type='AKAZE', matcher_type='BF', ratio_thresh=0.5):
    # 1. 画像読み込み
    imgs = [cv2.imread(p) for p in image_paths]

    # 2. 特徴点検出・記述子計算
    if detector_type == 'AKAZE':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif detector_type == 'ORB':
        detector = cv2.ORB_create(5000)
        norm = cv2.NORM_HAMMING
    else:  # SIFT
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2

    keypoints, descriptors = [], []
    for img in imgs:
        kp, des = detector.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # マッチャー準備
    if matcher_type == 'BF':
        matcher = cv2.BFMatcher(norm)
    else:
        # FLANN parameters for SIFT
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # 3–4. 隣接画像間でマッチング→ホモグラフィ推定
    Hs = [np.eye(3)]
    for i in range(len(imgs)-1):
        matches = matcher.knnMatch(descriptors[i], descriptors[i+1], k=2)
        good = [m for m,n in matches if m.distance < ratio_thresh * n.distance]

        src = np.float32([ keypoints[i][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst = np.float32([ keypoints[i+1][m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        Hs.append(H)

    # 累積ホモグラフィ計算
    cumHs = [np.eye(3)]
    for H in Hs[1:]:
        cumHs.append(cumHs[-1] @ H)

    # 5. 合成キャンバスサイズ計算
    corners = []
    for img, H in zip(imgs, cumHs):
        h, w = img.shape[:2]
        pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, H)
        corners.append(dst)
    all_pts = np.vstack(corners)
    x_min, y_min = all_pts[:,:,0].min(), all_pts[:,:,1].min()
    x_max, y_max = all_pts[:,:,0].max(), all_pts[:,:,1].max()

    # 平行移動行列
    tx, ty = -x_min, -y_min
    canvas_w, canvas_h = int(x_max - x_min), int(y_max - y_min)
    offset = np.array([[1,0,tx],[0,1,ty],[0,0,1]])

    # 6. 各画像をワーピングして合成
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for img, H in zip(imgs, cumHs):
        M = offset @ H
        warp = cv2.warpPerspective(img.astype(np.float32), M, (canvas_w, canvas_h))
        mask = (warp.sum(axis=2) > 0).astype(np.float32)

        canvas += warp
        weight += mask

    # 7. 平均化
    weight = np.maximum(weight, 1e-8)
    result = (canvas / weight[:,:,None]).astype(np.uint8)
    return result

if __name__ == "__main__":
    # 直接ファイル名をリストで指定
    image_files = [
        "IMG_8734.JPG",
        "IMG_8735.JPG",
        "IMG_8736.JPG",
        "IMG_8737.JPG",
        "IMG_8738.JPG",
    ]
    pano = stitch_images(image_files, detector_type='AKAZE', matcher_type='BF')
    cv2.imwrite("stitched_result.jpg", pano)
    print("stitched_result.jpg を出力しました")
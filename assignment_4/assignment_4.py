import cv2
import numpy as np


def harris_corner_detection(referance_image):
    referance_image = cv2.imread(referance_image)
    gray = cv2.cvtColor(referance_image,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    referance_image[dst>0.01 * dst.max()] = [0, 0, 255]

    cv2.imwrite('./assignment_4/solutions/corner_harris.jpg', referance_image)


def SIFT_image_alignment(image_to_align, referance_image, max_feature=10, good_match_percent=0.7):
    MIN_MATCH_COUNT = 4

    img1 = cv2.imread(image_to_align, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(referance_image, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # apply max_feature / good_match_percent
    good = sorted(good, key=lambda x: x.distance)
    good = good[:max_feature]
    num_good_matches = int(len(good) * good_match_percent)
    good = good[:num_good_matches]

    matchesMask = None
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

    draw_params = dict(matchColor=(0, 0, 255),
                        singlePointColor=None,
                        matchesMask=matchesMask,
                        flags=2)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good, None, **draw_params, )

    cv2.imwrite('./assignment_4/solutions/SIFT_image_matcher.png', img_matches)

    # algin image
    aligned = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    cv2.imwrite('./assignment_4/solutions/SIFT_image_align.png', aligned)


referance_image = '/Users/olehh/UiA/IKT213_Haugholt/assignment_4/reference_img.png'

image_to_align = '/Users/olehh/UiA/IKT213_Haugholt/assignment_4/align_this.jpg'


harris_corner_detection(referance_image)

SIFT_image_alignment(image_to_align, referance_image, max_feature=10, good_match_percent=0.7)
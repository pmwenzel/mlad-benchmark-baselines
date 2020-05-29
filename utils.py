import filecmp
import os

import cv2
import numpy as np
from skimage import io


def load_image(image_path: str) -> np.ndarray:
    """
    Read image from disk.
    :param image_path: Path to input image.
    :return: uint8 numpy array sized H x W.
    """
    # load image
    img = io.imread(image_path)

    # assert img dtype
    assert img.dtype == 'uint8'

    return img


def uint8tofloat32ndarray(image_uint8: np.ndarray) -> np.ndarray:
    """
    This methods convert uint8 numpy array to float32 numpy array.
    :param image_uint8: uint8 numpy array [0, 255].
    :return: float32 numpy array [0, 1].
    """
    assert image_uint8.dtype == np.uint8

    image_float32 = image_uint8.astype(np.float32) / 255

    return image_float32


def convert_kpt_to_cv(keypoints):
    return [cv2.KeyPoint(keypoints[i, 0], keypoints[i, 1], 1) for i in range(keypoints.shape[0])]


def estimate_pose_PnPRansac(pts0, pts1, matches, disparity, camMatrix, baseline):
    """
    This method estimates the camera pose using PnP + Ransac.
    :param pts0: keypoints of source image.
    :param pts1: keypoints of target image.
    :param matches: matched correspondences.
    :param disparity: disparity map.
    :param camMatrix: intrinsic camera parameters.
    :param baseline: stereo baseline.
    :return: tvecs: translation vector, rot_matrix: rotation matrix.
    """
    # set calibration params
    fx = camMatrix[0, 0]
    fy = camMatrix[1, 1]
    cx = camMatrix[0, 2]
    cy = camMatrix[1, 2]

    fxi = 1.0 / fx
    fyi = 1.0 / fy
    cxi = -cx / fx
    cyi = -cy / fy

    # check type of keypoints
    if type(pts0) == type(pts1) == np.ndarray:
        keypoints0cv = convert_kpt_to_cv(pts0)
        keypoints1cv = convert_kpt_to_cv(pts1)
    else:
        keypoints0cv = pts0
        keypoints1cv = pts1

    points3d = []
    points2d = []
    for match in matches:
        left_kpt = keypoints0cv[match.queryIdx]
        right_kpt = keypoints1cv[match.trainIdx]

        disp = disparity[int(round(left_kpt.pt[1])), int(round(left_kpt.pt[0]))]
        if disp <= 1:
            continue

        # filter out the hood
        if left_kpt.pt[1] > 330 or right_kpt.pt[1] > 330:
            continue

        # convert disparity to depths
        depth = baseline * fx / disp

        # point3d
        point3d = np.zeros(3)
        point3d[0] = (left_kpt.pt[0] * fxi + cxi) * depth
        point3d[1] = (left_kpt.pt[1] * fyi + cyi) * depth
        point3d[2] = depth

        # point2d
        point2d = np.array(right_kpt.pt)

        points3d.append(point3d)
        points2d.append(point2d)

    points3d = np.asarray(points3d)
    points2d = np.asarray(points2d)

    # PnPRansac
    try:
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(points3d, points2d, camMatrix, None, reprojectionError=3,
                                                      iterationsCount=1000, confidence=0.999999)
    except:
        # return identity
        return np.zeros((3, 1)), np.eye(3)

    # rotation matrix
    rot_matrix, _ = cv2.Rodrigues(rvecs)

    # return translation and rotation matrix
    return tvecs, rot_matrix


def load_calibration(path_to_calibration_folder: str) -> (np.ndarray, np.ndarray):
    """
    This method loads the relevant calibration files and returns camMatrix and the stereo baseline.
    :param path_to_calibration_folder: Path to the calibration folder.
    :return: camMatrix (3,3) matrix containing intrinsic parameters, stereo baseline.
    """

    # init camMatrix
    camMatrix = np.eye(3)

    # check if intrinsics are identical
    assert filecmp.cmp(os.path.join(path_to_calibration_folder, 'undistorted_calib_0.txt'),
                       os.path.join(path_to_calibration_folder, 'undistorted_calib_1.txt'))

    with open(os.path.join(path_to_calibration_folder, 'undistorted_calib_0.txt'), 'r') as f:
        lines = f.readlines()
        fx, fy, cx, cy = list(map(float, lines[0].split(' ')[1:5]))

    camMatrix[0, 0] = fx
    camMatrix[1, 1] = fy
    camMatrix[0, 2] = cx
    camMatrix[1, 2] = cy

    # load stereo calib
    stereo_calib = np.loadtxt(fname=os.path.join(path_to_calibration_folder, 'undistorted_calib_stereo.txt'))

    baseline = -stereo_calib[0, 3]

    return camMatrix, baseline

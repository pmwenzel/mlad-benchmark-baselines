import os
import sys

sys.path.append(os.path.join('third_party', 'r2d2'))

import argparse

import numpy as np
import torch
from pyquaternion import Quaternion
import cv2
from tqdm import tqdm
from utils import load_image, uint8tofloat32ndarray, estimate_pose_PnPRansac, load_calibration
from third_party.r2d2.tools import common
from third_party.r2d2.extract import load_network, NonMaxSuppression, extract_multiscale

# Based on third_party/r2d2/extract.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--model", type=str, default=os.path.join(
        'third_party', 'r2d2', 'models', 'r2d2_WAF_N16.pt'),
                        help='Model path')
    parser.add_argument("--top-k", type=int, default=5000, help='number of keypoints')
    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')
    parser.add_argument("--dataset-path", type=str, required=True, help='path to dataset.')
    parser.add_argument("--test-sequence", type=int, default=0, help='test sequence, select either 0 or 1')

    args = parser.parse_args()

    iscuda = common.torch_set_gpu(args.gpu)

    # load the network...
    net = load_network(args.model)
    if iscuda:
        net = net.cuda()

    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr=args.reliability_thr, rep_thr=args.repeatability_thr)

    # source sequence
    source_seq = 'recording_2020-04-07_10-20-32'
    if args.test_sequence == 0:
        target_seq = 'recording_2020-03-24_17-45-31'  # test_sequence0
    elif args.test_sequence == 1:
        target_seq = 'recording_2020-04-23_19-37-00'  # test_sequence1
    else:
        exit('Test sequence can either be 0 or 1.')

    tasks = ['easy', 'moderate', 'hard']

    # folder source
    folder_source = os.path.join(args.dataset_path, source_seq)
    # folder target
    folder_target = os.path.join(args.dataset_path, target_seq)

    # normalization gray scale
    gray_mean = 0.471
    gray_std = 0.232

    # initialize bf_matcher object
    bf_matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)

    # initialize stereoBM object
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)

    # load camMatrix and baseline
    camMatrix, baseline = load_calibration(os.path.join(folder_source, 'Calibration'))

    for tsk in tasks:

        # reloc file
        reloc_file = os.path.join(args.dataset_path, source_seq, 'RelocalizationFilesTest',
                                  'relocalizationFile_' + target_seq + '_' + tsk + '.txt')
        # Result file
        reloc_result = open(
            os.path.join(folder_source,
                         'relocalizationResult_r2d2_eccv-challenge-' + target_seq + '_' + tsk + '.txt'),
            'w')

        # Process the file
        with open(reloc_file, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines, total=len(lines)):
            if line.startswith('#'):
                continue
            l = line.rstrip().split(" ")

            img_source_cam0_path = os.path.join(folder_source, 'undistorted_images/cam0', l[0] + '.png')
            img_source_cam1_path = os.path.join(folder_source, 'undistorted_images/cam1', l[0] + '.png')
            img_target_cam0_path = os.path.join(folder_target, 'undistorted_images/cam0', l[1] + '.png')

            # load images
            img_source_cam0 = load_image(img_source_cam0_path)
            img_source_cam1 = load_image(img_source_cam1_path)
            img_target_cam0 = load_image(img_target_cam0_path)

            # estimate disparity
            disparity = stereo.compute(img_source_cam0, img_source_cam1).astype(np.float32) / 16.0

            # convert images to float32
            img_source_cam0 = uint8tofloat32ndarray(img_source_cam0)
            img_target_cam0 = uint8tofloat32ndarray(img_target_cam0)

            # normalize images
            img_source_cam0 = (img_source_cam0 - gray_mean) / gray_std
            img_target_cam0 = (img_target_cam0 - gray_mean) / gray_std

            if len(img_source_cam0.shape) == 2:
                img_source_cam0 = img_source_cam0[:, :, np.newaxis]
                img_source_cam0 = np.repeat(img_source_cam0, 3, -1)
                img_source_cam0 = torch.from_numpy(np.expand_dims(np.transpose(img_source_cam0, (2, 0, 1)), 0))

            if len(img_target_cam0.shape) == 2:
                img_target_cam0 = img_target_cam0[:, :, np.newaxis]
                img_target_cam0 = np.repeat(img_target_cam0, 3, -1)
                img_target_cam0 = torch.from_numpy(np.expand_dims(np.transpose(img_target_cam0, (2, 0, 1)), 0))

            if iscuda:
                img_source_cam0 = img_source_cam0.cuda()
                img_target_cam0 = img_target_cam0.cuda()

            # extract keypoints/descriptors for a single image
            xys0, desc0, scores0 = extract_multiscale(net, img_source_cam0, detector,
                                                      scale_f=args.scale_f,
                                                      min_scale=args.min_scale,
                                                      max_scale=args.max_scale,
                                                      min_size=args.min_size,
                                                      max_size=args.max_size,
                                                      verbose=False)

            # to cpu and sort
            kp0 = xys0.cpu().numpy()[:, :2]
            desc0 = desc0.cpu().numpy()
            scores0 = scores0.cpu().numpy()
            idxs0 = scores0.argsort()[-args.top_k or None:]

            # filter
            kp0 = kp0[idxs0]
            desc0 = desc0[idxs0]

            # extract keypoints/descriptors for a single image
            xys1, desc1, scores1 = extract_multiscale(net, img_target_cam0, detector,
                                                      scale_f=args.scale_f,
                                                      min_scale=args.min_scale,
                                                      max_scale=args.max_scale,
                                                      min_size=args.min_size,
                                                      max_size=args.max_size,
                                                      verbose=False)

            # to cpu and sort
            kp1 = xys1.cpu().numpy()[:, :2]
            desc1 = desc1.cpu().numpy()
            scores1 = scores1.cpu().numpy()
            idxs1 = scores1.argsort()[-args.top_k or None:]

            # filter
            kp1 = kp1[idxs1]
            desc1 = desc1[idxs1]

            # match descriptors
            matches = bf_matcher.match(desc0, desc1)

            # estimate pose
            translation, rot_matrix = estimate_pose_PnPRansac(kp0, kp1, matches, disparity, camMatrix, baseline)

            # pyquaternion uses w, x, y, z
            quaternion = Quaternion(matrix=rot_matrix)

            reloc_result.write(
                str(l[0]) + ' ' + str(l[1]) + ' ' + str(translation.squeeze()[0]) + ' ' + str(
                    translation.squeeze()[1]) + ' ' + str(translation.squeeze()[2]) + ' ' +
                str(quaternion[1]) + ' ' + str(quaternion[2]) + ' ' + str(quaternion[3]) + ' ' + str(
                    quaternion[0]) + '\n')

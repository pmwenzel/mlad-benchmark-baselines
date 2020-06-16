#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import os
import sys

sys.path.append(os.path.join('third_party', 'SuperGluePretrainedNetwork'))

import argparse
import cv2
import torch
from pyquaternion import Quaternion
from tqdm import tqdm
from utils import load_image, estimate_pose_PnPRansac, load_calibration
import numpy as np

from third_party.SuperGluePretrainedNetwork.models.matching import Matching
from third_party.SuperGluePretrainedNetwork.models.utils import frame2tensor

torch.set_grad_enabled(False)

# Based on third_party/SuperGluePretrainedNetwork/demo_superglue.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue.')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--dataset-path', type=str, required=True,
        help='path to dataset.')
    parser.add_argument(
        '--output-path', type=str, required=True,
        help='path to save the results.')
    parser.add_argument(
        '--test-sequence', type=int, default=0,
        help='test sequence, select either 0 or 1')

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # source sequence
    source_seq = 'recording_2020-04-07_10-20-32'
    if opt.test_sequence == 0:
        target_seq = 'recording_2020-03-24_17-45-31'  # test_sequence0
    elif opt.test_sequence == 1:
        target_seq = 'recording_2020-04-23_19-37-00'  # test_sequence1
    else:
        exit('Test sequence can either be 0 or 1.')
    tasks = ['easy', 'moderate', 'hard']

    # check if output folder exists
    if not os.path.isdir(opt.output_path):
        os.makedirs(opt.output_path)

    # folder source
    folder_source = os.path.join(opt.dataset_path, source_seq)
    # folder target
    folder_target = os.path.join(opt.dataset_path, target_seq)

    # initialize stereoBM object
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)

    # load camMatrix and baseline
    camMatrix, baseline = load_calibration(os.path.join(folder_source, 'Calibration'))

    for tsk in tasks:

        # reloc file
        reloc_file = os.path.join(opt.dataset_path, source_seq, 'RelocalizationFilesTest',
                                  'relocalizationFile_' + target_seq + '_' + tsk + '.txt')

        # Result file
        reloc_result = open(
            os.path.join(opt.output_path,
                         'relocalizationResult_superglue_eccv-challenge-' + target_seq + '_' + tsk + '.txt'),
            'w')

        # Process the file
        with open(reloc_file, 'r') as f:
            lines = f.readlines()
        lines = [l for l in lines if not l.startswith('#')]
        for line in tqdm(lines, total=len(lines)):
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

            # convert images to float32 and convert to torch tensor
            img_source_cam0 = frame2tensor(img_source_cam0, device)
            img_target_cam0 = frame2tensor(img_target_cam0, device)

            # Perform the matching.
            pred = matching({'image0': img_source_cam0, 'image1': img_target_cam0})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            assert len(mkpts0) == len(mkpts1)

            # create DMatch object
            matches = [cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _imgIdx=0, _distance=0) for idx in range(len(mkpts0))]

            # estimate pose
            translation, rot_matrix = estimate_pose_PnPRansac(mkpts0, mkpts1, matches, disparity, camMatrix, baseline)

            # pyquaternion uses w, x, y, z
            quaternion = Quaternion(matrix=rot_matrix)

            reloc_result.write(
                str(l[0]) + ' ' + str(l[1]) + ' ' + str(translation.squeeze()[0]) + ' ' + str(
                    translation.squeeze()[1]) + ' ' + str(translation.squeeze()[2]) + ' ' +
                str(quaternion[1]) + ' ' + str(quaternion[2]) + ' ' + str(quaternion[3]) + ' ' + str(
                    quaternion[0]) + '\n')

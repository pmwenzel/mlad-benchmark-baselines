import os
import sys

sys.path.append(os.path.join('third_party', 'd2-net'))

import argparse

import numpy as np
import scipy
import scipy.misc
import torch
from pyquaternion import Quaternion
from tqdm import tqdm
import cv2
from utils import load_image, estimate_pose_PnPRansac, load_calibration
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale


# Based on third_party/d2-net/extract_features.py
def extract_features_d2(model, image, multiscale):
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )

    # Remove gray scale mean
    mean_gray = np.array([114.81, 114.81, 114.81])
    input_image = input_image - mean_gray.reshape([3, 1, 1])

    with torch.no_grad():
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j

    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    # sort according to scores
    idxs = scores.argsort()[::-1]

    return keypoints[idxs], scores[idxs], descriptors[idxs]


# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True  # speedup

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--preprocessing', type=str, default=None,
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str, default=os.path.join(
        'third_party', 'd2-net', 'models', 'd2_tf.pth'),
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='path to dataset.'
)
parser.add_argument(
    '--test-sequence', type=int, default=0,
    help='test sequence, select either 0 or 1'
)

parser.set_defaults(use_relu=True)

args = parser.parse_args()

print(args)

# Creating CNN model
model = D2Net(
    model_file=args.model_file,
    use_relu=args.use_relu,
    use_cuda=use_cuda
)

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
        os.path.join(folder_source, 'relocalizationResult_d2-net_eccv-challenge-' + target_seq + '_' + tsk + '.txt'),
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

        keypoints0, _, descriptors0 = extract_features_d2(model, img_source_cam0, args.multiscale)
        keypoints1, _, descriptors1 = extract_features_d2(model, img_target_cam0, args.multiscale)

        # match descriptors
        matches = bf_matcher.match(descriptors0, descriptors1)

        # estimate pose
        translation, rot_matrix = estimate_pose_PnPRansac(keypoints0, keypoints1, matches, disparity, camMatrix,
                                                          baseline)

        # pyquaternion uses w, x, y, z
        quaternion = Quaternion(matrix=rot_matrix)

        reloc_result.write(
            str(l[0]) + ' ' + str(l[1]) + ' ' + str(translation.squeeze()[0]) + ' ' + str(
                translation.squeeze()[1]) + ' ' + str(translation.squeeze()[2]) + ' ' +
            str(quaternion[1]) + ' ' + str(quaternion[2]) + ' ' + str(quaternion[3]) + ' ' + str(
                quaternion[0]) + '\n')

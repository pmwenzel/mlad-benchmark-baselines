# MLAD-Benchmark-Baselines

This repository contains code to reproduce the baseline results for the [MLAD Challenge](https://sites.google.com/view/mlad-eccv2020/challenge).

## Getting started

We recommend Python 3.6+ for running the code. [Conda](https://docs.conda.io/en/latest/) is required. The conda environment requires CUDA 10.1 to be installed. You can install the environment with:

``
conda env create -f environment.yml
``

To clone the project and install the submodules please run the following:

```
git clone https://github.com/pmwenzel/mlad-benchmark-baselines.git
cd mlad-benchmark-baselines
git submodule update --init
```

## Pretrained models

To download the pretrained models for D2-Net please run the following:

```
mkdir third_party/d2-net/models
wget https://dsmn.ml/files/d2-net/d2_tf.pth -O third_party/d2-net/models/d2_tf.pth
``` 

## Downloading data

The dataset for the challenge can be downloaded [here](https://sites.google.com/view/mlad-eccv2020/challenge).
Please make sure to download the whole dataset, including reference, training, validation, and both test sequences. 

The code expects the following data structure after extraction of the files.

```
.
├── recording_2020-03-03_12-03-23
├── recording_2020-03-24_17-36-22
├── recording_2020-03-24_17-45-31
├── recording_2020-04-07_10-20-32
└── recording_2020-04-23_19-37-00
```

## Baselines

The baselines for our challenge are currently based on the following GitHub projects:

* [R2D2](https://github.com/naver/r2d2)
* [D2-Net](https://github.com/mihaidusmanu/d2-net)
* [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)

## Running the code

To run the baselines for `test_sequence0` run the following:

```
bash run_baselines.sh /path/to/dataset /path/to/output/folder 0
```

To run the baselines for `test_sequence1` run the following:

```
bash run_baselines.sh /path/to/dataset /path/to/output/folder 1
```

## Results

The result of each re-localization file will be saved as a `.txt` file in the format as expected to be submitted to the challenge. 

Each line of these text-files is saved in the following way: 

```
source_kf target_kf t_x t_y t_z q_x q_y q_z q_w
```

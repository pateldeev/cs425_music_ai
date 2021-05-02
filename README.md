# musicAi Backend
This repository contains backend code the [musicAi website](https://musicai.app). 
## About musicAi
[musicAi](https://projectmusicai.wordpress.com/about/) is projected created by Alden Bauman, Fahim Billah, Mir Hossain, Deev Patel & Ian Rinehart as a capstone project for CS 425 & CS 426 at UNR during the 2020-2021 school year. It aims to allow users of all backgrounds to leverage machine learning to generate new music.

## About Repository
This repository holds the backed Python/NodeJS code for musicAi. It supports running a LSTM based TensorFlow model to generate new music and interfacing with the results. It is meant to be fully compatible with the [front end](https://github.com/pateldeev/cs_426_final_site.git). The code in this repository also contains references to a MySQL database hosted on [freedb](https://freedb.tech/) where user login data and information is stored. The code also connects to the main musicAi website on [freehosting](https://freehosting.com) via FTP.

## Repository Structure
### local_node_server
This directory contains NodeJS code providing endpoints to kick off training jobs and get their status. This allows to front end to live stream training results as view old jobs. There are also endpoints to interface with Twitter and YouTube. On Twitter, we allow users to automatically generate posts with any generated songs. On YouTube, we allow users to download songs in a format that is compatible with training.
### ml_model
This directory contains the code for training and running the LSTM based ML model for generating new songs. The training is done with the help of [TensorFlow](https://www.tensorflow.org/). We have tested the training script on [python 3.8.5](https://www.python.org/downloads/release/python-380/) with [Tensorflow GPU version 2.4.1](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.1) linked to [cuda 11.0](https://developer.nvidia.com/cuda-11.0-download-archive) on [Ubuntu 20.04 LTS](https://releases.ubuntu.com/20.04/). We do not have plans on supporting training with any other configuration as there is only meant to be one training server that interfaces with the frontend code via a NodeJS server. We also make use of [python 3.9.0+](https://www.python.org/downloads/release/python-394/) to convert music between various file types necessary for user convenience.

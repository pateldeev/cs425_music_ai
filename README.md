# cs425_music_ai
Senior Project Repo

Developed with python3.8

Required Packages
* `pip3 install music21`
* `pip3 install tensorflow-gpu==2.3.0`

All the code is in `main.py`. 

It has some flags at the top (TODO: Convert these to command line flags):
* `USE_GPU`: If gpu should be used (gpu must be supported by tensorflow).
* `USE_CACHE`: If the the notes should be read from the cache file or produced from the input files
* `TRAIN_NETWORK`: If the network should be trained for if weights should be read from trained file.
* `TMP_DIR`: Directory to hold cached data and weights
* `CACHE_FILE`: Cache file where notes will be stored after input files have been processed.
* `INPUT_MIDI_FILES`: Input files. Will only be processed for notes if `USE_CACHE=False`.
* `LSTM_SEQ_LENGTH`: Sequence length for lstm layer. This is the size of the rolling history window.
* Training parameters. Valid if `TRAIN_NETWORK=True`
  * `NUM_EPOCHS`: Number of epochs to train for.
  * `BATCH_SIZE`: Batch size.
  * `TRAINING_WEIGHTS_FN`: Format of training checkpoint files for weights. Useful if training gets interrupted.
* `FINAL_WEIGHTS_FILE`: Weights file for trained model. 
  * If `TRAIN_NETWORK=True`: final weights will be written to this file.
  * If `TRAIN_NETWORK=False`: weights will be read from this file.
* `OUTPUT_NOTES`: Number of notes to output (length to output song).
* `OUTPUT_FILE`: Output file for generated song.

Output from training
```
dp@dp-XPS-15-9560:~/Downloads/cs425_music_ai/ml_model$ python3 main.py
2020-12-06 10:32:03.762088: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-12-06 10:32:05.096679: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2020-12-06 10:32:05.134801: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.135638: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1050 computeCapability: 6.1
coreClock: 1.493GHz coreCount: 5 deviceMemorySize: 3.95GiB deviceMemoryBandwidth: 104.43GiB/s
2020-12-06 10:32:05.135741: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-12-06 10:32:05.138782: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-12-06 10:32:05.141523: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2020-12-06 10:32:05.142096: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2020-12-06 10:32:05.144124: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2020-12-06 10:32:05.145482: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2020-12-06 10:32:05.149212: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2020-12-06 10:32:05.149395: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.150031: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.150613: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2020-12-06 10:32:05.150910: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-12-06 10:32:05.156943: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2799925000 Hz
2020-12-06 10:32:05.157361: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x53ed430 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-12-06 10:32:05.157380: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-12-06 10:32:05.235797: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.236308: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5feac40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-12-06 10:32:05.236327: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1050, Compute Capability 6.1
2020-12-06 10:32:05.236542: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.236797: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1050 computeCapability: 6.1
coreClock: 1.493GHz coreCount: 5 deviceMemorySize: 3.95GiB deviceMemoryBandwidth: 104.43GiB/s
2020-12-06 10:32:05.236834: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-12-06 10:32:05.236885: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-12-06 10:32:05.236921: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2020-12-06 10:32:05.236942: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2020-12-06 10:32:05.236961: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2020-12-06 10:32:05.236982: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2020-12-06 10:32:05.237001: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2020-12-06 10:32:05.237077: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.237398: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.237622: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Start
Python version:  3.8.5 (default, Jul 28 2020, 12:59:40) 
[GCC 9.3.0]
Version info: sys.version_info(major=3, minor=8, micro=5, releaselevel='final', serial=0)
Tensorflow version: 2.3.1
Tensorflow Devices: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:XLA_CPU:0', device_type='XLA_CPU'), PhysicalDevice(name='/physical_device:XLA_GPU:0', device_type='XLA_GPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
2020-12-06 10:32:05.237654: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-12-06 10:32:05.753160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-12-06 10:32:05.753191: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
2020-12-06 10:32:05.753199: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
2020-12-06 10:32:05.753380: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.753739: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-06 10:32:05.754002: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3580 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
WARNING:tensorflow:Layer lstm will not use cuDNN kernel since it doesn't meet the cuDNN kernel criteria. It will use generic GPU kernel as fallback when running on GPU
WARNING:tensorflow:Layer lstm_1 will not use cuDNN kernel since it doesn't meet the cuDNN kernel criteria. It will use generic GPU kernel as fallback when running on GPU
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 100, 512)          1052672   
_________________________________________________________________
lstm_1 (LSTM)                (None, 100, 512)          2099200   
_________________________________________________________________
lstm_2 (LSTM)                (None, 512)               2099200   
_________________________________________________________________
batch_normalization (BatchNo (None, 512)               2048      
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               131328    
_________________________________________________________________
activation (Activation)      (None, 256)               0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 256)               1024      
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 158)               40606     
_________________________________________________________________
activation_1 (Activation)    (None, 158)               0         
=================================================================
Total params: 5,426,078
Trainable params: 5,424,542
Non-trainable params: 1,536
_________________________________________________________________
None
Epoch 1/200
2020-12-06 10:32:13.943279: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2020-12-06 10:32:14.451063: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
43/43 [==============================] - 31s 712ms/step - loss: 5.2994
Epoch 2/200
43/43 [==============================] - 30s 699ms/step - loss: 4.8651
Epoch 3/200
43/43 [==============================] - 30s 708ms/step - loss: 4.7278
Epoch 4/200
43/43 [==============================] - 34s 802ms/step - loss: 4.6100
Epoch 5/200
43/43 [==============================] - 39s 905ms/step - loss: 4.9806
Epoch 6/200
43/43 [==============================] - 36s 833ms/step - loss: 4.8996
Epoch 7/200
43/43 [==============================] - 39s 912ms/step - loss: 4.6994
Epoch 8/200
43/43 [==============================] - 40s 936ms/step - loss: 4.6198
Epoch 9/200
43/43 [==============================] - 38s 892ms/step - loss: 4.6752
Epoch 10/200
43/43 [==============================] - 41s 954ms/step - loss: 4.6098
Epoch 11/200
43/43 [==============================] - 44s 1s/step - loss: 4.5716
Epoch 12/200
43/43 [==============================] - 38s 892ms/step - loss: 4.5372
Epoch 13/200
43/43 [==============================] - 47s 1s/step - loss: 4.4756
Epoch 14/200
43/43 [==============================] - 41s 958ms/step - loss: 4.4847
Epoch 15/200
43/43 [==============================] - 46s 1s/step - loss: 4.4645
Epoch 16/200
43/43 [==============================] - 42s 988ms/step - loss: 4.4474
Epoch 17/200
43/43 [==============================] - 44s 1s/step - loss: 4.4365
Epoch 18/200
43/43 [==============================] - 43s 989ms/step - loss: 4.4301
Epoch 19/200
43/43 [==============================] - 48s 1s/step - loss: 4.3922
Epoch 20/200
43/43 [==============================] - 51s 1s/step - loss: 4.3208
Epoch 21/200
43/43 [==============================] - 46s 1s/step - loss: 4.2698
Epoch 22/200
43/43 [==============================] - 39s 904ms/step - loss: 4.2246
Epoch 23/200
43/43 [==============================] - 47s 1s/step - loss: 4.2041
Epoch 24/200
43/43 [==============================] - 41s 961ms/step - loss: 4.1956
Epoch 25/200
43/43 [==============================] - 38s 877ms/step - loss: 4.1833
Epoch 26/200
43/43 [==============================] - 44s 1s/step - loss: 4.1324
Epoch 27/200
43/43 [==============================] - 32s 749ms/step - loss: 4.1225
Epoch 28/200
43/43 [==============================] - 32s 749ms/step - loss: 4.0932
Epoch 29/200
43/43 [==============================] - 33s 778ms/step - loss: 4.0811
Epoch 30/200
43/43 [==============================] - 35s 815ms/step - loss: 4.0545
Epoch 31/200
43/43 [==============================] - 35s 812ms/step - loss: 4.0321
Epoch 32/200
43/43 [==============================] - 35s 811ms/step - loss: 4.0272
Epoch 33/200
43/43 [==============================] - 35s 807ms/step - loss: 4.0054
Epoch 34/200
43/43 [==============================] - 35s 804ms/step - loss: 3.9618
Epoch 35/200
43/43 [==============================] - 35s 805ms/step - loss: 3.9549
Epoch 36/200
43/43 [==============================] - 35s 803ms/step - loss: 3.9398
Epoch 37/200
43/43 [==============================] - 35s 803ms/step - loss: 3.9098
Epoch 38/200
43/43 [==============================] - 35s 805ms/step - loss: 3.8911
Epoch 39/200
43/43 [==============================] - 34s 801ms/step - loss: 3.8805
Epoch 40/200
43/43 [==============================] - 30s 691ms/step - loss: 3.8669
Epoch 41/200
43/43 [==============================] - 30s 688ms/step - loss: 3.8422
Epoch 42/200
43/43 [==============================] - 31s 731ms/step - loss: 3.8264
Epoch 43/200
43/43 [==============================] - 31s 732ms/step - loss: 3.8034
Epoch 44/200
43/43 [==============================] - 31s 731ms/step - loss: 3.7956
Epoch 45/200
43/43 [==============================] - 34s 801ms/step - loss: 3.7699
Epoch 46/200
43/43 [==============================] - 30s 705ms/step - loss: 3.7707
Epoch 47/200
43/43 [==============================] - 30s 706ms/step - loss: 3.7426
Epoch 48/200
43/43 [==============================] - 30s 704ms/step - loss: 3.7892
Epoch 49/200
43/43 [==============================] - 30s 706ms/step - loss: 3.7330
Epoch 50/200
43/43 [==============================] - 30s 707ms/step - loss: 3.7087
Epoch 51/200
43/43 [==============================] - 30s 707ms/step - loss: 3.6909
Epoch 52/200
43/43 [==============================] - 30s 706ms/step - loss: 3.6755
Epoch 53/200
43/43 [==============================] - 30s 704ms/step - loss: 3.6632
Epoch 54/200
43/43 [==============================] - 30s 705ms/step - loss: 3.6394
Epoch 55/200
43/43 [==============================] - 30s 704ms/step - loss: 3.6217
Epoch 56/200
43/43 [==============================] - 30s 704ms/step - loss: 3.6090
Epoch 57/200
43/43 [==============================] - 30s 704ms/step - loss: 3.5929
Epoch 58/200
43/43 [==============================] - 30s 704ms/step - loss: 3.5897
Epoch 59/200
43/43 [==============================] - 30s 703ms/step - loss: 3.5749
Epoch 60/200
43/43 [==============================] - 30s 704ms/step - loss: 3.5737
Epoch 61/200
43/43 [==============================] - 30s 706ms/step - loss: 3.5472
Epoch 62/200
43/43 [==============================] - 30s 704ms/step - loss: 3.5309
Epoch 63/200
43/43 [==============================] - 30s 705ms/step - loss: 3.5165
Epoch 64/200
43/43 [==============================] - 30s 704ms/step - loss: 3.5144
Epoch 65/200
43/43 [==============================] - 30s 705ms/step - loss: 3.5019
Epoch 66/200
43/43 [==============================] - 30s 704ms/step - loss: 3.4875
Epoch 67/200
43/43 [==============================] - 30s 704ms/step - loss: 3.4766
Epoch 68/200
43/43 [==============================] - 30s 704ms/step - loss: 3.4683
Epoch 69/200
43/43 [==============================] - 30s 705ms/step - loss: 3.4369
Epoch 70/200
43/43 [==============================] - 30s 702ms/step - loss: 3.4485
Epoch 71/200
43/43 [==============================] - 30s 703ms/step - loss: 3.4286
Epoch 72/200
43/43 [==============================] - 30s 704ms/step - loss: 3.4116
Epoch 73/200
43/43 [==============================] - 30s 703ms/step - loss: 3.4028
Epoch 74/200
43/43 [==============================] - 30s 705ms/step - loss: 3.3880
Epoch 75/200
43/43 [==============================] - 30s 702ms/step - loss: 3.3684
Epoch 76/200
43/43 [==============================] - 30s 701ms/step - loss: 3.3719
Epoch 77/200
43/43 [==============================] - 30s 704ms/step - loss: 3.3534
Epoch 78/200
43/43 [==============================] - 30s 702ms/step - loss: 3.3426
Epoch 79/200
43/43 [==============================] - 30s 702ms/step - loss: 3.3272
Epoch 80/200
43/43 [==============================] - 30s 704ms/step - loss: 3.3236
Epoch 81/200
43/43 [==============================] - 30s 704ms/step - loss: 3.3130
Epoch 82/200
43/43 [==============================] - 30s 704ms/step - loss: 3.2874
Epoch 83/200
43/43 [==============================] - 30s 701ms/step - loss: 3.2880
Epoch 84/200
43/43 [==============================] - 30s 706ms/step - loss: 3.2899
Epoch 85/200
43/43 [==============================] - 30s 702ms/step - loss: 3.2536
Epoch 86/200
43/43 [==============================] - 30s 704ms/step - loss: 3.2425
Epoch 87/200
43/43 [==============================] - 30s 702ms/step - loss: 3.2442
Epoch 88/200
43/43 [==============================] - 30s 704ms/step - loss: 3.2339
Epoch 89/200
43/43 [==============================] - 30s 703ms/step - loss: 3.2085
Epoch 90/200
43/43 [==============================] - 30s 704ms/step - loss: 3.1875
Epoch 91/200
43/43 [==============================] - 30s 701ms/step - loss: 3.1903
Epoch 92/200
43/43 [==============================] - 30s 705ms/step - loss: 3.1689
Epoch 93/200
43/43 [==============================] - 30s 705ms/step - loss: 3.1469
Epoch 94/200
43/43 [==============================] - 30s 705ms/step - loss: 3.1383
Epoch 95/200
43/43 [==============================] - 30s 702ms/step - loss: 3.1386
Epoch 96/200
43/43 [==============================] - 30s 705ms/step - loss: 3.1059
Epoch 97/200
43/43 [==============================] - 30s 703ms/step - loss: 3.1095
Epoch 98/200
43/43 [==============================] - 30s 704ms/step - loss: 3.0769
Epoch 99/200
43/43 [==============================] - 30s 703ms/step - loss: 3.6812
Epoch 100/200
43/43 [==============================] - 30s 702ms/step - loss: 4.5371
Epoch 101/200
43/43 [==============================] - 30s 703ms/step - loss: 4.2555
Epoch 102/200
43/43 [==============================] - 30s 703ms/step - loss: 4.3543
Epoch 103/200
43/43 [==============================] - 30s 702ms/step - loss: 4.0160
Epoch 104/200
43/43 [==============================] - 30s 703ms/step - loss: 3.9024
Epoch 105/200
43/43 [==============================] - 30s 703ms/step - loss: 3.8156
Epoch 106/200
43/43 [==============================] - 30s 703ms/step - loss: 3.7713
Epoch 107/200
43/43 [==============================] - 30s 703ms/step - loss: 3.7365
Epoch 108/200
43/43 [==============================] - 30s 703ms/step - loss: 3.6590
Epoch 109/200
43/43 [==============================] - 30s 703ms/step - loss: 3.6353
Epoch 110/200
43/43 [==============================] - 30s 704ms/step - loss: 3.5219
Epoch 111/200
43/43 [==============================] - 30s 705ms/step - loss: 3.5214
Epoch 112/200
43/43 [==============================] - 30s 703ms/step - loss: 3.4717
Epoch 113/200
43/43 [==============================] - 30s 704ms/step - loss: 3.4303
Epoch 114/200
43/43 [==============================] - 30s 703ms/step - loss: 3.4011
Epoch 115/200
43/43 [==============================] - 30s 703ms/step - loss: 3.3534
Epoch 116/200
43/43 [==============================] - 30s 704ms/step - loss: 3.3221
Epoch 117/200
43/43 [==============================] - 30s 703ms/step - loss: 3.2891
Epoch 118/200
43/43 [==============================] - 30s 704ms/step - loss: 3.2409
Epoch 119/200
43/43 [==============================] - 30s 705ms/step - loss: 3.2006
Epoch 120/200
43/43 [==============================] - 30s 704ms/step - loss: 3.1599
Epoch 121/200
43/43 [==============================] - 30s 704ms/step - loss: 3.1331
Epoch 122/200
43/43 [==============================] - 30s 703ms/step - loss: 3.0806
Epoch 123/200
43/43 [==============================] - 30s 704ms/step - loss: 3.0607
Epoch 124/200
43/43 [==============================] - 30s 706ms/step - loss: 3.0334
Epoch 125/200
43/43 [==============================] - 30s 705ms/step - loss: 2.9968
Epoch 126/200
43/43 [==============================] - 30s 706ms/step - loss: 2.9660
Epoch 127/200
43/43 [==============================] - 30s 706ms/step - loss: 2.9112
Epoch 128/200
43/43 [==============================] - 30s 706ms/step - loss: 2.9064
Epoch 129/200
43/43 [==============================] - 30s 706ms/step - loss: 2.8763
Epoch 130/200
43/43 [==============================] - 30s 707ms/step - loss: 2.8092
Epoch 131/200
43/43 [==============================] - 30s 708ms/step - loss: 2.7967
Epoch 132/200
43/43 [==============================] - 30s 704ms/step - loss: 2.7474
Epoch 133/200
43/43 [==============================] - 30s 705ms/step - loss: 2.7238
Epoch 134/200
43/43 [==============================] - 30s 705ms/step - loss: 2.6789
Epoch 135/200
43/43 [==============================] - 30s 707ms/step - loss: 2.6516
Epoch 136/200
43/43 [==============================] - 30s 706ms/step - loss: 2.6066
Epoch 137/200
43/43 [==============================] - 30s 704ms/step - loss: 2.5642
Epoch 138/200
43/43 [==============================] - 30s 706ms/step - loss: 2.5324
Epoch 139/200
43/43 [==============================] - 30s 705ms/step - loss: 2.5086
Epoch 140/200
43/43 [==============================] - 30s 705ms/step - loss: 2.4457
Epoch 141/200
43/43 [==============================] - 30s 706ms/step - loss: 2.4330
Epoch 142/200
43/43 [==============================] - 30s 708ms/step - loss: 2.4417
Epoch 143/200
43/43 [==============================] - 30s 705ms/step - loss: 2.3412
Epoch 144/200
43/43 [==============================] - 30s 707ms/step - loss: 2.3145
Epoch 145/200
43/43 [==============================] - 30s 704ms/step - loss: 2.2771
Epoch 146/200
43/43 [==============================] - 30s 705ms/step - loss: 2.2210
Epoch 147/200
43/43 [==============================] - 30s 705ms/step - loss: 2.1957
Epoch 148/200
43/43 [==============================] - 30s 707ms/step - loss: 2.1744
Epoch 149/200
43/43 [==============================] - 30s 705ms/step - loss: 2.1408
Epoch 150/200
43/43 [==============================] - 30s 707ms/step - loss: 2.0989
Epoch 151/200
43/43 [==============================] - 30s 704ms/step - loss: 2.0515
Epoch 152/200
43/43 [==============================] - 30s 706ms/step - loss: 2.0478
Epoch 153/200
43/43 [==============================] - 30s 706ms/step - loss: 1.9637
Epoch 154/200
43/43 [==============================] - 30s 705ms/step - loss: 1.9533
Epoch 155/200
43/43 [==============================] - 30s 705ms/step - loss: 1.8853
Epoch 156/200
43/43 [==============================] - 30s 705ms/step - loss: 1.8611
Epoch 157/200
43/43 [==============================] - 30s 706ms/step - loss: 1.8303
Epoch 158/200
43/43 [==============================] - 30s 706ms/step - loss: 1.7885
Epoch 159/200
43/43 [==============================] - 30s 706ms/step - loss: 1.7245
Epoch 160/200
43/43 [==============================] - 30s 707ms/step - loss: 1.7079
Epoch 161/200
43/43 [==============================] - 30s 704ms/step - loss: 1.7135
Epoch 162/200
43/43 [==============================] - 30s 705ms/step - loss: 1.6224
Epoch 163/200
43/43 [==============================] - 30s 706ms/step - loss: 1.5994
Epoch 164/200
43/43 [==============================] - 30s 706ms/step - loss: 1.5689
Epoch 165/200
43/43 [==============================] - 31s 711ms/step - loss: 1.5221
Epoch 166/200
43/43 [==============================] - 41s 952ms/step - loss: 1.4715
Epoch 167/200
43/43 [==============================] - 34s 788ms/step - loss: 1.4531
Epoch 168/200
43/43 [==============================] - 32s 746ms/step - loss: 1.4071
Epoch 169/200
43/43 [==============================] - 32s 752ms/step - loss: 1.3769
Epoch 170/200
43/43 [==============================] - 32s 748ms/step - loss: 1.3610
Epoch 171/200
43/43 [==============================] - 32s 748ms/step - loss: 1.3235
Epoch 172/200
43/43 [==============================] - 32s 750ms/step - loss: 1.3005
Epoch 173/200
43/43 [==============================] - 32s 748ms/step - loss: 1.2451
Epoch 174/200
43/43 [==============================] - 32s 749ms/step - loss: 1.2047
Epoch 175/200
43/43 [==============================] - 32s 749ms/step - loss: 1.1990
Epoch 176/200
43/43 [==============================] - 32s 748ms/step - loss: 1.1540
Epoch 177/200
43/43 [==============================] - 32s 750ms/step - loss: 1.1163
Epoch 178/200
43/43 [==============================] - 37s 853ms/step - loss: 1.0838
Epoch 179/200
43/43 [==============================] - 32s 748ms/step - loss: 1.0645
Epoch 180/200
43/43 [==============================] - 32s 751ms/step - loss: 1.0151
Epoch 181/200
43/43 [==============================] - 32s 745ms/step - loss: 1.0244
Epoch 182/200
43/43 [==============================] - 33s 775ms/step - loss: 0.9566
Epoch 183/200
43/43 [==============================] - 33s 773ms/step - loss: 0.9283
Epoch 184/200
43/43 [==============================] - 35s 816ms/step - loss: 0.9401
Epoch 185/200
43/43 [==============================] - 32s 752ms/step - loss: 0.8841
Epoch 186/200
43/43 [==============================] - 32s 750ms/step - loss: 0.8607
Epoch 187/200
43/43 [==============================] - 34s 792ms/step - loss: 0.8432
Epoch 188/200
43/43 [==============================] - 33s 775ms/step - loss: 0.8065
Epoch 189/200
43/43 [==============================] - 33s 778ms/step - loss: 0.7673
Epoch 190/200
43/43 [==============================] - 33s 769ms/step - loss: 0.7794
Epoch 191/200
43/43 [==============================] - 33s 774ms/step - loss: 0.7271
Epoch 192/200
43/43 [==============================] - 33s 776ms/step - loss: 0.7237
Epoch 193/200
43/43 [==============================] - 33s 776ms/step - loss: 0.7028
Epoch 194/200
43/43 [==============================] - 33s 774ms/step - loss: 0.6548
Epoch 195/200
43/43 [==============================] - 33s 776ms/step - loss: 0.6734
Epoch 196/200
43/43 [==============================] - 33s 777ms/step - loss: 0.6457
Epoch 197/200
43/43 [==============================] - 33s 775ms/step - loss: 0.6031
Epoch 198/200
43/43 [==============================] - 33s 776ms/step - loss: 0.6152
Epoch 199/200
43/43 [==============================] - 34s 780ms/step - loss: 0.5878
Epoch 200/200
43/43 [==============================] - 33s 774ms/step - loss: 0.5598
Got trained model!
Generating note:  0
Generating note:  1
Generating note:  2
Generating note:  3
Generating note:  4
Generating note:  5
Generating note:  6
Generating note:  7
Generating note:  8
Generating note:  9
Generating note:  10
Generating note:  11
Generating note:  12
Generating note:  13
Generating note:  14
Generating note:  15
Generating note:  16
Generating note:  17
Generating note:  18
Generating note:  19
Generating note:  20
Generating note:  21
Generating note:  22
Generating note:  23
Generating note:  24
Generating note:  25
Generating note:  26
Generating note:  27
Generating note:  28
Generating note:  29
Generating note:  30
Generating note:  31
Generating note:  32
Generating note:  33
Generating note:  34
Generating note:  35
Generating note:  36
Generating note:  37
Generating note:  38
Generating note:  39
Generating note:  40
Generating note:  41
Generating note:  42
Generating note:  43
Generating note:  44
Generating note:  45
Generating note:  46
Generating note:  47
Generating note:  48
Generating note:  49
Generating note:  50
Generating note:  51
Generating note:  52
Generating note:  53
Generating note:  54
Generating note:  55
Generating note:  56
Generating note:  57
Generating note:  58
Generating note:  59
Generating note:  60
Generating note:  61
Generating note:  62
Generating note:  63
Generating note:  64
Generating note:  65
Generating note:  66
Generating note:  67
Generating note:  68
Generating note:  69
Generating note:  70
Generating note:  71
Generating note:  72
Generating note:  73
Generating note:  74
Generating note:  75
Generating note:  76
Generating note:  77
Generating note:  78
Generating note:  79
Generating note:  80
Generating note:  81
Generating note:  82
Generating note:  83
Generating note:  84
Generating note:  85
Generating note:  86
Generating note:  87
Generating note:  88
Generating note:  89
Generating note:  90
Generating note:  91
Generating note:  92
Generating note:  93
Generating note:  94
Generating note:  95
Generating note:  96
Generating note:  97
Generating note:  98
Generating note:  99
Generating note:  100
Generating note:  101
Generating note:  102
Generating note:  103
Generating note:  104
Generating note:  105
Generating note:  106
Generating note:  107
Generating note:  108
Generating note:  109
Generating note:  110
Generating note:  111
Generating note:  112
Generating note:  113
Generating note:  114
Generating note:  115
Generating note:  116
Generating note:  117
Generating note:  118
Generating note:  119
Generating note:  120
Generating note:  121
Generating note:  122
Generating note:  123
Generating note:  124
Generating note:  125
Generating note:  126
Generating note:  127
Generating note:  128
Generating note:  129
Generating note:  130
Generating note:  131
Generating note:  132
Generating note:  133
Generating note:  134
Generating note:  135
Generating note:  136
Generating note:  137
Generating note:  138
Generating note:  139
Generating note:  140
Generating note:  141
Generating note:  142
Generating note:  143
Generating note:  144
Generating note:  145
Generating note:  146
Generating note:  147
Generating note:  148
Generating note:  149
Generating note:  150
Generating note:  151
Generating note:  152
Generating note:  153
Generating note:  154
Generating note:  155
Generating note:  156
Generating note:  157
Generating note:  158
Generating note:  159
Generating note:  160
Generating note:  161
Generating note:  162
Generating note:  163
Generating note:  164
Generating note:  165
Generating note:  166
Generating note:  167
Generating note:  168
Generating note:  169
Generating note:  170
Generating note:  171
Generating note:  172
Generating note:  173
Generating note:  174
Generating note:  175
Generating note:  176
Generating note:  177
Generating note:  178
Generating note:  179
Generating note:  180
Generating note:  181
Generating note:  182
Generating note:  183
Generating note:  184
Generating note:  185
Generating note:  186
Generating note:  187
Generating note:  188
Generating note:  189
Generating note:  190
Generating note:  191
Generating note:  192
Generating note:  193
Generating note:  194
Generating note:  195
Generating note:  196
Generating note:  197
Generating note:  198
Generating note:  199
Generating note:  200
Generating note:  201
Generating note:  202
Generating note:  203
Generating note:  204
Generating note:  205
Generating note:  206
Generating note:  207
Generating note:  208
Generating note:  209
Generating note:  210
Generating note:  211
Generating note:  212
Generating note:  213
Generating note:  214
Generating note:  215
Generating note:  216
Generating note:  217
Generating note:  218
Generating note:  219
Generating note:  220
Generating note:  221
Generating note:  222
Generating note:  223
Generating note:  224
Generating note:  225
Generating note:  226
Generating note:  227
Generating note:  228
Generating note:  229
Generating note:  230
Generating note:  231
Generating note:  232
Generating note:  233
Generating note:  234
Generating note:  235
Generating note:  236
Generating note:  237
Generating note:  238
Generating note:  239
Generating note:  240
Generating note:  241
Generating note:  242
Generating note:  243
Generating note:  244
Generating note:  245
Generating note:  246
Generating note:  247
Generating note:  248
Generating note:  249
Generating note:  250
Generating note:  251
Generating note:  252
Generating note:  253
Generating note:  254
Generating note:  255
Generating note:  256
Generating note:  257
Generating note:  258
Generating note:  259
Generating note:  260
Generating note:  261
Generating note:  262
Generating note:  263
Generating note:  264
Generating note:  265
Generating note:  266
Generating note:  267
Generating note:  268
Generating note:  269
Generating note:  270
Generating note:  271
Generating note:  272
Generating note:  273
Generating note:  274
Generating note:  275
Generating note:  276
Generating note:  277
Generating note:  278
Generating note:  279
Generating note:  280
Generating note:  281
Generating note:  282
Generating note:  283
Generating note:  284
Generating note:  285
Generating note:  286
Generating note:  287
Generating note:  288
Generating note:  289
Generating note:  290
Generating note:  291
Generating note:  292
Generating note:  293
Generating note:  294
Generating note:  295
Generating note:  296
Generating note:  297
Generating note:  298
Generating note:  299
Generating note:  300
Generating note:  301
Generating note:  302
Generating note:  303
Generating note:  304
Generating note:  305
Generating note:  306
Generating note:  307
Generating note:  308
Generating note:  309
Generating note:  310
Generating note:  311
Generating note:  312
Generating note:  313
Generating note:  314
Generating note:  315
Generating note:  316
Generating note:  317
Generating note:  318
Generating note:  319
Generating note:  320
Generating note:  321
Generating note:  322
Generating note:  323
Generating note:  324
Generating note:  325
Generating note:  326
Generating note:  327
Generating note:  328
Generating note:  329
Generating note:  330
Generating note:  331
Generating note:  332
Generating note:  333
Generating note:  334
Generating note:  335
Generating note:  336
Generating note:  337
Generating note:  338
Generating note:  339
Generating note:  340
Generating note:  341
Generating note:  342
Generating note:  343
Generating note:  344
Generating note:  345
Generating note:  346
Generating note:  347
Generating note:  348
Generating note:  349
Generating note:  350
Generating note:  351
Generating note:  352
Generating note:  353
Generating note:  354
Generating note:  355
Generating note:  356
Generating note:  357
Generating note:  358
Generating note:  359
Generating note:  360
Generating note:  361
Generating note:  362
Generating note:  363
Generating note:  364
Generating note:  365
Generating note:  366
Generating note:  367
Generating note:  368
Generating note:  369
Generating note:  370
Generating note:  371
Generating note:  372
Generating note:  373
Generating note:  374
Generating note:  375
Generating note:  376
Generating note:  377
Generating note:  378
Generating note:  379
Generating note:  380
Generating note:  381
Generating note:  382
Generating note:  383
Generating note:  384
Generating note:  385
Generating note:  386
Generating note:  387
Generating note:  388
Generating note:  389
Generating note:  390
Generating note:  391
Generating note:  392
Generating note:  393
Generating note:  394
Generating note:  395
Generating note:  396
Generating note:  397
Generating note:  398
Generating note:  399
Generating note:  400
Generating note:  401
Generating note:  402
Generating note:  403
Generating note:  404
Generating note:  405
Generating note:  406
Generating note:  407
Generating note:  408
Generating note:  409
Generating note:  410
Generating note:  411
Generating note:  412
Generating note:  413
Generating note:  414
Generating note:  415
Generating note:  416
Generating note:  417
Generating note:  418
Generating note:  419
Generating note:  420
Generating note:  421
Generating note:  422
Generating note:  423
Generating note:  424
Generating note:  425
Generating note:  426
Generating note:  427
Generating note:  428
Generating note:  429
Generating note:  430
Generating note:  431
Generating note:  432
Generating note:  433
Generating note:  434
Generating note:  435
Generating note:  436
Generating note:  437
Generating note:  438
Generating note:  439
Generating note:  440
Generating note:  441
Generating note:  442
Generating note:  443
Generating note:  444
Generating note:  445
Generating note:  446
Generating note:  447
Generating note:  448
Generating note:  449
Generating note:  450
Generating note:  451
Generating note:  452
Generating note:  453
Generating note:  454
Generating note:  455
Generating note:  456
Generating note:  457
Generating note:  458
Generating note:  459
Generating note:  460
Generating note:  461
Generating note:  462
Generating note:  463
Generating note:  464
Generating note:  465
Generating note:  466
Generating note:  467
Generating note:  468
Generating note:  469
Generating note:  470
Generating note:  471
Generating note:  472
Generating note:  473
Generating note:  474
Generating note:  475
Generating note:  476
Generating note:  477
Generating note:  478
Generating note:  479
Generating note:  480
Generating note:  481
Generating note:  482
Generating note:  483
Generating note:  484
Generating note:  485
Generating note:  486
Generating note:  487
Generating note:  488
Generating note:  489
Generating note:  490
Generating note:  491
Generating note:  492
Generating note:  493
Generating note:  494
Generating note:  495
Generating note:  496
Generating note:  497
Generating note:  498
Generating note:  499
Predicted Notes: ['B-3', 'G3', 'B-3', 'D4', 'B-3', 'G3', 'B-3', '7.10', 'D4', 'B-3', '9.0', 'G3', 'B-3', '10.2', 'D4', 'B-3', '7.10.2', 'E-3', 'G3', 'B-3', 'D4', 'G5', 'E-3', 'G3', 'B-3', 'D4', '5.7.10', 'E-3', 'G3', 'B-3', 'C4', '3.9', 'E-3', 'G3', '10.2', 'B-3', 'D4', '5.9.0', 'E-3', 'F3', 'D6', 'A3', 'C4', '5.9', 'E-3', 'F3', 'A3', 'C4', 'E-3', 'F3', 'A3', 'C4', 'E-3', 'F3', 'A3', 'C4', '5.9.0', 'D3', 'F3', 'A3', 'C4', '5.9.0', 'D3', 'F3', 'A3', 'C4', '9.0', 'G6', 'D3', 'F3', 'F6', 'A3', 'C4', '7.11.2', 'D3', 'F3', 'A3', 'C3', '7.11.2', 'D3', 'C3', 'C3', 'C4', 'C3', 'C3', 'C5', 'C5', 'C3', 'C5', 'C5', 'C5', 'C3', 'C3', 'A4', 'C3', 'C3', 'C3', 'B-4', 'C3', 'C3', 'B-4', 'C3', 'C3', 'B-4', 'F3', 'E-5', 'D5', 'D5', 'C5', 'C5', 'A4', 'C3', 'C3', 'B-4', 'G5', 'E-5', 'C3', 'D5', 'C3', 'C3', 'C3', 'C3', 'C3', 'C3', 'C3', 'C3', 'D3', 'B-4', 'D3', 'E-3', 'E-5', 'B-4', 'B-2', 'C3', 'E-5', 'E-5', 'B-2', 'B-2', 'E-3', 'E-5', 'B-4', 'B-2', 'B-2', 'B-2', 'E-5', 'E-5', 'E-3', 'B-2', 'B-2', 'E-3', 'E-5', 'E-3', 'E-3', 'B-2', 'B-2', 'E-5', 'B-2', 'A2', 'G#3', 'B-2', 'A2', 'G2', 'E-5', 'F#4', 'B-2', 'G#2', 'G#4', 'F5', 'F#4', 'E-3', 'E-4', 'C#5', 'E-5', 'B-2', 'G#3', 'G#2', 'B-4', 'F#4', 'G4', 'B-4', 'D5', 'F4', 'B-4', 'B-2', 'F4', 'B-4', 'D5', 'F4', 'E-5', 'E-3', 'B-2', 'G#2', 'B-2', 'E-5', 'B-4', 'B-2', 'B-2', 'G2', 'B-2', 'B-2', 'E3', 'F#2', 'E-5', 'B-2', 'F#5', 'E-3', 'F5', 'B-4', 'E5', 'E5', 'G#4', 'F#4', 'B-4', 'G#5', 'G#4', 'G#4', 'F#4', 'E-5', 'E5', 'G#5', 'C#5', 'C#2', 'C#5', 'G#2', 'E2', 'C#5', '3.9', '1.7', 'C#2', 'F#5', 'G#5', 'E-2', 'E2', 'F#5', 'A5', 'E5', 'E5', 'E-2', 'E-2', '3.9', 'G#2', 'C#5', 'G#5', 'C#3', 'E-5', 'F#2', 'F#5', 'B5', 'B2', 'B-5', 'A5', 'E5', 'B-2', 'G#5', 'F#2', 'F#5', 'B5', 'G#2', 'F#2', 'C#6', 'G#5', 'E5', 'A2', 'G#5', '1.6', 'G#2', '1.6', 'G#2', 'G#4', 'C#6', 'C#6', 'E6', 'A5', 'B2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'G#5', 'E-2', 'C#2', 'G#5', 'G#5', 'E-2', 'C#3', 'E-5', 'E5', 'E-2', 'C#5', 'B2', 'C#3', 'F#2', 'F#2', 'B2', '3.9', '1.7', 'C#2', 'F#5', 'G#5', 'E-2', 'E2', 'F#5', 'A5', 'E5', 'E5', 'E-2', 'E-2', '3.9', 'G#2', 'C#5', 'G#5', 'C#3', 'E-5', 'F#2', 'F#5', 'E-5', 'B2', 'B-5', 'E5', 'E5', 'B-2', 'G#5', 'G#5', 'F#5', 'E-5', 'E-5', 'E5', 'G#5', 'E-2', 'E-2', 'E5', 'C#2', 'G#5', 'C#5', 'G#5', 'C#3', 'C#5', 'C#6', 'E5', 'F#2', 'B2', '3.9', '1.7', 'C#2', 'G#5', 'G#5', 'E-2', 'E2', 'A5', 'A5', 'E5', 'E5', 'E-2', 'C#2', '3.9', 'G#2', 'C#5', 'G#5', 'C#3', 'E-5', 'F#2', 'F#5', 'B5', 'B2', 'B-5', 'E5', 'E5', 'B-2', 'G#5', 'F#2', 'C#6', 'B2', 'G#5', 'E5', 'C#6', 'G#5', 'E5', 'G#4', 'B2', 'G#2', 'E5', 'E3', 'F#2', 'B2', '0.6', 'G#2', 'E-4', 'G#2', 'B-2', 'B5', 'B5', 'A5', 'E-2', 'E-2', 'F#2', 'F#2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'E-2', 'G#5', 'E5', 'C#2', 'G#5', 'G#5', 'C#3', 'F#3', 'C#5', 'G#5', 'B2', 'F#2', 'B2', 'E3', 'F#2', 'F#2', 'C#2', 'G#2', 'F#2', 'F#2', 'C#2', 'G#2', 'C#3', 'F#2', 'B2', 'C#2', 'F#2', 'E-2', 'E-2', 'C#2', 'G#2', 'E-2', 'E-2', 'B2', 'C#2', 'E-2', 'E2', 'E-2', 'C#2', 'G#2', 'C#3', 'F#2', 'B2', 'B-2', 'F#2', 'E2', 'E-2', 'G#2', 'C#3', 'F#2', 'B2', 'C#2', 'E-2', 'G#2', 'E-2', 'C#2', 'G#2', 'C#3', 'F#2', 'B2', 'C#2', 'F#2', 'G#2', 'A2', 'G#2', 'F#2', 'C#3', 'F#3', 'B2', 'C#2', 'F#2', 'G#2', 'A2', 'G#2', 'F#2', 'C#3', 'F#3', 'B2', 'E3', 'F#2', 'G#2', 'A2', 'G#2', 'F#2', 'C#3', 'F#3', 'B2', 'C#2', 'F#2', 'F#2']
Saved prediciton to:  model_output.mid
End
```

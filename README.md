# cnn_implementation
A simple implementation of CNN using Tensorflow.

## Requirements
0. Python 3.x
1. <a href="https://tensorflow.org">Tensorflow 1.3</a>
2. Numpy 1.11.3
3. PIL 5.1.0

## Installation
1. Clone this repository.
```
git clone https://github.com/ShiChenAI/cnn_implementation.git
```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
```
pip install -r requirements.txt  # or make install
```

## Usage
### Configuration
First configure your global parameters in **global.conf** as follows:
- **train_data_dir**: directory of training data.
- **eval_data_dir**: directory of evaluation data.
- **num_class**: number of classes to be classified.
- **resize_image_height**: resize the input images with the height.
- **resize_image_width**: resize the input images with the width.
- **chnnels**: channels of input images.
- **batch_size**: batch size in each step.
- **train_tfrecord_dir**: directory of training tfrecord.
- **train_data_count**: count of training data.
- **max_steps**: max steps during training.
- **eval_log_dir**: directory of evaluation log.
- **eval_tfrecord_dir**: directory of evaluation tfrecord.
- **eval_data_count**: count of evaluation data.
- **model_dir**: directory of the trained models.
- **moving_average_decay**: parameter of moving average decay.
- **num_epochs_per_decay**: parameter of number of epochs per decay.
- **learning_rate_decay_factor**: parameter of learning rate decay factor.
- **initial_average_decay**: parameter of initial average decay.
- **tower_name**: tower name.
- **keep_prob**: keep probability in dropout layer.

### Customizing network architecture
You can customize your network architecture using **network.json** with the layer names (**"layers"**), layers weights (**"weights"**) and biases (**"biases"**) as follows: 
```
{"layers": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "fc3"],
 "weights": 
    {"wconv1": [11, 11, 3, 64], 
     "wconv2": [5, 5, 64, 192],
     "wconv3": [3, 3, 192, 384],
     "wconv4": [3, 3, 384, 256], 
     "wconv5": [3, 3, 256, 256], 
     "wfc1": [12544, 4096], 
     "wfc2": [4096, 4096],
     "wfc3": [4096, 10]}, 
 "biases": 
    {"bconv1": [64], 
     "bconv2": [192],  
     "bconv3": [384], 
     "bconv4": [256], 
     "bconv5": [256], 
     "bfc1": [4096], 
     "bfc2": [4096],
     "bfc3": [10]}}
``` 

### Training model

``` 
python train_model.py
``` 
You will get:
``` 
2018-12-12 20:54:18.709988: step 0, loss = 7.31 (78.9 examples/sec; 0.634 sec/batch)
2018-12-12 20:54:57.119095: step 1, loss = 7.29 (78.9 examples/sec; 0.634 sec/batch)
``` 
The model will be evaluated in each 100 steps:
``` 
2018-12-12 21:41:39.556357: precision @ 1 = 1.000
``` 

### Visualization in TensorBoard
To start Tensorflow, run the following command on the console:
``` 
#!bash

tensorboard --logdir=./model
``` 

### Prediction
``` 
python predict_inputs.py --input_img ./data/1.png
``` 
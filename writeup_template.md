# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "nVidia_model"
[image2]: ./examples/cinter.png "Center Image"
[image3]: ./examples/left.png "Left Image"
[image4]: ./examples/right.png "Right Image"
[image5]: ./examples/flip.png "Flip Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results
* result.mp4 visualizing result

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer and Cropping2D. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98 and 105). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 85). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The model used 3 epohs and batch_size = 64.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the ready data set. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The finall architecture is based on nVidia nn.
![alt text][image1]

In this model i add 2 dropout layers and it helps with overfitting problem.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 
The vehicle was not in the center, to improve the driving behavior in these cases, I think we need more data for this part of road.

#### 2. Final Model Architecture

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


#### 3. Creation of the Training Set & Training Process

Here is an example image of center lane driving:

![alt text][image2]

to increase the data I used left and right cameras :

![alt text][image3]
![alt text][image4]

I also flipped images. For example, here is an image that has then been flipped:

![alt text][image5]

Preprocessing data include 3 steps:
* convert BGR to RGB
* normalization - (x / 255.0) - 0.5
* cropping - cropping=((50,20), (0,0)). In this step we drop information which does not improve network performance

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3.I used an adam optimizer so that manually training the learning rate wasn't necessary.

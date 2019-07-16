import numpy as np
import csv
import cv2
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from math import ceil
from sklearn.utils import shuffle


DATA_PATH = "data/"
IMAGE_PATH = DATA_PATH + "IMG/"
LEFT_IMAGE_ANGLE_CORRECTION = 0.20
RIGHT_IMAGE_ANGLE_CORRECTION = -0.20
BATCH_SIZE = 64 
EPOCHS = 3


def read_csv_data(data_path=DATA_PATH):
    csv_data = []
    with open(data_path + 'driving_log.csv') as csv_file:

        csv_reader = csv.reader(csv_file)
        # Skipping the headers
        next(csv_reader, None)
        for line in csv_reader:
            csv_data.append(line)
    
    return csv_data


def preprocess_image(image):
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return colored_image


def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    batch_size = int(batch_size / 4)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            print(len(batch_samples))
            for batch_sample in batch_samples:
                # original image
                name = IMAGE_PATH + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                
                if center_image is not None:
                    images.append(preprocess_image(center_image))
                    angles.append(center_angle)

                    # flip image
                    flip_img = images.append(np.fliplr(center_image))
                    flip_angle = - float(batch_sample[3])
                    angles.append(flip_angle)
                
                    # Processing the left image
                    name_left = IMAGE_PATH + batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name_left)
                    if left_image is not None:
                        images.append(preprocess_image(left_image))
                        angles.append(center_angle + LEFT_IMAGE_ANGLE_CORRECTION)

                    # Processing the right image
                    name_right = IMAGE_PATH + batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name_right)
                    if right_image is not None:
                        images.append(preprocess_image(right_image))
                        angles.append(center_angle + RIGHT_IMAGE_ANGLE_CORRECTION)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


data = read_csv_data()
train_samples, validation_samples = train_test_split(data, test_size=0.2)


def nvidia_model():
    ch, row, col = 3, 160, 320  # Trimmed image format

    model = Sequential()

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
    return model

model = nvidia_model()

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/BATCH_SIZE),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/BATCH_SIZE),
            epochs=EPOCHS, verbose=1)
model.save('model.h5')

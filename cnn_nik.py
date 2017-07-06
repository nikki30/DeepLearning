# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Adding the layers
#Step 1 :Convolution

#Adding 32 feature detectors with 3 cross 3 matrices
#Input_shape is for the image. 3 if it is a coloured image and 1 if it is black and white. And the other two are pixels.
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation = "relu"))

#Next we apply pooling on each of these feature maps so that we get lesser size and fewer nodes so it's easy to  compute
#Step 2: Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding another Convolution Layer
classifier.add(Convolution2D(32,3,3,activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding another CL with 64 feature maps is common and helps increase accuracy!

#Next, Flattening: Taking all pooled feature maps and creating one huge single vector containing all single cells of each pooled map
#So these high numbers generated after the first step (which are carried forward by the poolnig step) are specific features of the image which we retain
#What If we directly flatten the input image to generate the input vector. We dont do that cuz we'll get one large vector which has each of those pixels, independent of each other. 
#Instead, we perform the above steps, and that way you get info about the spatial structure.
#Basically, each feature map contains info about a specific info which is then put as one element in the vector

#Step 3: Flatten
classifier.add(Flatten())

#Step 4 :Fully connected
#We have a lot of input nodes. So we shall chumma pick 128
classifier.add(Dense(output_dim = 128, activation = "relu" ))
classifier.add(Dense(output_dim = 1, activation = "sigmoid" ))

#Step 5: Compile CNN: Since we have a binary outcome, we pick binary cross entropy
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

#Image augmentation so that we prevent overfitting (gettning good acccuracy on training set and less on test set)
#Look for image preprocessing in the Keras Documentation site (https://keras.io/preprocessing/image/)and use the ready to use code under flow_from_directory

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, #number of images in train
        epochs=25, 
        validation_data= test_set,
        validation_steps=2000) #number of images in test
        
#Making new predictions
#import numpy so we can use one of its module to preprocess the image we are going to load so it can be accepted by th predict method we will use

import numpy as np
from keras.preprocessing import image 

test_image = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 0:
    prediction = 'cat'
else :
    prediction = 'dog'
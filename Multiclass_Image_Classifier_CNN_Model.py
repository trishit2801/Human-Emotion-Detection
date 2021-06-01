#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# In[53]:


class_names = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']


# In[4]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


train_generator = train_datagen.flow_from_directory('/Users/trishit/Code/Udemy/Tensorflow-Bootcamp/Human-Emotion-Detection/dataset/train', 
                                                    batch_size = 32, target_size = (64,64), class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory('/Users/trishit/Code/Udemy/Tensorflow-Bootcamp/Human-Emotion-Detection/dataset/test', 
                                                   batch_size = 32, target_size = (64,64), class_mode = 'categorical')


# In[7]:


test_generator.class_indices


# In[8]:


# Creating the Image Classifier CNN Model
cnn = tf.keras.models.Sequential()

#Convolution and Max Pooling layer 1
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Convolution and Max Pooling layer 2
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Convolution and Max Pooling layer 3
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Converting the multi-dimension image data array to 1d array
cnn.add(tf.keras.layers.Flatten())

#Fully Connected layer 1
cnn.add(tf.keras.layers.Dense(128, activation='relu'))

#Fully Connected layer 2
cnn.add(tf.keras.layers.Dense(64, activation='relu'))

#output layer -> 7 Neurons for 7 different classes
#activation function used for multiclass classification is softmax, for binary use sigmoid as activation fxn
cnn.add(tf.keras.layers.Dense(7, activation='softmax'))


# In[11]:


# Defining the optimizer and loss function
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[12]:


#Training the CNN Model
cnn.fit(x=train_generator, validation_data=test_generator, epochs=30)


# In[70]:


# Using the Model to make Predictions
def predictEmotion(filepath):
    
    img1 = image.load_img(filepath, target_size = (64,64))
    
    plt.imshow(img1)
    
    Y = image.img_to_array(img1, dtype=int)
    X = np.expand_dims(Y, axis=0)
    
    result = cnn.predict(X)
    index = np.argmax(result)
    
    #print(index)
    
    print("Predicted Emotion: ", class_names[index])
        


# In[71]:


# Making Predictions of random images


# In[141]:


predictEmotion(r"/Users/trishit/Code/Udemy/Tensorflow-Bootcamp/Human-Emotion-Detection/prediction_images/14.jpg")


# In[ ]:





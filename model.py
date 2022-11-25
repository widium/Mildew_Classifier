# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/18 19:29:26 by ebennace          #+#    #+#              #
#    Updated: 2022/11/25 14:52:22 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf

from constant import IMAGE_SHAPE
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from keras import Model

# **************************************************************************** #

def Create_transfert_learning_Model(dim_prediction : int)->Model:
  
    vgg = tf.keras.applications.VGG19(include_top=False, 
                                      input_shape=IMAGE_SHAPE,
                                      weights='imagenet')
    for layer in vgg.layers:
      layer.trainable = False

    last_conv_layer = vgg.output

    flatten_layer = Flatten(name='flatten_layer')(last_conv_layer)
    fully_connected = Dense(4096, name='fully_connected')(flatten_layer) 
    prediction = Dense(dim_prediction, activation='sigmoid', name='lamioude')(fully_connected)


    model = Model(inputs=vgg.input, outputs=prediction)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


# def load_vgg(IMAGE_SHAPE : list):

#   vgg = tf.keras.applications.VGG19(include_top=False, 
#                                     input_shape=IMAGE_SHAPE,
#                                     weights='imagenet')
#   for layer in vgg.layers:
#     layer.trainable = False

#   return (vgg)

# class Mildew_Classifier(Model):

#     def __init__(self):
#       super(Mildew_Classifier, self).__init__()
#       self.vgg = load_vgg()
#       self.flatten = Flatten(name='flatten_layer')
#       self.dense = Dense(4096, name='fully_connected')
#       self.prediction = Dense(2, activation="softmax")
    
#     def build():
      
#     def call(self, input):
      
#       x = self.vgg(input)
#       x = self.flatten(x)
#       x = self.dense(x)
#       output = self.prediction(x)
      
#       return (output)
      
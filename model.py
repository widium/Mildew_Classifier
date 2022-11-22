# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/18 19:29:26 by ebennace          #+#    #+#              #
#    Updated: 2022/11/22 18:41:23 by ebennace         ###   ########.fr        #
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

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/18 19:20:48 by ebennace          #+#    #+#              #
#    Updated: 2022/11/18 19:24:12 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from keras.layers import Layer
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from augmentator import RandomSaturation
from augmentator import RandomBrightness
from augmentator import RandomCentralCrop
from augmentator import RandomColorShifting

# **************************************************************************** #

class Data_Augmentation(Layer):
    
    # ************************************** #
    def __init__(self, nbr_of_outputs, **kwargs):
        
        super().__init__(**kwargs)
        self.nbr_of_outputs = nbr_of_outputs
        self.set = list()
        self.Flip = RandomFlip("horizontal_and_vertical")
        self.Rotation = RandomRotation(0.8)
        self.Saturate = RandomSaturation(lower=0.5, higher=1.5)
        self.Bright = RandomBrightness(ratio=0.5)
        self.Crop = RandomCentralCrop()
        self.ColorShifting = RandomColorShifting(ratio=0.2)
    
    # ************************************** #
    def generator_images(self, img):
        
        for nbr in range(self.nbr_of_outputs):
            self.set.append(self.Flip(img))
            self.set.append(self.Rotation(img))
            self.set.append(self.Saturate(img))
            self.set.append(self.Bright(img))
            self.set.append(self.Crop(img))
            self.set.append(self.ColorShifting(img))
    
    # ************************************** #
    def call(self, img)-> list:
        self.generator_images(img)
        return (self.set)
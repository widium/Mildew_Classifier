# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    augmentator.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/18 19:21:44 by ebennace          #+#    #+#              #
#    Updated: 2022/11/25 11:06:12 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from keras.layers import RandomFlip
from keras.layers import RandomRotation

from .distortion_layer import RandomSaturation
from .distortion_layer import RandomBrightness
from .distortion_layer import RandomCrop
from .distortion_layer import RandomColorShifting

# **************************************************************************** #

class Image_Augmentation:
    
    # ************************************** #
    def __init__(self, nbr_augmented):
        
        self.nbr_augmented = nbr_augmented
        self.Flip = RandomFlip("horizontal_and_vertical")
        self.Rotation = RandomRotation(0.8)
        self.Saturate = RandomSaturation(lower=0.5, higher=1.5)
        self.Bright = RandomBrightness(ratio=0.5)
        self.Crop = RandomCrop()
        self.ColorShifting = RandomColorShifting(ratio=0.2)
    
    # ************************************** #
    def __call__(self, img):
        
        set = list()
        set.append(img)
        
        for nbr in range(self.nbr_augmented):
            set.append(self.Flip(img))
            set.append(self.Rotation(img))
            set.append(self.Saturate(img))
            set.append(self.Bright(img))
            set.append(self.Crop(img))
            set.append(self.ColorShifting(img))
        return (set)
    
    # ************************************** #
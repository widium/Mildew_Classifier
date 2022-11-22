# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    distortion_layer.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/18 19:20:48 by ebennace          #+#    #+#              #
#    Updated: 2022/11/22 16:48:15 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from keras.layers import Resizing
from tensorflow.image import random_saturation
from tensorflow.image  import random_brightness
from tensorflow.image  import random_crop
from tensorflow.image  import random_hue
from keras.layers import Layer

from constant import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_COLOR

# **************************************************************************** #

class RandomSaturation(Layer):
    
    def __init__(self, lower, higher, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.higher = higher
        
    def call(self, img):
        saturated = random_saturation(img, self.lower, self.higher)
        return (saturated)

# **************************************************************************** #

class RandomBrightness(Layer):

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        
    def call(self, img):
        bright = random_brightness(img, self.ratio)
        return (bright)

# **************************************************************************** #  
    
class RandomCrop(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Resize = Resizing(224, 224)
        
    def call(self, img):
        croped = random_crop(value=img, size=(IMAGE_HEIGHT - 125, IMAGE_WIDTH - 125, IMAGE_COLOR))
        img_resize = self.Resize(croped)
        return (img_resize)

# **************************************************************************** #
      
class RandomColorShifting(Layer):
    
    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        
    def call(self, img):
        image_hue = random_hue(img, self.ratio)
        return (image_hue)

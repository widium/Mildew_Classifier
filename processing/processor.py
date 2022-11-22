# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    processor.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/22 17:30:03 by ebennace          #+#    #+#              #
#    Updated: 2022/11/22 17:42:33 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Layer
from keras.layers import Resizing
from keras.layers import Rescaling

# **************************************************************************** #

class Image_Processing(Layer):
    
    # **************************************************************************** #
    
    def __init__(self, IMAGE_HEIGHT=224, IMAGE_WIDTH=224, IMAGE_COLOR=3, **kwargs):
        super().__init__(**kwargs)
        self.height =IMAGE_HEIGHT
        self.width = IMAGE_WIDTH
        self.color = IMAGE_COLOR
        self.Resizer = Resizing(height=self.height, width=self.width)
        self.Rescaler = Rescaling(1./255)
        
    # **************************************************************************** #
    
    def call(self, path : str):
        
        img = load_img(path)
        img = self.Resizer(img)
        img = img_to_array(img)
        img = self.Rescaler(img)
        return (img)

    # **************************************************************************** #
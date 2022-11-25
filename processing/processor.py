# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    processor.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/22 17:30:03 by ebennace          #+#    #+#              #
#    Updated: 2022/11/25 14:25:46 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.layers import Resizing
from keras.layers import Rescaling

# **************************************************************************** #

class Image_Processing:
    
    # **************************************************************************** #
    
    def __init__(self, IMAGE_HEIGHT=224, IMAGE_WIDTH=224, IMAGE_COLOR=3):
        self.height =IMAGE_HEIGHT
        self.width = IMAGE_WIDTH
        self.color = IMAGE_COLOR
        self.Resizer = Resizing(height=self.height, width=self.width)
        self.Rescaler = Rescaling(1./255)
        
    # **************************************************************************** #
    
    def __call__(self, path : str):
        
        img = load_img(path)
        img = img_to_array(img)
        img = self.Resizer(img)
        img = self.Rescaler(img)
        return (img)

    # **************************************************************************** #
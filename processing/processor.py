# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    processor.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/22 10:02:28 by ebennace          #+#    #+#              #
#    Updated: 2022/11/22 10:14:50 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

from keras.layers import Rescaling
from keras.layers import Resizing

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from os import listdir

from tqdm import tqdm

from .augmentator import Image_Augmentation

# **************************************************************************** #

class Image_Processing:
    
    # **************************************************************************** #
    
    def __init__(self, IMAGE_HEIGHT=224, IMAGE_WIDTH=224, nbr_augmentation=10, train_size=0.90, dev_size=0.30):
        self.height = IMAGE_HEIGHT
        self.width = IMAGE_WIDTH
        self.nbr_augmentation = nbr_augmentation
        self.train_size = train_size
        self.dev_size = dev_size
        self.train_set = list()
        self.dev_set = list()
        self.test_set = list()
        self.Resizer = Resizing(224, 224)
        self.Rescaler = Rescaling(1./255)
        self.Augmentator = Image_Augmentation(nbr_augmentation)

    # **************************************************************************** #
    
    def load_img(self, path : str):
        
        img = load_img(path)
        img = self.Resizer(img)
        img = img_to_array(img)
        img = self.Rescaler(img)
        return (img)

    # **************************************************************************** #
    
    def split_data_in_set(self, set : tuple):
        
        train_random = np.random.uniform(0, 1)
        dev_random = np.random.uniform(0, 1)
        
        if (train_random < self.train_size):
            self.train_set.append(set)
        else :
            if (dev_random < self.dev_size):
                self.dev_set.append(set)
            else :
                self.test_set.append(set)
        
    # **************************************************************************** #
        
    def labeling_img_directory(self, directory : str, classe : int) :

        for filename in tqdm(listdir(directory)):

            path = directory + '/' + filename
            img = self.load_img(path)
            augmented_imgs = self.Augmentator(img)
            for augmented in augmented_imgs:
                self.split_data_in_set((augmented, classe))
    
    # **************************************************************************** #
    
    def create_X_Y(self, list_tuple : list):
    
        X = list()
        Y = list()
        
        for tuple in list_tuple :
            X.append(tuple[0])
            Y.append(tuple[1])

        X = np.array(X)
        Y = np.array(Y)
        Y = Y.reshape(Y.shape[0], 1)
    
        return (X, Y)

    # **************************************************************************** #
    
    def output(self, healthy_dir_name : str,  disease_dir_name : str):
        
        self.labeling_img_directory(directory=disease_dir_name, classe=1)
        self.labeling_img_directory(directory=healthy_dir_name, classe=0)
        
        X_train, Y_train = self.create_X_Y(self.train_set)
        X_dev, Y_dev = self.create_X_Y(self.dev_set)
        X_test, Y_test = self.create_X_Y(self.test_set)
        
        print(X_train.shape, Y_train.shape)
        print(X_dev.shape, Y_dev.shape)
        print(X_test.shape, Y_test.shape)
        
        return (X_train, Y_train,
                X_dev, Y_dev,
                X_test, Y_test)
        
    # **************************************************************************** #
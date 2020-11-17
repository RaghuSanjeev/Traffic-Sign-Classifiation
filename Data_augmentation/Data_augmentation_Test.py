"""
@author: Raghu Sanjeev

Image shifts via the width_shift_range and height_shift_range arguments.
Image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument
Image brightness via the brightness_range argument.
Image zoom via the zoom_range argument.
"""

from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
from skimage import io
import os
from PIL import Image



####################################################################
#Multiple images.
#Manually read each image and create an array to be supplied to datagen via flow method
dataset = []
image_directory = 'Traffic_signs/'
SIZE = 128


#####################################################################
#Multiclass. Read dirctly from the folder structure using flow_from_directory
#Creates 32 images for each class.


def data_augment_nearest_mode():
    print("data_augment_nearest_mode- Enter")


    # Construct an instance of the ImageDataGenerator class
    # Pass the augmentation parameters through the constructor.
    datagen = ImageDataGenerator(rotation_range=45,  # Random rotation between 0 and 45
                                 width_shift_range=0.2,  # % shift
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')  # Other values nearest, constant, reflect, wrap

    i = 0
    for batch in datagen.flow_from_directory(directory='./Traffic_Signs/',
                                             batch_size=16,
                                             target_size=(256, 256),
                                             color_mode="rgb",
                                             save_to_dir='augmented',
                                             save_prefix='aug_nearest',
                                             save_format='png'):
        #print("batch= {}".format(batch))
        i += 1
        if i > 31:
            break



def data_augment_constant_mode():
    print("data_augment_constant_mode- Enter")
    datagen = ImageDataGenerator(rotation_range=45,     #Random rotation between 0 and 45
                                 width_shift_range=0.2,   #% shift
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='constant',
                                 cval=125) #Other values nearest, constant, reflect, wrap

    i = 0
    for batch in datagen.flow_from_directory(directory='./Traffic_Signs/',
                                             batch_size=16,
                                             target_size=(256, 256),
                                             color_mode="rgb",
                                             save_to_dir='augmented',
                                             save_prefix='aug_constant',
                                             save_format='png'):
        #print("batch= {}".format(batch))
        i += 1
        if i > 31:
            break



data_augment_constant_mode()
data_augment_nearest_mode()
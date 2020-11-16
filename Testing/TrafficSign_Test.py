import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# ***********   Changes to Support Linux ************************
import sys
from sys import platform
# **************************************************************


print ('Tensorflow version = {}'.format(tf.__version__))
print ('OpenCV version = {}'.format(cv2.__version__))

################# Parameters #####################

dataset_path = "datasets" # folder with all the class folders
test_path_dir = "../datasets/Testing/all_classes"

PICKLED_MODEL_FILE = "../TSign_Classification_Training/Belgium_Traffic_OpenCV_model_trained.p"
MODEL_FILE_H5 = '../TSign_Classification_Training/Belgium_Traffic_OpenCV_model_trained.h5'


# PROBABLITY THRESHOLD
#PROBABILITY_THRESHOLD_VALUE = 0.75
PROBABILITY_THRESHOLD_VALUE = 0.69


batch_size_val=50  # how many to process together
steps_per_epoch_val=2000
epochs_val=10
###################################################

IMAGE_WIDTH_RESIZED = 64
IMAGE_HEIGHT_RESIZED = 64
NR_OF_CHANNELS = 3
COLOR_IMAGE_DIMENSIONS = (IMAGE_WIDTH_RESIZED,IMAGE_HEIGHT_RESIZED,NR_OF_CHANNELS)
GRAY_IMAGE_DIMENSIONS = (IMAGE_WIDTH_RESIZED,IMAGE_HEIGHT_RESIZED,1)

#test_images = []
test_preprocessed_images_array = []
test_original_resized_images_array = []
test_image_descriptions = []
test_image_probabilities = []


traffic_sign_class_desc = {
                        0: "Bumpy road",
                        1: "Speed bump",
                        2: "Slippery Road",
                        3: "Dangerous left curve",
                        4: "Dangerous right curve",
                        5: "Left curve followed by right curve",
                        6: "Right curve followed by left curve",
                        7: "Place where a lot of children come",
                        8: "Bicycle crossing",
                        9: "******* UNKNOWN *****",
                        10: "Construction",
                        11: "******* UNKNOWN *****",
                        12: "Railway crossing with gates",
                        13: "Caution sign",
                        14: "Road narrows",
                        15: "******* UNKNOWN *****",
                        16: "Road narrows",
                        17: "Priority at next intersection",
                        18: "Intersection with priority to the right",
                        19: "Give way",
                        20: "Two-way traffic after part with one-way traffic",
                        21: "Stop Sign",
                        22: "Forbidden direction for all drivers of a vehicle",
                        23: "No entry for bicycles",
                        24: "No entry for vehicles which are wider than indicated",
                        25: "No entry for vehicles used for goods transport above a specific weight.",
                        26: "******* UNKNOWN *****",
                        27: "No entry for vehicles which are wider than indicated",
                        28: "No entry, in both directions, for all drivers of a vehicle",
                        29: "No Left Turn",
                        30: "No Right Turn",
                        31: "No overtaking of vehicles with more than two wheels until the next intersection",
                        32: "Maximum speed as indicated until the next intersection",
                        33: "******* UNKNOWN *****",
                        34: "MANDATORY to follow the direction indicated by the arrow",
                        35: "MANDATORY to follow the direction indicated by the arrow",
                        36: "******* UNKNOWN *****",
                        37: "Roundabout",
                        38: "Mandatory cycleway",
                        39: "Part of the road reserved for pedestrians, cyclists and mofas",
                        40: "No parking allowed",
                        41: "No parking or standing still allowed",
                        42: "No parking allowed on this side of the road from 1st day of the month until the 15th",
                        43: "No parking allowed on this side of the road from the 16th day of the month until the last",
                        44: "Narrow passage, priority over traffic from opposite side",
                        45: "Parking allowed",
                        46: "Parking only for Handicapped",
                        47: "Parking exclusively for motorcycles, motorcars and minibuses",
                        48: "******* UNKNOWN *****",
                        49: "Parking exclusively for tourist buses",
                        50: "******* UNKNOWN *****",
                        51: "Start of woonerf zone résidentielle",
                        52: "******* UNKNOWN *****",
                        53: "Road with one-way traffic (Informatory sign)",
                        54: "No exit",
                        55: "End of road works",
                        56: "Pedestrian crossing",
                        57: "Bicycle and moped crossing",
                        58: "Indicating parking",
                        59: "Speed bump",
                        60: "End of priority road",
                        61: "Priority road",
}


#Images that are to be tested against the training results. Images to be tested are appended to this list
test_images_list = [
                    "50_MaxSpeedLimit_C0032.svg.png",
                    "Stop_Sign_C0021.svg.png",

                    "Bumpy_Road_C000.svg.png",
                    "Dangerous_Right_Curve_C004.svg.png",

                    "Intersection_Priority_To_Right_C0018.svg.png",
                    "Roundabout_C0037.svg.png",

                    "Straight_Direction_Follow_C0053.svg.png",
                    #"Mandatory-to-follow-sign_C0034.svg.png",
                    "02446_00002_C007.ppm",

                    "No-Overtaking-of-vehicles_C0031.svg.png",
                    "02631_00001_C057.ppm",
                   ]



### Pre-Processing the Images ****
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)     # CONVERT TO GRAYSCALE
    img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img


# import the trained model
def load_CNN_Model():
    print("load_CNN_Model")
    pickle_in = open(PICKLED_MODEL_FILE,"rb")  ## rb = READ BYTE
    model = pickle.load(pickle_in)
    return model

def load_CNN_Model_h5():
    print("load_CNN_Model")
    model = load_model(MODEL_FILE_H5)
    return model


def get_traffic_sign_desc_from_label(classIndex):
    int_traffic_class = classIndex.__int__()

    traffic_desc = traffic_sign_class_desc.get(int_traffic_class)
    print("get_traffic_sign_desc_from_label:traffic_label = {}, traffic_desc= {}"
          .format(int_traffic_class, traffic_desc))
    return traffic_desc



def display_test_images_and_desc_from_list():
    print("display_test_images_and_desc:Enter")

    limit = 24  # show a max of 24 images
    nrows = 5
    ncols = 2
    plt.figure(figsize=(15, 5))
    i = 1

    cnt = 0
    for image in test_original_resized_images_array[:][:limit]:
        plt.subplot(nrows, ncols, i)  # 5 rows, 2 per row
        plt.axis('off')
        plt.title("{0}".format(test_image_descriptions[cnt]))
        i += 1
        plt.imshow(image)
        cnt += 1
    plt.tight_layout()
    plt.show()


def display_test_images_and_desc_from_list_with_probability():
    print("display_test_images_and_desc_from_list_with_probability:Enter")

    limit = 24  # show a max of 24 images
    nrows = 5
    ncols = 2
    plt.figure(figsize=(15, 5))
    i = 1

    cnt = 0
    for image in test_original_resized_images_array[:][:limit]:
        plt.subplot(nrows, ncols, i)  # 5 rows, 2 per row
        plt.axis('off')
        plt.title("{0} [Probability: {1}]".format(test_image_descriptions[cnt],test_image_probabilities[cnt]))
        i += 1
        plt.imshow(image)
        cnt += 1
    plt.tight_layout()
    plt.show()



def get_traffic_sign_desc_for_test_images(model):
    cnt=0
    dimension = (IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)

    for i in test_images_list:
        file_path = test_path_dir + "/" + test_images_list[cnt]
        print("get_traffic_sign_desc_for_test_images:test_img= {}".format(file_path))
        curimg = cv2.imread(file_path)
        #print("get_traffic_sign_desc_for_test_images:curimg.shape= {}".format(curimg.shape))

        resized_img = cv2.resize(curimg,
                                 dimension,
                                 interpolation=cv2.INTER_AREA)
        #print("get_traffic_sign_desc_for_test_images:resized_img.shape= {}"
        #     .format(resized_img.shape))
        test_original_resized_images_array.append(resized_img)

        preprocessed_img = preprocessing(resized_img)
        # print("get_traffic_sign_desc_for_test_images:preprocessed_img.shape= {}"
        #       .format(preprocessed_img.shape))

        test_preprocessed_images_array.append(preprocessed_img)

        reshaped_img =preprocessed_img.reshape(1,
                                               IMAGE_WIDTH_RESIZED,
                                               IMAGE_HEIGHT_RESIZED,
                                               1)
        # print("get_traffic_sign_desc_for_test_images:reshaped_img.shape= {}"
        #       .format(reshaped_img.shape))



        classIndex = model.predict_classes(reshaped_img)
        print("get_traffic_sign_desc_for_test_images:classIndex= {}".format(classIndex))


        # Predict Image
        predictions = model.predict(reshaped_img)
        print("get_traffic_sign_desc_for_test_images:predictions= {}".format(predictions))

        probabilityValue = np.amax(predictions)
        print("get_traffic_sign_desc_for_test_images:probabilityValue= {}".format(probabilityValue))
        if probabilityValue > PROBABILITY_THRESHOLD_VALUE:
            traffic_sign_desc = get_traffic_sign_desc_from_label(classIndex)
            #print("get_traffic_sign_desc_for_test_images:traffic_sign_desc= {}".format(traffic_sign_desc))
            test_image_descriptions.append(traffic_sign_desc)
            probability_percentage = "{:.0%}".format(probabilityValue)
            test_image_probabilities.append(probability_percentage)
        else:
            test_image_descriptions.append("Unknown")
            test_image_probabilities.append("0")
            print("get_traffic_sign_desc_for_test_images:** UNKNOWN **")

        cnt+=1

    #display_test_images_and_desc_from_list()
    display_test_images_and_desc_from_list_with_probability()

# ***********   Changes to Support Linux ************************
if sys.platform.startswith('linux'):

    print("TSign_Classification_Test: *** Linux Platform ***")
    # linux-specific code here...
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
elif sys.platform.startswith('win32'):
    # Windows-specific code here...
    print("TSign_Classification_Test: *** Windows Platform ***")
# **************************************************************


#cnn_model = load_CNN_Model()
cnn_model = load_CNN_Model_h5()
get_traffic_sign_desc_for_test_images(cnn_model)


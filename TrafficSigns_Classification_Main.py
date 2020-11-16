import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model

import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
import contextlib
from collections import Counter

from keras.preprocessing.image import ImageDataGenerator

from BarChart_utils import horizontal_bar_chart, plot_stacked_bar

# ***********   Changes to Support Linux ************************
import sys
from sys import platform
# ***************************************************************

print ('Tensorflow version = {}'.format(tf.__version__))
print ('OpenCV version = {}'.format(cv2.__version__))


################# Parameters #####################

dataset_path = "datasets" # folder with all the class folders
training_path_dir = "../datasets/Training"
path = training_path_dir
validation_path_dir = "../datasets/Validation"
#batch_size_val=50  # how many to process together
batch_size_val=32  # how many to process together
steps_per_epoch_val=2000
#epochs_val=10
epochs_val=225
###################################################

labelFile = 'Belgium_Traffic_labels.csv' # file with all names of classes
MODEL_FILE = "Belgium_Traffic_OpenCV_model_trained.p"
MODEL_FILE_H5 = "Belgium_Traffic_OpenCV_model_trained.h5"
BELGIUM_TRAFFIC_MODEL_SUMMARY_FNAME_PNG = "Belgium_Traffic_Model_Summary.png"

IMAGE_WIDTH_RESIZED = 64
IMAGE_HEIGHT_RESIZED = 64
NR_OF_CHANNELS = 3
imageDimesions = (IMAGE_WIDTH_RESIZED,IMAGE_HEIGHT_RESIZED,NR_OF_CHANNELS)


#### Variables for Training Images
training_images = []
training_labels = []
nr_of_samples_in_training_labels = []
total_nr_of_training_labels = 0

#### Variables for Validation Images
validation_images = []
validation_labels = []
nr_of_samples_in_validation_labels = []
total_nr_of_validation_labels = 0


def load_training_data():
    global training_images, training_labels, total_nr_of_training_labels, nr_of_samples_in_training_labels
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(training_path_dir)
                    if os.path.isdir(os.path.join(training_path_dir, d))]

    print("load_training_data:data_dir {0}".format(training_path_dir))
    print("load_training_data:directories {0}".format(directories))
    total_nr_of_training_labels = len(directories)
    print("load_training_data:# of Labels {0}".format(total_nr_of_training_labels))

    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    filecount = 0
    dimension = (IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)

    for d in directories:
         label_dir = os.path.join(training_path_dir, d)
         file_names = [os.path.join(label_dir, f)
                       for f in os.listdir(label_dir) if f.endswith(".ppm")]
         # print ("load_data: file_names- {0}" .format(file_names))

         # For each label, load it's images and add them to the images list.
         # And add the label number (i.e. directory name) to the labels list.
         #dirListing = os.listdir(label_dir)
         print("load_training_data: label_dir= {0}, nr_of_files= {1}".format(label_dir, len(file_names)))
         nr_of_samples_in_training_labels.append(len(file_names))

         if (len(file_names)== 0):
             print ("load_training_data:File empty d={0}, int(d)={1}".format(d, int(d)))
             training_labels.append(int(d))

         for f in file_names:
             curImg = cv2.imread(f)
             #print("curImg.shape= {}".format(curImg.shape))
             resized_img = cv2.resize(curImg,
                                      dimension,
                                      interpolation = cv2.INTER_AREA)
             training_images.append(resized_img)
             training_labels.append(int(d))

         # print("load_training_data: nr of files= {0}, labels= {1}"
         #       .format(len(training_images),len(training_labels)))

    training_images = np.array(training_images)
    training_labels = np.array(training_labels)
    print("load_training_data:training_images.shape= {}".format(training_images.shape))
    print("load_training_data:training_labels.shape= {}".format(training_labels.shape))

    print("load_training_data:training_labels= {}".format(training_labels))
    print("load_training_data:nr_of_samples_in_training_labels= {}".format(nr_of_samples_in_training_labels))
    print("load_training_data:total_nr_of_training_labels= {}".format(total_nr_of_training_labels))


#Display the first image of each training label.
def display_sample_training_images_and_labels():
    """Display the first image of each label."""
    print("display_sample_training_images_and_labels: labels= {0}".format(training_labels))

    unique_labels = set(training_labels)
    print("display_sample_training_images_and_labels: unique_labels= {0}".format(unique_labels))

    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        #print("display_images_and_labels: labels.index(label)= {0}".format(labels.index(label)))
        img_index = np.where(training_labels == label)
        #print("display_sample_training_images_and_labels:img_index= {}".format(img_index))
        image = training_images[img_index][0]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, np.count_nonzero(training_labels==label)))
        i += 1
        _ = plt.imshow(image)
    plt.tight_layout()
    plt.show()

    #print the shape, min and max of the images from index 0 to 5
    for image in training_images[:5]:
        print("display_images_and_labels:shape: {0}, min: {1}, max: {2}"
              .format(image.shape, image.min(), image.max()))


### Display a Bar chart showing # of samples for each category #####
def show_bar_chart_for_training_data():
    print("show_bar_chart_for_training_data:total_nr_of_training_labels = {}".format(total_nr_of_training_labels))
    print("show_bar_chart_for_training_data:nr_of_samples_in_training_labels= {}".format(nr_of_samples_in_training_labels))
    plt.figure(figsize=(12, 4))

    # assign your bars to a variable so their attributes can be accessed
    bars = plt.bar(range(0, total_nr_of_training_labels), nr_of_samples_in_training_labels)
    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)

    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()


#### PreProcessing the Images #######
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

def preprocess_training_images():
    global training_images
    training_images = np.array(list(map(preprocessing,training_images)))

# ***********   Changes to Support Linux ************************
    if sys.platform.startswith('linux'):
        print("preprocess_training_images: *** Linux Platform ***")
        # linux-specific code here...
        #    cv2.imshow("GrayScale Images",training_images[index-1])
        #    cv2.waitKey(0)
    elif sys.platform.startswith('win32'):
        # Windows-specific code here...
        print("preprocess_training_images: *** Windows Platform ***")
        index = random.randint(0, len(training_images))
        print("preprocess_training_images:index = {}".format(index))
        cv2.imshow("GrayScale Images",training_images[index-1])
        cv2.waitKey(0)
# ***************************************************************


# Add Depth of 1 to the Images
def reshape_grayscale_training_images():
    global training_images
    print("reshape_grayscale_training_images:training_images.shape= {}".format(training_images.shape))
    print("reshape_grayscale_training_images:training_images.shape[0]= {}".format(training_images.shape[0]))
    print("reshape_grayscale_training_images:training_images.shape[1]= {}".format(training_images.shape[1]))
    print("reshape_grayscale_training_images:training_images.shape[2]= {}".format(training_images.shape[2]))

    training_images = training_images.reshape(training_images.shape[0],
                                              training_images.shape[1],
                                              training_images.shape[2],
                                              1)

    print("reshape_grayscale_training_images:training_images.shape= {}".format(training_images.shape))


def load_validation_data():
    global validation_images, validation_labels, total_nr_of_validation_labels, nr_of_samples_in_validation_labels
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(validation_path_dir)
                    if os.path.isdir(os.path.join(validation_path_dir, d))]

    print("load_validation_data:data_dir {0}".format(validation_path_dir))
    print("load_validation_data:directories {0}".format(directories))
    total_nr_of_validation_labels = len(directories)
    print("load_validation_data:# of Labels {0}".format(total_nr_of_validation_labels))

    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    filecount = 0
    dimension = (IMAGE_WIDTH_RESIZED, IMAGE_HEIGHT_RESIZED)

    for d in directories:
         label_dir = os.path.join(validation_path_dir, d)
         file_names = [os.path.join(label_dir, f)
                       for f in os.listdir(label_dir) if f.endswith(".ppm")]
         # print ("load_data: file_names- {0}" .format(file_names))

         # For each label, load it's images and add them to the images list.
         # And add the label number (i.e. directory name) to the labels list.
         #dirListing = os.listdir(label_dir)
         print("load_validation_data: label_dir= {0}, nr_of_files= {1}".format(label_dir, len(file_names)))
         nr_of_samples_in_validation_labels.append(len(file_names))

         if (len(file_names)== 0):
             print ("load_validation_data:File empty d={0}, int(d)={1}".format(d, int(d)))
             validation_labels.append(int(d))

         for f in file_names:
             curImg = cv2.imread(f)
             #print("curImg.shape= {}".format(curImg.shape))
             resized_img = cv2.resize(curImg,
                                      dimension,
                                      interpolation = cv2.INTER_AREA)
             validation_images.append(resized_img)
             validation_labels.append(int(d))

         # print("load_validation_data: nr of files= {0}, labels= {1}"
         #       .format(len(validation_images),len(validation_labels)))

    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)
    print("load_validation_data:validation_images.shape= {}".format(validation_images.shape))
    print("load_validation_data:validation_labels.shape= {}".format(validation_labels.shape))

    print("load_validation_data:validation_labels= {}".format(validation_labels))
    print("load_validation_data:nr_of_samples_in_validation_labels= {}".format(nr_of_samples_in_validation_labels))
    print("load_validation_data:total_nr_of_validation_labels= {}".format(total_nr_of_validation_labels))


### Display a Bar chart showing # of samples for each category #####
def show_bar_chart_for_validation_data():
    print("show_bar_chart_for_validation_data:total_nr_of_validation_labels = {}".format(total_nr_of_validation_labels))
    print("show_bar_chart_for_validation_data:nr_of_samples_in_validation_labels= {}".format(nr_of_samples_in_validation_labels))
    plt.figure(figsize=(12, 4))

    # assign your bars to a variable so their attributes can be accessed
    bars = plt.bar(range(0, total_nr_of_validation_labels), nr_of_samples_in_validation_labels)
    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)

    plt.title("Distribution of the Validation dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()



def show_horizontal_bar_chart_for_training_validation_data():
    global training_labels, nr_of_samples_in_training_labels, nr_of_samples_in_validation_labels

    set_training_labels = list(set(training_labels))
    print("show_horizontal_bar_chart_for_training_validation_data:set_training_labels= {}"
          .format(set_training_labels))

    zipped_training_validation_data_counter = zip(nr_of_samples_in_training_labels, nr_of_samples_in_validation_labels)
    zipped_labels_and_nr_of_samples = zip(set_training_labels, zipped_training_validation_data_counter)
    dict_labels_samples_distr = dict(zipped_labels_and_nr_of_samples)

    for i in zipped_labels_and_nr_of_samples:
        print('show_horizontal_bar_chart_for_training_validation_data:zipped_class_nr_of_samples_object= {}'
              .format(i))

    print('show_horizontal_bar_chart_for_training_validation_data:dict_labels_samples_distr= {}'
          .format(dict_labels_samples_distr))

    category_names = ['training samples', 'validation samples']
    horizontal_bar_chart(dict_labels_samples_distr, category_names)


def show_stacked_vertical_bar_chart_for_training_validation_data():
    global training_labels, nr_of_samples_in_training_labels, nr_of_samples_in_training_labels

    category_names = ['training samples', 'validation samples']
    data = []

    set_training_labels = list(set(training_labels))
    print("show_stacked_vertical_bar_chart_for_training_validation_data:set_training_labels= {}"
          .format(set_training_labels))

    data.append(nr_of_samples_in_training_labels)
    data.append(nr_of_samples_in_validation_labels)

    print('show_stacked_vertical_bar_chart_for_training_validation_data:data= {}'.format(data))

    plot_stacked_bar(data,
                     category_names,
                     category_labels=set_training_labels,
                     show_values=True,
                     value_format="{:.1f}",
                     colors=['tab:orange', 'tab:green'],
                     x_label="Labels",
                     y_label="Nr of Samples")







def preprocess_validation_images():
    global validation_images
    validation_images = np.array(list(map(preprocessing,validation_images)))

# ***********   Changes to Support Linux ************************
    if sys.platform.startswith('linux'):
        print("preprocess_validation_images: *** Linux Platform ***")
        # linux-specific code here...
    #   cv2.imshow("GrayScale Images",validation_images[index-1])
    #   cv2.waitKey(0)
    elif sys.platform.startswith('win32'):
        # Windows-specific code here...
        print("preprocess_validation_images: *** Windows Platform ***")
        index = random.randint(0,len(validation_images))
        print("preprocess_validation_images:index = {}".format(index))
        cv2.imshow("GrayScale Images",validation_images[index-1])
        cv2.waitKey(0)
# ***************************************************************


# Add Depth of 1 to the Images
def reshape_grayscale_validation_images():
    global validation_images
    print("reshape_grayscale_validation_images:validation_images.shape= {}".format(validation_images.shape))
    print("reshape_grayscale_validation_images:validation_images.shape[0]= {}".format(validation_images.shape[0]))
    print("reshape_grayscale_validation_images:validation_images.shape[1]= {}".format(validation_images.shape[1]))
    print("reshape_grayscale_validation_images:validation_images.shape[2]= {}".format(validation_images.shape[2]))

    validation_images = validation_images.reshape(validation_images.shape[0],
                                                  validation_images.shape[1],
                                                  validation_images.shape[2],1)

    print("reshape_grayscale_training_images:validation_images.shape= {}".format(validation_images.shape))


def augment_training_data():
    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                 # 0.1 = 10%.If more than 1 e.g. 10 Then it refers to # of Pixels e.g. 10 pixels
                                 height_shift_range=0.1,
                                 zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                                 shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                                 rotation_range=10)  # DEGREES
    dataGen.fit(training_images)
    batches = dataGen.flow(training_images,
                           training_labels,
                           # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
                           batch_size=20)
    X_batch, y_batch = next(batches)

    # TO SHOW Augmented Image Samples
    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()

    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
        axs[i].axis('off')
    plt.show()
    return dataGen


def convert_to_onehot_encoding():
    global training_labels, validation_labels, nr_of_training_labels, total_nr_of_validation_labels
    print("convert_to_onehot_encoding:total_nr_of_training_labels= {}".format(total_nr_of_training_labels))
    print("convert_to_onehot_encoding:total_nr_of_validation_labels= {}".format(total_nr_of_validation_labels))

    training_labels = to_categorical(training_labels, total_nr_of_training_labels)
    validation_labels = to_categorical(validation_labels, total_nr_of_validation_labels)


#### CONVOLUTION NEURAL NETWORK MODEL #####
def define_CNN_model():
    no_Of_Filters=60
    # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
    # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_Filter=(5,5)
    size_of_Filter2=(3,3)
    # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    size_of_pool=(2,2)
    # NO. OF NODES IN HIDDEN LAYERS
    no_Of_Nodes = 500

    model= Sequential()
    model.add((Conv2D(no_Of_Filters,
                      size_of_Filter,
                      input_shape=(imageDimesions[0],imageDimesions[1],1),
                      activation='relu')))

    # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters,
                      size_of_Filter,
                      activation='relu')))
    # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_Of_Filters//2,
                      size_of_Filter2,
                      activation='relu')))

    model.add((Conv2D(no_Of_Filters // 2,
                      size_of_Filter2,
                      activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dropout(0.5))
    # OUTPUT LAYER
    model.add(Dense(total_nr_of_training_labels,
                    activation='softmax'))

    # COMPILE MODEL
    model.compile(Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    with contextlib.suppress(FileNotFoundError):
        os.remove(BELGIUM_TRAFFIC_MODEL_SUMMARY_FNAME_PNG)
    plot_model(model,
               to_file=BELGIUM_TRAFFIC_MODEL_SUMMARY_FNAME_PNG,
               show_shapes=True,
               show_layer_names=True)

    return model


def plot_loss_accuracy_graphs(history):
    print("plot_loss_accuracy_graphs")
    
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Acurracy')
    plt.xlabel('epoch')
    plt.show()


def save_the_model(model):
    # Store the Model as a  PICKLE OBJECT
    print("save_the_model: {}".format(MODEL_FILE))
    # wb = Write Byte
    pickle_out = open(MODEL_FILE, "wb")  
    pickle.dump(model, pickle_out)
    pickle_out.close()
    cv2.waitKey(0)

def save_the_model_to_h5(model):
    # Store the Model as a  PICKLE OBJECT
    print("save_the_model_to_h5: {}".format(MODEL_FILE_H5))
    model.save(MODEL_FILE_H5)
    cv2.waitKey(0)




# ***********   Changes to Support Linux ************************
if sys.platform.startswith('linux'):
    print("TSign_Classification_Training: *** Linux Platform ***")
    # linux-specific code here...
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
elif sys.platform.startswith('win32'):
    # Windows-specific code here...
    print("TSign_Classification_Training: *** Windows Platform ***")
# ***************************************************************

## Prepare the Training Data #####
load_training_data()
display_sample_training_images_and_labels()
show_bar_chart_for_training_data()
preprocess_training_images()
reshape_grayscale_training_images()

## Prepare the Validation Data #####
load_validation_data()
show_bar_chart_for_validation_data()
preprocess_validation_images()
reshape_grayscale_validation_images()

#show_horizontal_bar_chart_for_training_validation_data()
show_stacked_vertical_bar_chart_for_training_validation_data()


# Augment training Data and show few samples
training_datagen = augment_training_data()
convert_to_onehot_encoding()
cnn_model = define_CNN_model()


# Start Training the Model
#temp_steps_per_epoch_val = steps_per_epoch_val
temp_steps_per_epoch_val = int(steps_per_epoch_val/batch_size_val)
history = cnn_model.fit_generator(training_datagen.flow(training_images,
                                                        training_labels,
                                                        batch_size=batch_size_val),
                                  steps_per_epoch = temp_steps_per_epoch_val,
                                  epochs = epochs_val,
                                  validation_data=(validation_images,validation_labels),
                                  shuffle=1)

plot_loss_accuracy_graphs(history)
#save_the_model(cnn_model)
save_the_model_to_h5(cnn_model)



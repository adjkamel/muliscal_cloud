import os
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
import sys
import time
import warnings
from keras.models import *
from keras.layers import *

from sklearn.utils import shuffle
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from PIL import Image
import time

import tifffile

from datetime import datetime

from keras.callbacks import TensorBoard


def Extract(dir_root):

    tiff_o = os.listdir(dir_root+"tiff_o/")
    tiff_mask = os.listdir(dir_root+"tiff_mask/")

    # from tiff 10 channels to 4 channels in floders

    for k in range(0, len(tiff_o)):

        with tifffile.TiffFile(dir_root+"tiff_o/"+tiff_o[k]) as tif:
            data = tif.asarray()
            cv2.imwrite(dir_root+"train_blue/"+tiff_o[k], data[:, :, 1])
            cv2.imwrite(dir_root+"train_red/"+tiff_o[k], data[:, :, 2])
            cv2.imwrite(dir_root+"train_green/"+tiff_o[k], data[:, :, 3])
            cv2.imwrite(dir_root+"train_nir/"+tiff_o[k], data[:, :, 4])

            print(data[:, :, 1].shape)

    for k in range(0, len(tiff_mask)):
        img = cv2.imread(dir_root+"tiff_mask/"+tiff_mask[k], 0)
        #img=cv2.resize(img,(224 , 224))
        cv2.imwrite(dir_root+"grey_mask/"+tiff_o[k], img)


Extract("l8cloudmasks/")

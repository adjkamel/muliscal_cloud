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
from keras import backend


from sklearn.utils import shuffle
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from PIL import Image
import time

import tifffile

from datetime import datetime

from keras.callbacks import TensorBoard

#  dir_root = "l8cloudmasks/"


def prepareData(opt, dir_root):

    # tiff_o = os.listdir(dir_root+"tiff_o/")
    # tiff_mask = os.listdir(dir_root+"tiff_mask/")
    # tiff_qmask = os.listdir(dir_root+"tiff_qmask/")

    dir_blue = os.listdir(dir_root+"train_blue/")
    dir_red = os.listdir(dir_root+"train_red/")
    dir_green = os.listdir(dir_root+"train_green/")
    dir_nir = os.listdir(dir_root+"train_nir/")
    # dir_gt = os.listdir(dir_root+"train_gt")
    # dir_gtall = os.listdir(dir_root+"train_gtall/")
    dir_grey = os.listdir(dir_root+"grey_mask/")

    # from tiff 10 channels to 4 channels in floders

    hieght = 224
    width = 224

    List_img_train = []
    List_gt_train = []
    List_img_test = []
    List_gt_test = []

    # range(0,len(dir_gt)-1):  #range(0,2000):

    list_test = ["015024", "221066", "001081", "201033",
                 "181059", "148035", "137045", "094080"]

    for k in range(0, len(dir_blue)):
        # normalistation
        x = np.zeros((224, 224, 4))
        b = cv2.imread(dir_root+"train_blue/"+dir_blue[k], 0)
        g = cv2.imread(dir_root+"train_green/"+dir_green[k], 0)
        r = cv2.imread(dir_root+"train_red/"+dir_red[k], 0)
        n = cv2.imread(dir_root+"train_nir/"+dir_nir[k], 0)
        gt = cv2.imread(dir_root+"grey_mask/"+dir_grey[k], 0)

        #  b = cv2.resize(b/255,(hieght , width))
        #  g = cv2.resize(g/255,(hieght , width))
        #  r = cv2.resize(r/255,(hieght , width))
        #  n = cv2.resize(n/255,(hieght , width))
        # gt = cv2.resize(gt/255,(hieght , width))

        x = np.dstack([b, g, r, n])
        # x= cv2.merge((b,g,r,n))

        seg_labels = np.zeros((1000, 1000, 5))

        # each channel is a class
        # thresh = 127
        # gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)[1]

        seg_labels[:, :, 0] = np.where((gt == 0) | (gt == 46), gt, 200)
        seg_labels[:, :, 0] = np.where(
            seg_labels[:, :, 0] == 200, seg_labels[:, :, 0], 205)
        seg_labels[:, :, 0] = np.where(
            seg_labels[:, :, 0] == 205, seg_labels[:, :, 0], 0)

        seg_labels[:, :, 1] = np.where((gt == 95) | (gt == 122),   gt, 0)
        seg_labels[:, :, 2] = np.where(gt == 217,  gt, 0)
        seg_labels[:, :, 3] = np.where(gt == 128,  gt, 0)
        seg_labels[:, :, 4] = np.where(gt == 255,  gt, 0)

        seg_labels[:, :, 0] = np.where(
            seg_labels[:, :, 0] == 0, seg_labels[:, :, 0], 1)
        seg_labels[:, :, 1] = np.where(
            seg_labels[:, :, 1] == 0,  seg_labels[:, :, 1], 1)
        seg_labels[:, :, 2] = np.where(
            seg_labels[:, :, 2] == 0,  seg_labels[:, :, 2], 1)
        seg_labels[:, :, 3] = np.where(
            seg_labels[:, :, 3] == 0,  seg_labels[:, :, 3], 1)
        seg_labels[:, :, 4] = np.where(
            seg_labels[:, :, 4] == 0,  seg_labels[:, :, 4], 1)

        # img = Image.fromarray(seg_labels[: , : , 1 ])
        # cv2.imshow('',img)

        # plt.imshow(seg_labels[: , : , 2 ],'gray')
        # plt.show()
        # time.sleep(5)

        im = Image.fromarray(seg_labels[:, :, 0])
        im.save(dir_root+"ground/"+"0_"+dir_grey[k])
        im = Image.fromarray(seg_labels[:, :, 1])
        im.save(dir_root+"ground/"+"1_"+dir_grey[k])
        im = Image.fromarray(seg_labels[:, :, 2])
        im.save(dir_root+"ground/"+"2_"+dir_grey[k])
        im = Image.fromarray(seg_labels[:, :, 3])
        im.save(dir_root+"ground/"+"3_"+dir_grey[k])
        im = Image.fromarray(seg_labels[:, :, 4])
        im.save(dir_root+"ground/"+"4_"+dir_grey[k])

        y = seg_labels

        x = np.float32(cv2.resize(x/255, (hieght, width)))
        y = np.float32(cv2.resize(y, (hieght, width)))

        # x=np.float32(cv2.resize(x/255,(hieght , width)))
        # y=np.float32(cv2.resize(y/255,(hieght , width)))

        # y=np.where(y == 0, y, 1)

        # print(x)
        # print('--------------------------------------')
        # print(y[:,:,1])

        if any(substring in dir_blue[k] for substring in list_test):

            List_img_test.append(x)
            List_gt_test.append(y)

        else:

            List_img_train.append(x)
            List_gt_train.append(y)

    if opt == "train":
        return List_img_train, List_gt_train
    else:
        return List_img_test, List_gt_test

    # cv2.imwrite('image.tif', X)

    # return List_img_train, List_gt_train


warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
gpus = tf.config.experimental.list_physical_devices('GPU')
sess = tf.compat.v1.Session(config=config)


config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = '0'
tf.compat.v1.Session(config=config)

print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__))
del keras
print("tensorflow version {}".format(tf.__version__))
# weights
'''
VGG_Weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

weights = {}

keys = []
with h5py.File(VGG_Weights_path, 'r') as f: # open file
    f.visit(keys.append) # append all keys to list
    for key in keys:
        if ':' in key: # contains data if ':' in key
            print(f[key].name)
            weights[f[key].name] = f[key].value
'''
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

# vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8

# w1 = h5py.File(VGG_Weights_path,'r')
# allKeys = w1.keys()
# print(w1['futures_data'] )
# print('++++++++++++++++++++++++++++++++++++++++++')
# first_layer_wts = w1[allKeys['block1_conv1']] # assuming first layer has weights
# print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# model


def FCN8(nClasses,  input_height=224, input_width=224):
    # input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    # which makes the input_height and width 2^5 = 32 times smaller
  #  assert input_height%32 == 0
   # assert input_width%32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 4))  # Assume 224,224,4
    print('----------------------------')
    print(img_input)
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    print('----------------------------')
    print(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    # f2 = x

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)

    # pyramid
    features = x

    # red
    red = AveragePooling2D(pool_size=(2, 2), name='red_pool')(features)

    # red = tf.keras.layers.Reshape((1,1,256))(red)
    red = Convolution2D(filters=64, kernel_size=(1, 1), name='red_1_by_1')(red)
    red = UpSampling2D(size=2, interpolation='bilinear',
                       name='red_upsampling')(red)

    # yellow
    # yellow = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(features)
    # yellow = Convolution2D(filters=64, kernel_size=(
    #    1, 1), name='yellow_1_by_1')(yellow)
    # yellow = UpSampling2D(size=2, interpolation='bilinear',
    #                      name='yellow_upsampling')(yellow)

    # blue
    blue = AveragePooling2D(pool_size=(4, 4), name='blue_pool')(features)
    blue = Convolution2D(filters=64, kernel_size=(1, 1),
                         name='blue_1_by_1')(blue)
    blue = UpSampling2D(size=4, interpolation='bilinear',
                        name='blue_upsampling')(blue)

    # green

    # green = AveragePooling2D(pool_size=(7, 7), name='green_pool')(features)
    # green = Convolution2D(filters=64, kernel_size=(1, 1),
    # name = 'green_1_by_1')(green)
    # green = UpSampling2D(size=7, interpolation='bilinear',
    #   name='green_upsampling')(green)

    # base + red + yellow + blue + green
    concat = concatenate([features, red, blue])  # ,green

    # last segmentation layers
    pool311 = (Conv2D(nClasses, (1, 1), activation='relu', padding='same',
                      name="pool3_11", data_format=IMAGE_ORDERING))(concat)
    o = Conv2DTranspose(nClasses, kernel_size=(8, 8),  strides=(
        8, 8), use_bias=False, data_format=IMAGE_ORDERING)(pool311)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)

    return model


n_classes = 5

model = FCN8(nClasses=n_classes,
             input_height=224,
             input_width=224)

model.summary()

############# May be getImageArr ################
###  Split between training and testing data  ###


X, Y = prepareData("train", "l8cloudmasks/")


'''
train_rate = 0.85
X,Y=prepareData()
index_train = np.random.choice(
    X.shape[0],int(Y.shape[0]*train_rate),replace=False)
index_test  = list(set(range(X.shape[0])) - set(index_train))

X, Y = shuffle(X,Y)
X_train, y_train = X[index_train],Y[index_train]
X_test, y_test = X[index_test],Y[index_test]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

'''

# Training starts here


X, Y = np.array(X), np.array(Y)

print(X.shape)
print(Y.shape)
# print(X.shape,Y.shape)
'''
Adadelta = optimizers.Adadelta(learning_rate=0.15, rho=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta,
              metrics=['accuracy'])

'''
# sgd = optimizers.SGD(lr=0.001, decay=5**(-4), momentum=0.9, nesterov=True)


model = tf.keras.models.load_model(os.path.abspath("best_model_l8.hdf5"))
# backend.get_value(model.optimizer.lr)
# backend.set_value(model.optimizer.lr, 1e-2)

# model.compile(optimizer=optimizers.Adam(lr=1e-4),
# loss='binary_crossentropy', metrics=['accuracy'])


checkpoint = ModelCheckpoint(
    "best_model_l8.hdf5", monitor='loss', verbose=1, save_best_only=False, mode='max')
logdir = "logs" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)


# hist1 = model.fit(X, Y, batch_size=16, epochs=10, verbose=1, callbacks=[tensorboard_callback])
#  # validation_data=(X_test,y_test),

hist1 = model.fit(X, Y, batch_size=20, epochs=1,
                  verbose=1, callbacks=[checkpoint])

### Plot the change in loss over epochs ###


for key in ['loss']:
    plt.plot(hist1.history['loss'], label=key)
plt.legend()
plt.show()


Xtest, Ytest = prepareData("test", "l8cloudmasks/")


Xtest, Ytest = np.array(Xtest), np.array(Ytest)
y_pred = np.array(model.predict(Xtest))
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(Ytest, axis=3)
# print(y_testi.shape, y_predi.shape)

print("y_pred",  type(y_pred), y_pred.shape)
print("Ytest",  type(Ytest), Ytest.shape)
print("y_predi", y_predi, type(y_predi), y_predi.shape)
print("y_predi max", np.max(y_predi), type(y_predi), y_predi.shape)
print("y_predi min", np.min(y_predi), type(y_predi), y_predi.shape)

print("Xtest",  type(y_pred), Xtest.shape)
print("y_testi", y_testi, type(y_predi), y_testi.shape)


def IoU(Yi, y_predi):
    # mean Intersection over Union
    # Mean IoU = TP/(FN + TP + FP)

    Nclass = int(np.max(Yi)) + 1
    for i in range(0, 8):
        pixall = []
        print("............scene:", i)
        for c in range(Nclass):
            TPP = np.sum((y_predi[i] == c) & (Yi[i] == c))
            TPG = np.sum(y_predi
                         [i] == c)
            print("TPP", TPP)
            print("TPG", TPG)
            if (TPP <= TPG):
                pix = TPP/TPG
            else:
                pix = TPG/TPP

            print("class=", c, "pix=", pix)

            pixall.append(pix)
        mpix = np.mean(pixall)
        print("_________________")
        print("Mean mpix=", mpix)


'''
def IoU(Yi, y_predi):
    # mean Intersection over Union
    # Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum((Yi == c) & (y_predi == c))
        FP = np.sum((Yi != c) & (y_predi == c))
        FN = np.sum((Yi == c) & (y_predi != c))
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(
            c, TP, FP, FN, IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))

'''


IoU(y_testi, y_predi)


shape = (224, 224)

'''
def give_color_to_seg_img(seg, n_classes):

    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc*(colors[c][0]))
        seg_img[:, :, 1] += (segc*(colors[c][1]))
        seg_img[:, :, 2] += (segc*(colors[c][2]))

    return(seg_img)

'''
cpr = ["clsh", "wt", "ic-sn", "Ln", "cl"]
for i in range(8):

    img_is = (Xtest[i] + 1)*(255.0/2)
    seg = y_predi[i]
    segtest = y_testi[i]
    # ----------------------------------- img

    fig = plt.figure(figsize=(10, 30))
    ax = fig.add_subplot(3, 5, 1)
    ax.imshow(img_is/255.0)
    ax.set_title("original")
    # --------------------------------- pred

    k = 0
    for i in range(6, 11):
        ax = fig.add_subplot(3, 5, i)
        ax.imshow(np.where(seg == k,  1, 0),
                  cmap='Greys',  interpolation='nearest')
        ax.set_title(cpr[k])
        k += 1
    k = 0
    for i in range(11, 16):
        ax = fig.add_subplot(3, 5, i)
        ax.imshow(np.where(segtest == k,  1, 0),
                  cmap='Greys',  interpolation='nearest')
        ax.set_title(cpr[k])
        k += 1

    plt.show()

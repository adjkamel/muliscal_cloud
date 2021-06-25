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
from keras import backend


def prepareData():

    dir_root = "38-Cloud_training/"
    dir_blue = os.listdir(dir_root+"train_blue")
    dir_red = os.listdir(dir_root+"train_red")
    dir_green = os.listdir(dir_root+"train_green")
    dir_nir = os.listdir(dir_root+"train_nir")
    dir_gt = os.listdir(dir_root+"train_gt")

    hieght = 224
    width = 224

    List_img_train = []
    List_gt_train = []

    for k in range(0, len(dir_gt)):  # range(0,2000):
        # normalistation
        x = np.zeros((224, 224, 4))
        b = cv2.imread(dir_root+"train_blue/"+dir_blue[k], 0)
        g = cv2.imread(dir_root+"train_green/"+dir_green[k], 0)
        r = cv2.imread(dir_root+"train_red/"+dir_red[k], 0)
        n = cv2.imread(dir_root+"train_nir/"+dir_nir[k], 0)
        gt = cv2.imread(dir_root+"train_gt/"+dir_gt[k], 0)

      #  b = cv2.resize(b/255,(hieght , width))
      #  g = cv2.resize(g/255,(hieght , width))
      #  r = cv2.resize(r/255,(hieght , width))
       # n = cv2.resize(n/255,(hieght , width))
       # gt = cv2.resize(gt/255,(hieght , width))
        x = np.dstack([b, g, r, n])
        #x= cv2.merge((b,g,r,n))

        seg_labels = np.zeros((384, 384, 2))

        # each channel is a class
        thresh = 127
        gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)[1]
        seg_labels[:, :, 0] = 0
        seg_labels[:, :, 1] = gt

        y = seg_labels

        x = np.float32(cv2.resize(x/255, (hieght, width)))
        y = np.float32(cv2.resize(y/255, (hieght, width)))

        y = np.where(y == 0, y, 1)

        # print(x)
        # print('--------------------------------------')
        # print(y[:,:,1])

        List_img_train.append(x)
        List_gt_train.append(y)

        #cv2.imwrite('image.tif', X)

    return List_img_train, List_gt_train


def prepareDataTest():

    dir_root = "38-Cloud_testing/"
    dir_blue = os.listdir(dir_root+"test_blue")
    dir_red = os.listdir(dir_root+"test_red")
    dir_green = os.listdir(dir_root+"test_green")
    dir_nir = os.listdir(dir_root+"test_nir")

    hieght = 224
    width = 224

    List_img_test = []
    List_img_test_names = []

    for k in range(0, 100):  # range(0,len(dir_gt)-1):  #range(0,2000):
        # normalistation
        x = np.zeros((224, 224, 4))
        b = cv2.imread(dir_root+"test_blue/"+dir_blue[k], 0)
        g = cv2.imread(dir_root+"test_green/"+dir_green[k], 0)
        r = cv2.imread(dir_root+"test_red/"+dir_red[k], 0)
        n = cv2.imread(dir_root+"test_nir/"+dir_nir[k], 0)

        x = np.dstack([b, g, r, n])
        #x= cv2.merge((b,g,r,n))

        x = np.float32(cv2.resize(x/255, (hieght, width)))

        List_img_test.append(x)
        List_img_test_names.append(dir_blue[k])
        #cv2.imwrite('image.tif', X)

    return List_img_test, List_img_test_names

    # configuration


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

VGG_Weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

weights = {}

keys = []
with h5py.File(VGG_Weights_path, 'r') as f:  # open file
    f.visit(keys.append)  # append all keys to list
    for key in keys:
        if ':' in key:  # contains data if ':' in key
            print(f[key].name)
            weights[f[key].name] = f[key].value

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

# vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8

#w1 = h5py.File(VGG_Weights_path,'r')
#allKeys = w1.keys()
#print(w1['futures_data'] )
# print('++++++++++++++++++++++++++++++++++++++++++')
# first_layer_wts = w1[allKeys['block1_conv1']] # assuming first layer has weights
# print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# model


def FCN8(nClasses,  input_height=224, input_width=224):
    # input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    # which makes the input_height and width 2^5 = 32 times smaller
  #  assert input_height%32 == 0
   #assert input_width%32 == 0
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

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    #f2 = x

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


n_classes = 2

model = FCN8(nClasses=n_classes,
             input_height=224,
             input_width=224)

model.summary()

############# May be getImageArr ################
###  Split between training and testing data  ###


X, Y = prepareData()


'''
train_rate = 0.85
X,Y=prepareData()
index_train = np.random.choice(X.shape[0],int(Y.shape[0]*train_rate),replace=False)
index_test  = list(set(range(X.shape[0])) - set(index_train))

X, Y = shuffle(X,Y)
X_train, y_train = X[index_train],Y[index_train]
X_test, y_test = X[index_test],Y[index_test]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

'''

# Training starts here


X, Y = np.array(X), np.array(Y)

# print(X.shape,Y.shape)
'''
Adadelta = optimizers.Adadelta(learning_rate=0.15, rho=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta,
              metrics=['accuracy'])
            
'''
#sgd = optimizers.SGD(lr=0.001, decay=5**(-4), momentum=0.9, nesterov=True)

model = tf.keras.models.load_model(os.path.abspath("best_model_38.hdf5"))
backend.get_value(model.optimizer.lr)
backend.set_value(model.optimizer.lr, 1e-7)

# model.compile(optimizer=optimizers.Adam(lr=1e-4),
# loss='binary_crossentropy', metrics=['accuracy'])


checkpoint = ModelCheckpoint("best_model_38.hdf5", monitor='loss',
                             verbose=1, save_best_only=False, mode='auto', period=1)

hist1 = model.fit(X, Y, batch_size=4, epochs=100, verbose=1, callbacks=[
                  checkpoint])  # validation_data=(X_test,y_test),

### Plot the change in loss over epochs ###


# for key in ['loss', 'val_loss']:
#plt.plot(hist1.history['loss'], label=key)
# plt.legend()
# plt.show()

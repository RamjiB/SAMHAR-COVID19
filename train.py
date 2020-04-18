import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score,accuracy_score,classification_report,confusion_matrix,f1_score,recall_score,precision_score
from sklearn.model_selection import KFold,train_test_split

import os
from tqdm import tqdm_notebook
import tensorflow_addons as tfa

import warnings
sns.set()
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

tr = pd.read_csv('data/train_split.txt', sep=' ', header=None)
print('train shape: ', tr.shape)
tr.fillna('COVID-19', inplace=True)

x = []
y = []
classnames = {'normal': 0, 'bacteria': 1, 'viral': 2, 'COVID-19': 3}
for idx, img in tqdm_notebook(enumerate(range(tr.shape[0]))):
    img = cv2.cvtColor(cv2.imread('data/train/' + tr[1][idx]), cv2.COLOR_BGR2RGB)
    x.append(cv2.resize(img, (256, 256)))
    y.append(classnames[tr[2][idx]])
    del img

x = np.array(x)
y = np.array(y)

print('x shape: ', x.shape)
print('y shape: ', y.shape)

x_tr,x_va,y_tr,y_va = train_test_split(x,y,test_size=0.10,random_state=1205,stratify=y)
print(x_tr.shape,x_va.shape)

del x,y,tr

model = keras.models.Sequential()
model.add(keras.layers.Lambda(lambda x: x/255))
#Feature extraction layers
model.add(keras.layers.Convolution2D(32,3,input_shape=(256,256,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Convolution2D(32,3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D(strides=2))
model.add(keras.layers.Convolution2D(64,3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Convolution2D(64,3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D(strides=2))
model.add(keras.layers.Convolution2D(128,3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Convolution2D(128,3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D(strides=2))
model.add(keras.layers.Convolution2D(128,3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Convolution2D(128,3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D(strides=2))

#Classification layer
model.add(keras.layers.Convolution2D(128,4))

##average pooling
model.add(keras.layers.Flatten())

##Dropout(0.3)
model.add(keras.layers.Dropout(0.3))

#output
model.add(keras.layers.Dense(4,activation='softmax'))

model.compile(keras.optimizers.Adam(3e-5),loss='categorical_crossentropy',metrics=['accuracy',tfa.metrics.CohenKappa(num_classes=4,weightage='quadratic')])
csv = keras.callbacks.CSVLogger(CSV,append=True)
mcp = keras.callbacks.ModelCheckpoint(MODEL,monitor='val_accuracy',mode = 'max',verbose = 1,save_best_only=True,save_weights_only=True)
history = model.fit(x_tr, keras.utils.to_categorical(y_tr),epochs=30,batch_size=8,verbose =1,validation_data=(x_va,keras.utils.to_categorical(y_va)),callbacks=[mcp,csv])

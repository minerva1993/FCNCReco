from __future__ import print_function
import sys, os
import google.protobuf

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

import tensorflow as tf
import keras
from keras.utils import np_utils, multi_gpu_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback

class roc_callback(Callback):
  def __init__(self, training_data, validation_data):
      self.x = training_data[0]
      self.y = training_data[1]
      self.x_val = validation_data[0]
      self.y_val = validation_data[1]


  def on_train_begin(self, logs={}):
      return

  def on_train_end(self, logs={}):
      #compute roc only at the end of training
      y_pred = self.model.predict(self.x)
      roc = roc_auc_score(self.y, y_pred)
      y_pred_val = self.model.predict(self.x_val)
      roc_val = roc_auc_score(self.y_val, y_pred_val)
      print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)), str(round(roc_val,4))),end=100*' '+'\n')

      #print(self.y_val)
      #print(y_pred_val)
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      #fpr[0], tpr[0], thresholds0 = roc_curve(self.y_val[:,0], y_pred_val[:,0], pos_label=1)
      fpr[1], tpr[1], thresholds1 = roc_curve(self.y_val[:,1], y_pred_val[:,1], pos_label=1)
      #plt.plot(1-fpr[0], 1-(1-tpr[0]), 'b')#same as [1]
      plt.plot(tpr[1], 1-fpr[1], 'r')#HEP style ROC
      #plt.plot([0,1], [0,1], 'r--')
      #plt.legend(['class 1'], loc = 'lower right')
      plt.xlabel('Signal Efficiency')
      plt.ylabel('Background Rejection')
      plt.title('ROC Curve')
      plt.show()
      return

  def on_epoch_begin(self, epoch, logs={}):
      return

  """
  def on_epoch_end(self, epoch, logs={}):
      y_pred = self.model.predict(self.x)
      roc = roc_auc_score(self.y, y_pred)
      y_pred_val = self.model.predict(self.x_val)
      roc_val = roc_auc_score(self.y_val, y_pred_val)
      print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)), str(round(roc_val,4))),end=100*' '+'\n')
      return
  """

  def on_epoch_end(self, epoch, logs={}):
      return

  def on_batch_begin(self, batch, logs={}):
      return

  def on_batch_end(self, batch, logs={}):
      return

####################
#read input and skim
####################
data = pd.read_hdf('./ntuples/ttbarJetCombinations.h5')
#print(data.index.is_unique)#check if indices are duplicated
data["genMatch"] = (data['genMatch'] == 1111).astype(int)
data = shuffle(data)
NumEvt = data['genMatch'].value_counts()
#print(NumEvt)
print('bkg/sig events : '+ str(NumEvt.tolist()))
data = data.drop(data.query('genMatch < 1').sample(frac=.9, axis=0).index)
NumEvt2 = data['genMatch'].value_counts()
#print(NumEvt2)
print('bkg/sig events after bkg skim : '+ str(NumEvt2.tolist()))


############################
#drop phi and label features
############################
#col_names = list(data_train)
labels = data.filter(['genMatch'], axis=1)
labels = labels.values
labels = np_utils.to_categorical(labels)
data = data.drop(['nevt', 'file', 'GoodPV', 'EventCategory', 'EventWeight', 'genMatch',
                  'njets', 'nbjets_m',
                  'lepton_pt', 'lepton_eta', 'lepton_phi', 'MET', 'MET_phi', 'lepDPhi',
                  'jet0phi', 'jet0csv', 'jet0cvsl', 'jet0cvsb', 'jet0Idx',
                  'jet1phi', 'jet1csv', 'jet1cvsl', 'jet1cvsb', 'jet1Idx',
                  'jet2phi', 'jet2csv', 'jet2cvsl', 'jet2cvsb', 'jet2Idx',
                  'jet3phi', 'jet3csv', 'jet3cvsl', 'jet3cvsb', 'jet3Idx',
                  'jet12phi', 'jet23phi', 'jet31phi',
                  'lepWeta', 'lepWphi', 'lepTpt', 'lepTeta', 'lepTdeta', 'lepTphi', 'lepTdR',
                  'hadTpt', 'hadTphi',
                  ], axis=1)
data.astype('float32')
#print list(data_train)

########################
#Standardization and PCA
########################
scaler = StandardScaler()
scaler.fit(data)
data_sc = scaler.transform(data)

NCOMPONENTS = 46
pca = PCA(n_components=NCOMPONENTS)
X = pca.fit_transform(data_sc)
#pca_std = np.std(data_pca)
#print(data_pca_train.shape)


###############
#split datasets
###############
totcombi = len(data)
numTrain = int(round(totcombi*0.5))
numTest = int(round(totcombi*0.95))#the last 5 percent of data
X_train = data_sc[:numTrain]
X_test = data_sc[numTest:]
Y_train = labels[:numTrain]
Y_test = labels[numTest:]
#print str(totcombi)
#print str(len(X_train)) +' '+ str(len(Y_train)) +' ' + str(len(X_test)) +' '+ str(len(Y_test))
#print labels

#################################
#Keras model compile and training
#################################
a = 50
b = 0.1
init = 'glorot_uniform'

with tf.device("/cpu:0"):
  inputs = Input(shape=(46,))
  x = Dense(a, kernel_regularizer=l2(1E-2))(inputs)
  x = BatchNormalization()(x)

  branch_point1 = Dense(a, name='branch_point1')(x)

  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  x = BatchNormalization()(x)
  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  x = add([x, branch_point1])

  x = BatchNormalization()(x)
  branch_point2 = Dense(a, name='branch_point2')(x)

  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)
  x = BatchNormalization()(x)
  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  x = add([x, branch_point2])

  x = BatchNormalization()(x)
  branch_point3 = Dense(a, name='branch_point3')(x)

  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)
  x = BatchNormalization()(x)
  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  x = add([x, branch_point3])

  x = BatchNormalization()(x)
  branch_point4 = Dense(a, name='branch_point4')(x)

  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)
  x = BatchNormalization()(x)
  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  x = add([x, branch_point4])

  x = BatchNormalization()(x)
  branch_point5 = Dense(a, name='branch_point5')(x)

  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)
  x = BatchNormalization()(x)
  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  x = add([x, branch_point5])

  x = BatchNormalization()(x)
  branch_point6 = Dense(a, name='branch_point6')(x)

  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)
  x = BatchNormalization()(x)
  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  x = add([x, branch_point6])
  
  x = BatchNormalization()(x)
  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  predictions = Dense(2, activation='softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)

parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3), metrics=['binary_accuracy'])
#parallel_model.summary()
history = parallel_model.fit(X_train, Y_train, 
                             epochs=100, batch_size=1000, 
                             validation_data=(X_test, Y_test), 
                             #class_weight={ 0: 14, 1: 1 }, 
                             callbacks=[roc_callback(training_data=(X_train, Y_train), validation_data=(X_test, Y_test))]
                             )

model.save('model.h5')#save template model, rather than the model returned by multi_gpu_model.

#print(history.history.keys())
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('binary crossentropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


from __future__ import print_function
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from root_numpy import array2tree, tree2array
from ROOT import TFile, TTree

import tensorflow as tf
import keras
from keras.utils import np_utils, multi_gpu_model
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback

bestModel = sys.argv[1]
ver = '02'
configDir = '/home/minerva1993/deepReco/'
weightDir = 'recoTT'
scoreDir = 'scoreTT'

if not os.path.exists(configDir+scoreDir+ver):
  os.makedirs(configDir+scoreDir+ver)

test = os.listdir(configDir+scoreDir+ver)
for item in test:
  if item.endswith(".root"):
      os.remove(os.path.join(configDir+scoreDir+ver, item))

print('Start evaluation on '+ver+' samples with the model '+bestModel)

#for filename in os.listdir("/home/minerva1993/deepReco/j4b2_tt"):
for filename in os.listdir(configDir+"j4b2_tt/eff"):
  model_best = load_model(configDir+weightDir+ver+'/'+bestModel)
  print('model is loaded')
  infile = TFile.Open(configDir+'j4b2_tt/'+filename)
  print('processig '+filename)
  intree = infile.Get('test_tree')
  inarray = tree2array(intree)
  eval_df = pd.DataFrame(inarray)
  print(eval_df.shape)

  outfile = TFile.Open(scoreDir+ver+'/score_'+filename,'RECREATE')
  outtree = TTree("tree","tree")

  spectator = eval_df.filter(['nevt', 'file', 'EventCategory', 'genMatch', 'jet0Idx', 'jet1Idx', 'jet2Idx', 'jet3Idx', 'lepton_pt', 'MET', 'jet12m', 'lepTm', 'hadTm'], axis=1)
  eval_df = eval_df.drop(['nevt', 'file', 'GoodPV', 'EventCategory', 'EventWeight', 'genMatch',
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
  eval_df.astype('float32')

  eval_scaler = StandardScaler()
  eval_scaler.fit(eval_df)
  eval_df_sc = eval_scaler.transform(eval_df)
  X = eval_df_sc
  y = model_best.predict(X, batch_size=2000)
  y.dtype = [('KerasScore', np.float32)]
  y = y[:,1]
  array2tree(y, name='tree', tree=outtree)

  for colname, value in spectator.iteritems():
    spect = spectator[colname].values
    if colname == 'lepton_pt': branchname = 'lepPt'
    elif colname == 'MET'    : branchname = 'missinget'
    elif colname == 'jet12m' : branchname = 'whMass'
    elif colname == 'lepTm'  : branchname = 'leptMass'
    elif colname == 'hadTm'  : branchname = 'hadtMass'
    else: branchname = colname

    if branchname in ['nevt', 'file', 'EventCategory', 'genMatch', 'jet0Idx', 'jet1Idx', 'jet2Idx', 'jet3Idx' ]: spect.dtype = [(branchname, np.int32)]
    else:
      spect.dtype = [(branchname, np.float32)]
    #print(branchname)
    #print(spect.shape)
    array2tree(spect, name='tree', tree=outtree)

  outtree.Fill()
  outfile.Write()
  outfile.Close()




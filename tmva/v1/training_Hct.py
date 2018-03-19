#!/usr/bin/env python
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from ROOT import *
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

fout = TFile("output_keras_Hct.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader("keras_Hct1")
loader.AddVariable("njets", "I")
loader.AddVariable("nbjets_m",'I')
loader.AddVariable("lepWpt",'F')
loader.AddVariable("lepWeta",'F')
loader.AddVariable("lepWdphi",'F')
loader.AddVariable("lepWm",'F')
loader.AddVariable("jet0pt",'F')
loader.AddVariable("jet0eta",'F')
loader.AddVariable("jet0m",'F')
loader.AddVariable("jet0csv",'F')
loader.AddVariable("jet0cvsl",'F')
loader.AddVariable("jet0cvsb",'F')
loader.AddVariable("jet1pt",'F')
loader.AddVariable("jet1eta",'F')
loader.AddVariable("jet1m",'F')
loader.AddVariable("jet1csv",'F')
loader.AddVariable("jet1cvsl",'F')
loader.AddVariable("jet1cvsb",'F')
loader.AddVariable("jet2pt",'F')
loader.AddVariable("jet2eta",'F')
loader.AddVariable("jet2m",'F')
loader.AddVariable("jet2csv",'F')
loader.AddVariable("jet2cvsl",'F')
loader.AddVariable("jet2cvsb",'F')
loader.AddVariable("jet3pt",'F')
loader.AddVariable("jet3eta",'F')
loader.AddVariable("jet3m",'F')
loader.AddVariable("jet3csv",'F')
loader.AddVariable("jet3cvsl",'F')
loader.AddVariable("jet3cvsb",'F')
loader.AddVariable("jet12pt",'F')
loader.AddVariable("jet12eta",'F')
loader.AddVariable("jet12deta",'F')
loader.AddVariable("jet12dphi",'F')
loader.AddVariable("jet12m",'F')
loader.AddVariable("jet12DR",'F')
loader.AddVariable("jet23pt",'F')
loader.AddVariable("jet23eta",'F')
loader.AddVariable("jet23deta",'F')
loader.AddVariable("jet23dphi",'F')
loader.AddVariable("jet23m",'F')
loader.AddVariable("jet31pt",'F')
loader.AddVariable("jet31eta",'F')
loader.AddVariable("jet31deta",'F')
loader.AddVariable("jet31dphi",'F')
loader.AddVariable("jet31m",'F')
loader.AddVariable("lepTpt",'F')
loader.AddVariable("lepTeta",'F')
loader.AddVariable("lepTdeta",'F')
loader.AddVariable("lepTdphi",'F')
loader.AddVariable("lepTm",'F')
loader.AddVariable("hadTpt",'F')
loader.AddVariable("hadTeta",'F')
loader.AddVariable("hadTHbdeta",'F')
loader.AddVariable("hadTWbdeta",'F')
loader.AddVariable("hadTHbdphi",'F')
loader.AddVariable("hadTWbdphi",'F')
loader.AddVariable("hadTm",'F')

## Load input files
signalA = TFile("input/tmva_TopHct.root")
signalB = TFile("input/tmva_AntiTopHct.root")
background1 = TFile("input/tmva_tchannel.root")
background2 = TFile("input/tmva_tbarchannel.root")
background3 = TFile("input/tmva_tWchannel.root")
background4 = TFile("input/tmva_tbarWchannel.root")
background5 = TFile("input/tmva_ttbb.root")
background6 = TFile("input/tmva_ttbj.root")
background7 = TFile("input/tmva_ttcc.root")
background8 = TFile("input/tmva_ttLF.root")
background9 = TFile("input/tmva_ttother.root")

sigTreeA = signalA.Get("tmva_tree")
sigTreeB = signalB.Get("tmva_tree")
backgroundTree1 = background1.Get("tmva_tree")
backgroundTree2 = background2.Get("tmva_tree")
backgroundTree3 = background3.Get("tmva_tree")
backgroundTree4 = background4.Get("tmva_tree")
backgroundTree5 = background5.Get("tmva_tree")
backgroundTree6 = background6.Get("tmva_tree")
backgroundTree7 = background7.Get("tmva_tree")
backgroundTree8 = background8.Get("tmva_tree")
backgroundTree9 = background9.Get("tmva_tree")

loader.AddSignalTree(sigTreeA,0.06316)
loader.AddSignalTree(sigTreeB,0.06317)
loader.AddBackgroundTree(backgroundTree1,0.08782)
loader.AddBackgroundTree(backgroundTree2,0.07553)
loader.AddBackgroundTree(backgroundTree3,0.19063)
loader.AddBackgroundTree(backgroundTree4,0.19332)
loader.AddBackgroundTree(backgroundTree5,0.11397)
loader.AddBackgroundTree(backgroundTree6,0.09117)
loader.AddBackgroundTree(backgroundTree7,0.09117)
loader.AddBackgroundTree(backgroundTree8,0.09117)
loader.AddBackgroundTree(backgroundTree9,0.09117)

#background10 = TFile("input/tmva_Top_Hct.root")
#background11 = TFile("input/tmva_AntiTop_Hct.root")
#backgroundTree10 = background10.Get("tmva_tree")
#backgroundTree11 = background11.Get("tmva_tree")
#loader.AddBackgroundTree(backgroundTree10,0.06316)
#loader.AddBackgroundTree(backgroundTree11,0.06317)

loader.SetWeightExpression("EventWeight")
loader.AddSpectator("GoodPV")
loader.AddSpectator("EventCategory")
loader.AddSpectator("GenMatch")

sigCut = TCut("nevt %5 != 0")# GenMatch == 2")

bkgCut = TCut("nevt %5 != 0")# && GenMatch < 2")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    "nTrain_Signal=33000:nTrain_Background=370000:SplitMode=Random:NormMode=NumEvents:!V"
#    "nTrain_Signal=33000:nTrain_Background=90000:SplitMode=Random:NormMode=NumEvents:!V"
)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=250:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=15")

#Keras
a = 1000
b = 0.6
init = 'glorot_uniform'

inputs = Input(shape=(58,))
x = Dense(a, kernel_regularizer=l2(5E-3))(inputs)
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

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3), metrics=['binary_accuracy'])
model.save('model_Hct.h5')
#model.summary()

factory.BookMethod(loader, TMVA.Types.kPyKeras, 'Keras_TF',"H:!V:VarTransform=G,D,P:FilenameModel=model_Hct.h5:NumEpochs=40:BatchSize=1000")

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()


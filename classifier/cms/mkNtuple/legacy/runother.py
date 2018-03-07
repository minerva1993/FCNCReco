#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas, gROOT, TTree
from ROOT import *
import os, sys
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  #chain.SetProof();
  chain.Process("dataTuple.C+",name)
  """
  f = TFile("tmva_"+name+".root","update")
  tr = f.Get("tmva_tree")
  totalnevt = np.zeros(1, dtype=float)
  tr.Branch('totnevt', totalnevt, 'totnevt/D')
  nevt = tr.GetEntries()
  for i in xrange(nevt):
    #tr.GetEntry(i)
    totalnevt[0] = nevt
  tr.Fill()
  f.Write()
  f.Close()
  """
#p = TProof.Open("", "workers=8")

#runAna(inputdir+version, tuples, name)

"""
for mub in range(0, 66):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016B","Tree_ttbbLepJets_"+str(mub)+".root","DataSingleMuB_"+ str(mub))

for muc in range(0, 22):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016C","Tree_ttbbLepJets_"+str(muc)+".root","DataSingleMuC_"+ str(muc))

for mud in range(0, 37):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016D","Tree_ttbbLepJets_"+str(mud)+".root","DataSingleMuD_"+ str(mud))

for mue in range(0, 31):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016E","Tree_ttbbLepJets_"+str(mue)+".root","DataSingleMuE_"+ str(mue))

for muf in range(0, 23):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016F","Tree_ttbbLepJets_"+str(muf)+".root","DataSingleMuF_"+ str(muf))

for mug in range(0, 54):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016G","Tree_ttbbLepJets_"+str(mug)+".root","DataSingleMuG_"+ str(mug))

for muh2 in range(0, 58):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016H_v2","Tree_ttbbLepJets_"+str(muh2)+".root","DataSingleMuHv2_"+ str(muh2))

for muh3 in range(0, 2):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016H_v3","Tree_ttbbLepJets_"+str(muh3)+".root","DataSingleMuHv3_"+ str(muh3))
"""
for elb in range(0, 66):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016B","Tree_ttbbLepJets_"+str(elb)+".root","DataSingleEGB_"+ str(elb))

for elc in range(0, 22):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016C","Tree_ttbbLepJets_"+str(elc)+".root","DataSingleEGC_"+ str(elc))

for eld in range(0, 37):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016D","Tree_ttbbLepJets_"+str(eld)+".root","DataSingleEGD_"+ str(eld))

for ele in range(0, 31):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016E","Tree_ttbbLepJets_"+str(ele)+".root","DataSingleEGE_"+ str(ele))

for elf in range(0, 23):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016F","Tree_ttbbLepJets_"+str(elf)+".root","DataSingleEGF_"+ str(elf))

for elg in range(0, 54):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016G","Tree_ttbbLepJets_"+str(elg)+".root","DataSingleEGG_"+ str(elg))

for elh2 in range(0, 58):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016H_v2","Tree_ttbbLepJets_"+str(elh2)+".root","DataSingleEGHv2_"+ str(elh2))

for elh3 in range(0, 2):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016H_v3","Tree_ttbbLepJets_"+str(elh3)+".root","DataSingleEGHv3_"+ str(elh3))
"""
for st in range(0, 36):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleTop_t","Tree_ttbbLepJets_"+str(st)+".root","tchannel_"+str(st))

for stb in range(0, 21):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleTbar_t","Tree_ttbbLepJets_"+str(stb)+".root","tbarchannel_"+str(stb))

for stw in range(0, 8):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleTop_tW","Tree_ttbbLepJets_"+str(stw)+".root","tWchannel_"+str(stw))

for stbw in range(0, 7):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleTbar_tW","Tree_ttbbLepJets_"+str(stbw)+".root","tbarWchannel_"+str(stbw))

for dy in range(0, 73):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/DYJets","Tree_ttbbLepJets_"+str(dy)+".root","zjets_"+str(dy))

for dy1 in range(0, 12):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/DYJets_10to50_part1","Tree_ttbbLepJets_"+str(dy1)+".root","zjets10to50V2_part1_"+str(dy1))

for dy2 in range(0, 39):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/DYJets_10to50_part2","Tree_ttbbLepJets_"+str(dy2)+".root","zjets10to50V2_part2_"+str(dy2))

for dy3 in range(0, 24):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/DYJets_10to50_part3","Tree_ttbbLepJets_"+str(dy3)+".root","zjets10to50V2_part3_"+str(dy3))

for wj1 in range(0, 15):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/WJets_part1","Tree_ttbbLepJets_"+str(wj1)+".root","wjetsV2_part1_"+str(wj1))

for wj2 in range(0, 134):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/WJets_part2","Tree_ttbbLepJets_"+str(wj2)+".root","wjetsV2_part2_"+str(wj2))

for ww in range(0, 1):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/WW","Tree_ttbbLepJets_"+str(ww)+".root","ww_"+str(ww))

for wz in range(0, 2):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/WZ","Tree_ttbbLepJets_"+str(wz)+".root","wz_"+str(wz))

for zz in range(0, 1):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/ZZ","Tree_ttbbLepJets_"+str(zz)+".root","zz_"+str(zz))
"""

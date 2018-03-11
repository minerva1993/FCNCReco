#!/usr/bin/python

from ROOT import TFile, TChain, gSystem
import os, sys
#gROOT.SetBatch(True)

inputdir = sys.argv[1]
dataset = sys.argv[2]
tuples = sys.argv[3]
name = sys.argv[4]

def runAna(dir, file, name):
  f = TFile.Open(dir+'/'+file, "READ")
  if f.IsOpen():
    f = TFile.Open(dir+'/'+file, "READ")
    print 'processing '+dir+'/'+file
    chain = TChain("ttbbLepJets/tree","events")
    chain.Add(dir+'/'+file)
    chain.Process("MyAnalysis.C+",name)
    #print chain.GetCurrentFile().GetName()

    ## save Event Summary histogram ##
    out = TFile("temp/hist_"+name+".root","update")
    hevt = f.Get("ttbbLepJets/EventInfo")
    hevt.Write()
    out.Write()
    out.Close()

runAna(inputdir+'/'+dataset, tuples, name)

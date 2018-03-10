#!/usr/bin/python

from ROOT import *
import os, sys
gROOT.SetBatch(True)

inputdir = sys.argv[1]
dataset = sys.argv[2]
tuples = sys.argv[3]
name = sys.argv[4]

def runAna(dir, file, name):
  #print 'processing '+dir+'/'+file
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+'/'+file)
  chain.Process("MyAnalysis.C+",name)
  print chain.GetCurrentFile().GetName()

  #f = TFile(dir+'/'+file, "READ")
  f = TFile(chain.GetCurrentFile().GetName(),"READ")

  ## save Event Summary histogram ##
  #out = TFile("hist_"+name+".root","update")
  #hevt = f.Get("ttbbLepJets/EventInfo")
  #hevt.Write()
  #out.Write()
  #out.Close()

runAna(inputdir+'/'+dataset, tuples, name)


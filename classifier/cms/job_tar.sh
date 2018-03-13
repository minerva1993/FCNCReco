#!/bin/sh
VER=04

cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

tar zcvf 0.tar assign${VER}/assign_deepReco_TopHct_*.root
tar zcvf 1.tar assign${VER}/assign_deepReco_AntiTopHct_*.root
tar zcvf 2.tar assign${VER}/assign_deepReco_TopHut_*.root
tar zcvf 3.tar assign${VER}/assign_deepReco_AntiTopHut_*.root
tar zcvf 4.tar assign${VER}/assign_deepReco_DataSingleEGB*.root
tar zcvf 5.tar assign${VER}/assign_deepReco_DataSingleEGC*.root
tar zcvf 6.tar assign${VER}/assign_deepReco_DataSingleEGD*.root
tar zcvf 7.tar assign${VER}/assign_deepReco_DataSingleEGE*.root
tar zcvf 8.tar assign${VER}/assign_deepReco_DataSingleEGF*.root
tar zcvf 9.tar assign${VER}/assign_deepReco_DataSingleEGG*.root
tar zcvf 10.tar assign${VER}/assign_deepReco_DataSingleEGH*.root
tar zcvf 11.tar assign${VER}/assign_deepReco_DataSingleMuB*.root
tar zcvf 12.tar assign${VER}/assign_deepReco_DataSingleMuC*.root
tar zcvf 13.tar assign${VER}/assign_deepReco_DataSingleMuD*.root
tar zcvf 14.tar assign${VER}/assign_deepReco_DataSingleMuE*.root
tar zcvf 15.tar assign${VER}/assign_deepReco_DataSingleMuF*.root
tar zcvf 16.tar assign${VER}/assign_deepReco_DataSingleMuG*.root
tar zcvf 17.tar assign${VER}/assign_deepReco_DataSingleMuH*.root
tar zcvf 18.tar assign${VER}/assign_deepReco_ttbb_*.root
tar zcvf 19.tar assign${VER}/assign_deepReco_ttbj_*.root
tar zcvf 20.tar assign${VER}/assign_deepReco_ttcc_*.root
tar zcvf 21.tar assign${VER}/assign_deepReco_ttLF_*.root
tar zcvf 22.tar assign${VER}/assign_deepReco_ttother_*.root
tar zcvf 23.tar assign${VER}/assign_deepReco_tchannel_*.root
tar zcvf 24.tar assign${VER}/assign_deepReco_tWchannel_*.root
tar zcvf 25.tar assign${VER}/assign_deepReco_tbarchannel_*.root
tar zcvf 26.tar assign${VER}/assign_deepReco_tbarWchannel_*.root
tar zcvf 27.tar assign${VER}/assign_deepReco_wjetsV2_*.root
tar zcvf 28.tar assign${VER}/assign_deepReco_zjets_*.root
tar zcvf 29.tar assign${VER}/assign_deepReco_zjets10to50V2_*.root
tar zcvf 30.tar assign${VER}/assign_deepReco_ww_*.root
tar zcvf 31.tar assign${VER}/assign_deepReco_wz_*.root
tar zcvf 32.tar assign${VER}/assign_deepReco_zz_*.root

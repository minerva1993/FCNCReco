#!/bin/sh

rm tmva_*.root
hadd tmva_TopHct.root temp/tmva_TopHct_*.root
hadd tmva_AntiTopHct.root temp/tmva_AntiTopHct_*.root
hadd tmva_TopHut.root temp/tmva_TopHut_*.root
hadd tmva_AntiTopHut.root temp/tmva_AntiTopHut_*.root
hadd tmva_DataSingleEG.root temp/tmva_DataSingleEG*.root
hadd tmva_DataSingleMu.root temp/tmva_DataSingleMu*.root
hadd tmva_ttbb.root temp/tmva_ttbb_*.root
hadd tmva_ttbj.root temp/tmva_ttbj_*.root
hadd tmva_ttcc.root temp/tmva_ttcc_*.root
hadd tmva_ttLF.root temp/tmva_ttLF_*.root
hadd tmva_ttother.root temp/tmva_ttother_*.root
hadd tmva_tchannel.root temp/tmva_tchannel_*.root
hadd tmva_tWchannel.root temp/tmva_tWchannel_*.root
hadd tmva_tbarchannel.root temp/tmva_tbarchannel_*.root
hadd tmva_tbarWchannel.root temp/tmva_tbarWchannel_*.root
hadd tmva_wjetsV2.root temp/tmva_wjetsV2_*.root
hadd tmva_zjets.root temp/tmva_zjets_*.root
hadd tmva_zjets10to50V2.root temp/tmva_zjets10to50V2_*.root
hadd tmva_ww.root temp/tmva_ww_*.root
hadd tmva_wz.root temp/tmva_wz_*.root
hadd tmva_zz.root temp/tmva_zz_*.root

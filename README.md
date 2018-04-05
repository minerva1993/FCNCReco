# Reconstruction for FCNC and Analysis

This is the repository for top quark pair reconstruction using BDT for FCNC search.

There is several test steps for compile macros and check if the code is running well or not. To check and debug code, run python for root by giving exact arguments without using xargs.

  * Make file lists
```{.Bash}
find /path/to/ntuples/*/*.root -type f -printf "%h %f\n" >> files.txt #for path and file name only
```
You can use job.sh from legacy folders to write down out names as well as file path.


  * Make nutples for training, evaluation, assignment.

```{.Bash}
ssh compute-0-1 #(compute-0-2, compute-0-3)
cd classifier/cms/mkNtuple
##You MUST check OUTPUT directory
python runother.py /data/users/minerva1993/ntuple_Run2016/v4/production SingleTbar_t Tree_ttbbLepJets_0.root tbarchannel_0 #check compile
python runtt.py /data/users/minerva1993/ntuple_Run2016/v4/production TT_powheg_ttbj Tree_ttbbLepJets_0.root ttbj_0 #check compile
cat fileList/file_other1.txt | xargs -i -P$(nproc) -n4 python runother.py
cat fileList/file_other2.txt | xargs -i -P$(nproc) -n4 python runother.py
cat fileList/file_tt.txt | xargs -i -P$(nproc) -n4 python runtt.py
cat fileList/file_st.txt | xargs -i -P$(nproc) -n4 python runtt.py #for ST fcnc
```
To make ttbar reco signals, change output directory in makeTuple.C, modify b tagging restrictions, and run
```{.Bash}
cat fileList/file_recoTTsig1.txt | xargs -i -P$(nproc) -n4 python runtt.py
cat fileList/file_recoTTsig2.txt | xargs -i -P$(nproc) -n4 python runtt.py
cat fileList/file_recoTTsig3.txt | xargs -i -P$(nproc) -n4 python runtt.py
```

  * Evaluate classifier
```{.Bash}
#Make sure thar you're using correct file path in evaluation.py
cd classifier/cms
find mkNtuple/j4b2/*.root -type f -printf "04 %f\n" >> file_eval.txt #make another list containing classifier version and ntuple names
cat fileList/file_eval1.txt | xargs -i -P$(nproc) -n2 python evaluation.py
cat fileList/file_eval2.txt | xargs -i -P$(nproc) -n2 python evaluation.py
cat fileList/file_eval3.txt | xargs -i -P$(nproc) -n2 python evaluation.py
cat fileList/file_eval4.txt | xargs -i -P$(nproc) -n2 python evaluation.py
```

  * Assign best combinations
```{.Bash}
cd classifier/cms
find score04/*.root -type f -printf "%f\n" >> file_assign.txt #Make another list containing score ntuple names
root -l run.C'("score_deepReco_ttbb_10.root")' #check the code!
cat fileList/file_assign1.txt | xargs -i -P$(nproc) -n1 root -l -b run.C'("{}")'
cat fileList/file_assign2.txt | xargs -i -P$(nproc) -n1 root -l -b run.C'("{}")'
cat fileList/file_assign3.txt | xargs -i -P$(nproc) -n1 root -l -b run.C'("{}")'
cat fileList/file_assign4.txt | xargs -i -P$(nproc) -n1 root -l -b run.C'("{}")'
```

  * Histograms
```{.Bash}
cd analyzer
python runAnalysis.py /data/users/minerva1993/ntuple_Run2016/v4/production signalReco/TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut Tree_ttbbLepJets_0.root AntiTopHut_0 #Test main analyzer code
cat fileList/file_other1.txt | xargs -i -P$(nproc) -n4 python runAnalysis.py #This list is a copy from mkNtuple
cat fileList/file_other2.txt | xargs -i -P$(nproc) -n4 python runAnalysis.py
cat fileList/file_tt.txt | xargs -i -P$(nproc) -n4 python runAnalysis.py
source job_merge.sh #Make merged root files for each channels
python ratioPlot.py -b #If seg falut occur and stops, comment one line of importing packages from style.py, run ratioPlat.py again and revive the line and run ratioPlat.py.
```

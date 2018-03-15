# Reconstruction for FCNC and Analysis

This is the repository for top quark pair reconstruction using BDT for FCNC search.

There is several test steps for compile macros and check if the code is running well or not. To check and debug code, run python for root by giving exact arguments without using xargs.

  * Make nutples for training, evaluation, assignment.

  cd classifier/cms/mkNtuple

  cat fileList/file_other1.txt | xargs -i -P$(nproc) -n4 python runother.py

  cat fileList/file_other2.txt | xargs -i -P$(nproc) -n4 python runother.py

  cat fileList/file_tt.txt | xargs -i -P$(nproc) -n4 python runtt.py

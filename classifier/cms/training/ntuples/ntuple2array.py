import numpy as np
from numpy.lib.recfunctions import stack_arrays
from ROOT import *
from root_numpy import tree2array
import glob
import pandas as pd
import deepdish.io as io

ttbb = TFile.Open('deepReco_ttbb.root')
ttbb_tree = ttbb.Get('test_tree')
ttbb_array = tree2array(ttbb_tree)
ttbb_df = pd.DataFrame(ttbb_array)
#io.save('ttbb.h5', ttbb_df)

ttbj = TFile.Open('deepReco_ttbj.root')
ttbj_tree = ttbj.Get('test_tree')
ttbj_array = tree2array(ttbj_tree)
ttbj_df = pd.DataFrame(ttbj_array)
#io.save('ttbj.h5', ttbj_df)

ttcc = TFile.Open('deepReco_ttcc.root')
ttcc_tree = ttcc.Get('test_tree')
ttcc_array = tree2array(ttcc_tree)
ttcc_df = pd.DataFrame(ttcc_array)
#io.save('ttcc.h5', ttcc_df)

ttLF = TFile.Open('deepReco_ttLF.root')
ttLF_tree = ttLF.Get('test_tree')
ttLF_array = tree2array(ttLF_tree)
ttLF_df = pd.DataFrame(ttLF_array)
#io.save('ttLF.h5', ttLF_df)

ttother = TFile.Open('deepReco_ttother.root')
ttother_tree = ttother.Get('test_tree')
ttother_array = tree2array(ttother_tree)
ttother_df = pd.DataFrame(ttother_array)
#io.save('ttother.h5', ttother_df)

frames = [ttbb_df, ttbj_df, ttcc_df, ttLF_df, ttother_df]
result = pd.concat(frames, ignore_index=True)
io.save('ttbarJetCombinations.h5', result)

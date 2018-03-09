#!/bin/bash

declare -a arr=( "ttbb" "ttbj" "ttcc" "ttLF" "ttother" "TopHct" "AntiTopHct" "TopHut" "AntiTopHut" "tchannel" "tbarchannel" "tWchannel" "tbarWchannel" "zjets10to50V2_part1" "zjets10to50V2_part2" "zjets10to50V2_part3" "wjetsV2_part1" "wjetsV2_part2" "zjets" "zz" "ww" "wz" "DataSingleMuB" "DataSingleMuC" "DataSingleMuD" "DataSingleMuE" "DataSingleMuF" "DataSingleMuG" "DataSingleMuHv2" "DataSingleMuHv3" "DataSingleEGB" "DataSingleEGC" "DataSingleEGD" "DataSingleEGE" "DataSingleEGF" "DataSingleEGG" "DataSingleEGHv2" "DataSingleEGHv3" )

cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

MAX=134
NPERJOB=1

BEGIN=$(($1*$NPERJOB))
for i in `seq $BEGIN $(($BEGIN+$NPERJOB-1))`; do
  [ $i -ge $MAX ] && break
  for j in "${arr[@]}"; do
      filename='"score_deepReco_'${j}'_'${i}'.root"'
      root -l 'run.C('${filename}')'
  done
done

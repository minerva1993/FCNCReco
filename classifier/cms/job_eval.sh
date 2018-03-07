#!/bin/bash
#declare -a arr=("ttbb" "ttbj" "ttcc" "ttLF" "ttother" "TopHct" "AntiTopHct" "tchannel" "tbarchannel" "tWchannel" "tbarWchannel" "wjetsV2_part1" "wjetsV2_part2" "zjets" "zz" "ww" "wz")
declare -a arr=("zjets10to50V2_part1" "zjets10to50V2_part2" "zjets10to50V2_part3")

cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

MAX=134
NPERJOB=1

BEGIN=$(($1*$NPERJOB))
for j in "${arr[@]}"; do
  for i in `seq $BEGIN $(($BEGIN+$NPERJOB-1))`; do
      [ $i -ge $MAX ] && break
      python evaluation.py 04 deepReco_${j}_${i}.root
  done
done

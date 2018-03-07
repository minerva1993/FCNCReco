#!/bin/bash
declare -a arr=("ttbb" "ttbj" "ttcc" "ttLF" "ttother" "TopHct" "AntiTopHct" "TopHut" "AntiTopHut")

cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

MAX=134
NPERJOB=1

BEGIN=$(($1*$NPERJOB))
for j in "${arr[@]}"; do
  for i in `seq $BEGIN $(($BEGIN+$NPERJOB-1))`; do
      [ $i -ge $MAX ] && break
      filename='"score_deepReco_'${j}'_'${i}'.root"'
      root -l 'run.C('${filename}')'
  done
done

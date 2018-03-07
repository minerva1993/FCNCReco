declare -A arr
arr["tchannel"]="SingleTop_t" arr["tbarchannel"]="SingleTbar_t" arr["tWchannel"]="SingleTop_tW" arr["tbarWchannel"]="SingleTbar_tW"
arr["zjets"]="DYJets" arr["zjets10to50V2_part1"]="DYJets_10to50_part1" arr["zjets10to50V2_part2"]="DYJets_10to50_part2" arr["zjets10to50V2_part3"]="DYJets_10to50_part3"
arr["wjetsV2_part1"]="WJets_part1" arr["wjetsV2_part2"]="WJets_part2" arr["ww"]="WW" arr["wz"]="WZ" arr["zz"]="ZZ"
arr["DataSingleMuB"]="SingleMuon_Run2016B" arr["DataSingleMuC"]="SingleMuon_Run2016C" arr["DataSingleMuD"]="SingleMuon_Run2016D" arr["DataSingleMuE"]="SingleMuon_Run2016E"
arr["DataSingleMuF"]="SingleMuon_Run2016F" arr["DataSingleMuG"]="SingleMuon_Run2016G" arr["DataSingleMuHv2"]="SingleMuon_Run2016H_v2" arr["DataSingleMuHv3"]="SingleMuon_Run2016H_v3"
arr["DataSingleEGB"]="SingleElectron_Run2016B" arr["DataSingleEGC"]="SingleElectron_Run2016C" arr["DataSingleEGD"]="SingleElectron_Run2016D" arr["DataSingleEGE"]="SingleElectron_Run2016E"
arr["DataSingleEGF"]="SingleElectron_Run2016F" arr["DataSingleEGG"]="SingleElectron_Run2016G" arr["DataSingleEGHv2"]="SingleElectron_Run2016H_v2" arr["DataSingleEGHv3"]="SingleElectron_Run2016H_v3"

cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

MAX=134
NPERJOB=1

INPUTDIR="/xrootd/store/user/minerva1993/ntuple_jw/2016/v4/production"

BEGIN=$(($1*$NPERJOB))
for key in "${!arr[@]}"; do
  for i in `seq $BEGIN $(($BEGIN+$NPERJOB-1))`; do
      [ $i -ge $MAX ] && break
      filename='Tree_ttbbLepJets_'${i}'.root'
      outname=${key}'_'${i}
      python runother.py ${INPUTDIR} ${arr[${key}]} ${filename} ${outname}
      #echo ${key} ${arr[${key}]}
  done
done

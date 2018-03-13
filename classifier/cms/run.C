#include "idxToNtuple.h"
#include <TSystem.h>
#include <string>

R__LOAD_LIBRARY(idxToNtuple.C+)

void run( TString name )
{

  //TString path = "root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/reco/score04/";
  TFile *f = TFile::Open("root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/reco/score04/"+name, "READ");
  if (f->IsOpen()==kFALSE){
    f->Close();
    gSystem->Exit(0);
  }
  else f->Close();


  TChain assign("tree");
  assign.Add("root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/reco/score04/"+name);

  idxToNtuple t(&assign);

  t.Loop();

  gSystem->Exit(0);
}

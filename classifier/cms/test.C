#include "idxToNtuple.h"
#include <TSystem.h>

R__LOAD_LIBRARY(idxToNtuple.C+)

void test()
{
/*
  TFile *f = new TFile("/cms/ldap_home/minerva1993/fcnc/recoFCNC/classifier/cms/score04/score_deepReco_zjets10to50V2_part3_14.root", "READ");
  if (f->IsOpen()==kFALSE){
    f->Close();
    gSystem->Exit(0);
  }
  else f->Close();
*/

  TChain assign("tree");
  assign.Add("/cms/ldap_home/minerva1993/fcnc/recoFCNC/classifier/cms/score04/score_deepReco_ttbj_12.root");

  idxToNtuple t(&assign);

  t.Loop();

  gSystem->Exit(0);
}

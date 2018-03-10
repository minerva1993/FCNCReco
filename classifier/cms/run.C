#include "idxToNtuple.h"
#include <TSystem.h>

R__LOAD_LIBRARY(idxToNtuple.C+)

void run( TString name )
{

  TFile *f = new TFile("/cms/ldap_home/minerva1993/fcnc/recoFCNC/classifier/cms/score04/"+name);
  if (f->IsOpen()==kFALSE){
    f->Close();
    gSystem->Exit(0);
  }
  else f->Close();
  

  TChain assign("tree");
  assign.Add("/cms/ldap_home/minerva1993/fcnc/recoFCNC/classifier/cms/score04/"+name);

  idxToNtuple t(&assign);

  t.Loop();

  gSystem->Exit(0);
}

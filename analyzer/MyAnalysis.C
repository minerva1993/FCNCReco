#define MyAnalysis_cxx

#include "MyAnalysis.h"
#include <TH2.h>
#include <TStyle.h>

void MyAnalysis::Begin(TTree * /*tree*/)
{
  TString option = GetOption();
}

void MyAnalysis::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
   
   cout << "SlaveBegin" << endl;
   for(int ich=0; ich < 2; ich++){
     for(int i=0; i < 11; i++){
     
      h_NJet[ich][i] = new TH1D(Form("h_NJet_Ch%i_S%i_%s",ich,i,option.Data()), "Number of jets", 12, 0, 12);
      h_NJet[ich][i]->SetXTitle("Jet Multiplicity");
      h_NJet[ich][i]->Sumw2();
      fOutput->Add(h_NJet[ich][i]);

      h_NBJetCSVv2M[ich][i] = new TH1D(Form("h_NBJetCSVv2M_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (medium)", 6, 0, 6);
      h_NBJetCSVv2M[ich][i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2M)");
      h_NBJetCSVv2M[ich][i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2M[ich][i]);

      h_NBJetCSVv2T[ich][i] = new TH1D(Form("h_NBJetCSVv2T_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (tight)", 6, 0, 6);
      h_NBJetCSVv2T[ich][i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2T)");
      h_NBJetCSVv2T[ich][i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2T[ich][i]);

      h_NCJetM[ich][i] = new TH1D(Form("h_NCJetM_Ch%i_S%i_%s",ich,i,option.Data()), "Number of c tagged jets", 6, 0, 6);
      h_NCJetM[ich][i]->SetXTitle("c-tagged Jet Multiplicity (M)");
      h_NCJetM[ich][i]->Sumw2();
      fOutput->Add(h_NCJetM[ich][i]);  

      h_MET[ich][i] = new TH1D(Form("h_MET_Ch%i_S%i_%s",ich,i,option.Data()), "MET", 30,0,300);
      h_MET[ich][i]->SetXTitle("MET (GeV)");
      h_MET[ich][i]->Sumw2();
      fOutput->Add(h_MET[ich][i]);

      h_LepPt[ich][i] = new TH1D(Form("h_LepPt_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton p_{T}", 30,0,300);
      h_LepPt[ich][i]->SetXTitle("Lepton p_{T} (GeV)");
      h_LepPt[ich][i]->Sumw2();
      fOutput->Add(h_LepPt[ich][i]);

      h_LepPhi[ich][i] = new TH1D(Form("h_LepPhi_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton #phi", 30 ,0 ,3.2);
      h_LepPhi[ich][i]->SetXTitle("Lepton |#phi|");
      h_LepPhi[ich][i]->Sumw2();
      fOutput->Add(h_LepPhi[ich][i]);

      h_LepEta[ich][i] = new TH1D(Form("h_LepEta_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton #eta", 30 ,0 ,2.5);
      h_LepEta[ich][i]->SetXTitle("Lepton |#eta|");
      h_LepEta[ich][i]->Sumw2();
      fOutput->Add(h_LepEta[ich][i]);

      h_WMass[ich][i] = new TH1D(Form("h_WMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass", 20 ,0 ,200);
      h_WMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
      h_WMass[ich][i]->Sumw2();
      fOutput->Add(h_WMass[ich][i]);

      h_cJetPt[ich][i] = new TH1D(Form("h_cJetPt_Ch%i_S%i_%s",ich,i,option.Data()), "leading c jet (medium) pT", 30 , 0 ,300);
      h_cJetPt[ich][i]->SetXTitle("leading c Jet (medium) pT (GeV)");
      h_cJetPt[ich][i]->Sumw2();
      fOutput->Add(h_cJetPt[ich][i]);

      h_DPhi[ich][i] = new TH1D(Form("h_DPhi_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton MET #Delta#phi", 30 ,0 ,3.2);
      h_DPhi[ich][i]->SetXTitle("|#Delta#phi_{l,MET}|");
      h_DPhi[ich][i]->Sumw2();
      fOutput->Add(h_DPhi[ich][i]);

      h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 20 ,0 ,0.2);
      h_LepIso[ich][i]->SetXTitle("Relative Isolation");
      h_LepIso[ich][i]->Sumw2();
      fOutput->Add(h_LepIso[ich][i]);

      h_csvv2[ich][i] = new TH1D(Form("h_csvv2_Ch%i_S%i_%s",ich,i,option.Data()), "CSVv2", 20 ,0 ,1);
      h_csvv2[ich][i]->SetXTitle("CSVv2");
      h_csvv2[ich][i]->Sumw2();
      fOutput->Add(h_csvv2[ich][i]);

      h_cvsl[ich][i] = new TH1D(Form("h_cvsl_Ch%i_S%i_%s",ich,i,option.Data()), "CvsL", 20 , -0.1 ,1);
      h_cvsl[ich][i]->SetXTitle("CvsL");
      h_cvsl[ich][i]->Sumw2();
      fOutput->Add(h_cvsl[ich][i]);

      h_cvsb[ich][i] = new TH1D(Form("h_cvsb_Ch%i_S%i_%s",ich,i,option.Data()), "CvsB", 20 , 0.08 ,1);
      h_cvsb[ich][i]->SetXTitle("CvsB");
      h_cvsb[ich][i]->Sumw2();
      fOutput->Add(h_cvsb[ich][i]);

      h_DRFCNHkinLepWMass[ich][i] = new TH1D(Form("h_DRFCNHkinLepWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass (Lep)", 30 , 0, 300);
      h_DRFCNHkinLepWMass[ich][i]->SetXTitle("W Mass (Lep) (GeV)");
      h_DRFCNHkinLepWMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinLepWMass[ich][i]);

      h_DRFCNHkinHadWMass[ich][i] = new TH1D(Form("h_DRFCNHkinHadWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass (Had)", 30, 0, 300);
      h_DRFCNHkinHadWMass[ich][i]->SetXTitle("W Mass (Had) (GeV)");
      h_DRFCNHkinHadWMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHadWMass[ich][i]);

      h_DRFCNHkinHMass[ich][i] = new TH1D(Form("h_DRFCNHkinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs Mass (bb)", 30, 0,300);
      h_DRFCNHkinHMass[ich][i]->SetXTitle("Higgs Mass (GeV)");
      h_DRFCNHkinHMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHMass[ich][i]);

      h_DRFCNHkinDR[ich][i] = new TH1D(Form("h_DRFCNHkinDR_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta R of bb", 30, 0,4);
      h_DRFCNHkinDR[ich][i]->SetXTitle("#Delta R of b jets from Higgs");
      h_DRFCNHkinDR[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinDR[ich][i]);

      h_DRFCNHkinTopMWb[ich][i] = new TH1D(Form("h_DRFCNHkinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass (Lep)", 30 , 0, 400);
      h_DRFCNHkinTopMWb[ich][i]->SetXTitle("Top Mass (Lep) (GeV)");
      h_DRFCNHkinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMWb[ich][i]);

      h_DRFCNHkinTopMHc[ich][i] = new TH1D(Form("h_DRFCNHkinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from Hc/u", 30, 0, 400);
      h_DRFCNHkinTopMHc[ich][i]->SetXTitle("Top Mass (Had) (GeV)");
      h_DRFCNHkinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMHc[ich][i]);

      h_DRFCNHkinHPt[ich][i] = new TH1D(Form("h_DRFCNHkinHPt_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs p_{T}", 30, 0,300);
      h_DRFCNHkinHPt[ich][i]->SetXTitle("Higgs p_{T} (GeV)");
      h_DRFCNHkinHPt[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHPt[ich][i]);

      h_DRFCNHkinHdPhi[ich][i] = new TH1D(Form("h_DRFCNHkinHdPhi_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta#phi of bb from Higgs", 30, 0, 3.2);
      h_DRFCNHkinHdPhi[ich][i]->SetXTitle("|#Delta#phi_{bb}|");
      h_DRFCNHkinHdPhi[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHdPhi[ich][i]);

      h_DRFCNHkinHdEta[ich][i] = new TH1D(Form("h_DRFCNHkinHdEta_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta#eta of bb from Higgs", 30, 0, 2.5);
      h_DRFCNHkinHdEta[ich][i]->SetXTitle("|#Delta#eta_{bb}|");
      h_DRFCNHkinHdEta[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHdEta[ich][i]);

      h_DRFCNHkinHb1CSV[ich][i] = new TH1D(Form("h_DRFCNHkinHb1CSV_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs b_{1} CSVv2", 20, 0.8 ,1);
      h_DRFCNHkinHb1CSV[ich][i]->SetXTitle("Higgs b_{1} CSVv2 (GeV)");
      h_DRFCNHkinHb1CSV[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHb1CSV[ich][i]);

      h_DRFCNHkinHb2CSV[ich][i] = new TH1D(Form("h_DRFCNHkinHb2CSV_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs b_{2} CSVv2", 20, 0.8 ,1);
      h_DRFCNHkinHb2CSV[ich][i]->SetXTitle("Higgs b_{2} CSVv2 (GeV)");
      h_DRFCNHkinHb2CSV[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHb2CSV[ich][i]);

      h_DRFCNHkinLepTopPt[ich][i] = new TH1D(Form("h_DRFCNHkinLepTopPt_Ch%i_S%i_%s",ich,i,option.Data()), "Top p_{T} (Lep)", 30 , 0, 400);
      h_DRFCNHkinLepTopPt[ich][i]->SetXTitle("Top p_{T} (Lep) (GeV)");
      h_DRFCNHkinLepTopPt[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinLepTopPt[ich][i]);

      h_DRFCNHkinHadTopPt[ich][i] = new TH1D(Form("h_DRFCNHkinHadTopPt_Ch%i_S%i_%s",ich,i,option.Data()), "Top p_{T} (Had)", 30 , 0, 400);
      h_DRFCNHkinHadTopPt[ich][i]->SetXTitle("Top p_{T} (Had) (GeV)");
      h_DRFCNHkinHadTopPt[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHadTopPt[ich][i]);

      //GenInfo
      h_genDR[ich][i] = new TH1D(Form("h_genDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen b jets from Higgs", 30, 0, 4);
      h_genDR[ich][i]->SetXTitle("gen #Delta R");
      h_genDR[ich][i]->Sumw2();
      fOutput->Add(h_genDR[ich][i]);

      h_matchDR[ich][i] = new TH1D(Form("h_matchDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen matched b jets from Higgs", 30, 1, 4);
      h_matchDR[ich][i]->SetXTitle("gen matched #Delta R");
      h_matchDR[ich][i]->Sumw2();
      fOutput->Add(h_matchDR[ich][i]);

      h_genHm[ich][i] = new TH1D(Form("h_genHm_Ch%i_S%i_%s",ich,i,option.Data()), "Gen Higgs mass", 30, 0, 250);
      h_genHm[ich][i]->SetXTitle("gen Higgs Mass (GeV)");
      h_genHm[ich][i]->Sumw2();
      fOutput->Add(h_genHm[ich][i]);

      h_matchHm[ich][i] = new TH1D(Form("h_matchHm_Ch%i_S%i_%s",ich,i,option.Data()), "Gen matched Higgs mass", 30, 0, 250);
      h_matchHm[ich][i]->SetXTitle("gen matched Higgs Mass (GeV)");
      h_matchHm[ich][i]->Sumw2();
      fOutput->Add(h_matchHm[ich][i]);
      }
    }

    assignF = new TFile(Form("assign/ref_%s.root", option.Data()), "READ");
    assignT = (TTree*) assignF->Get("tree");
    int nevt = assignT->GetEntries();
    for(int i = 0; i < nevt; i++){
      assignT->GetEntry(i);
      double pt = assignT->GetLeaf("leptonPt")->GetValue(0);
      double met = assignT->GetLeaf("missingET")->GetValue(0);
      lepPt.push_back(pt);
      missET.push_back(met);
    }
} 

Bool_t MyAnalysis::Process(Long64_t entry)
{

    fReader.SetEntry(entry);
    TString option = GetOption();   

    int mode = 999; 
    mode = *channel;

    if( mode > 2) return kTRUE;

    float lep_SF = 1.0;
    if( !option.Contains("Data") ) lep_SF = lepton_SF[0];
    float genweight = *genWeight;
    float puweight = PUWeight[0];
    float jetsf = jet_SF_CSV_30[0];
    float EventWeight = puweight*genweight*lep_SF*jetsf;

    float relIso = *lepton_relIso; 

    //Object selection
    int njets = 0;
    int nbjets_m = 0; 
    int nbjets_t = 0; 
    int ncjets_m = 0; 

    TLorentzVector p4met;
    double met = *MET;
    double met_phi = *MET_phi;
    double apt = TMath::Abs(met);
    double met_x =  apt*TMath::Cos(met_phi);
    double met_y =  apt*TMath::Sin(met_phi);
    p4met.SetPxPyPzE( met_x, met_y, 0, met);

    TLorentzVector lepton;
    lepton.SetPtEtaPhiE(*lepton_pT, *lepton_eta, *lepton_phi, *lepton_E);

    double transverseM = transverseMass(lepton, p4met);
    double lepDphi = lepton.DeltaPhi(p4met);
    double cjetPt = 0;

    //for Goh's Kin fit
    int b_kin_bjetcode;
    bool match1 = false;
    bool match2 = false;
    double gendR = -1.0;
    double matchdR = -1.0;
    double genHm = 0;
    double matchHm = 0;

    //Selection Option
    bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
    bool makeIso = true;
    bool isIso = *lepton_isIso; 
    if( makeIso && !isIso ) return kTRUE;
    if( !makeIso && isIso ) return kTRUE;

    //Event selection 
    bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
    bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

  if( passmuon || passelectron ){

    vector<float> v_cjet_m;
    vector<TLorentzVector> v_bjet_m;
    vector<TLorentzVector> v_bjet_t;
    vector<TLorentzVector> v_jet;
    int jetIdx[4];
    TLorentzVector jetP4s[4];

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);
      if( !option.Contains("Data") ) jet = jet * jet_JER_Nom[iJet];

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        if( jet_CSV[iJet] > 0.8484 ){
          nbjets_m++;
          v_bjet_m.push_back(jet);
        }
        if( jet_CSV[iJet] > 0.9535 ){
          nbjets_t++;
          v_bjet_t.push_back(jet);
        }
        if( jet_CvsL[iJet] > -0.1 && jet_CvsB[iJet] > 0.08 ){
          ncjets_m++;
          v_cjet_m.push_back(jet.Pt());
        }
      }
    }

    //if( (option.Contains("Hct") || option.Contains("Hut")) && (*addHbjet1_pt)*(*addHbjet2_pt) == 0) return kTRUE;

    TLorentzVector hbjet1, hbjet2, genH;
    if(*addHbjet1_pt > 20 && *addHbjet2_pt > 20 && abs(*addHbjet1_eta) < 2.4 && abs(*addHbjet2_eta) < 2.4){
      hbjet1.SetPtEtaPhiE(*addHbjet1_pt, *addHbjet1_eta, *addHbjet1_phi, *addHbjet1_e);
      hbjet2.SetPtEtaPhiE(*addHbjet2_pt, *addHbjet2_eta, *addHbjet2_phi, *addHbjet2_e);

      genH = hbjet1 + hbjet2;
      gendR = hbjet1.DeltaR(hbjet2);
      genHm = genH.M();
    }

    if( ncjets_m != 0 ) cjetPt = *max_element(v_cjet_m.begin(), v_cjet_m.end());

    //Jet Assignment
    vector<double>::iterator iter;
    int evtIdx = 0;
    if( njets >= 4 && nbjets_m >= 2 ){
      for( iter = lepPt.begin(); iter != lepPt.end(); iter++){
        if( *iter == static_cast<float>(lepton.Pt()) ){
          int tmpIdx = distance(lepPt.begin(), iter);
          if( missET[tmpIdx] == met ) evtIdx = tmpIdx;
          else continue;
        }
      }
      dupCheck.push_back(evtIdx);

      assignT->GetEntry(evtIdx);
      int i0 = assignT->GetLeaf("idx0")->GetValue(0);
      int i1 = assignT->GetLeaf("idx1")->GetValue(0);
      int i2 = assignT->GetLeaf("idx2")->GetValue(0);
      int i3 = assignT->GetLeaf("idx3")->GetValue(0);
      jetIdx[0] = i0; jetIdx[1] = i1; jetIdx[2] = i2; jetIdx[3] = i3;
      //cout << i0 << endl;

      jetP4s[0].SetPtEtaPhiE(jet_pT[i0], jet_eta[i0], jet_phi[i0], jet_E[i0]);
      jetP4s[1].SetPtEtaPhiE(jet_pT[i1], jet_eta[i1], jet_phi[i1], jet_E[i1]);
      jetP4s[2].SetPtEtaPhiE(jet_pT[i2], jet_eta[i2], jet_phi[i2], jet_E[i2]);
      jetP4s[3].SetPtEtaPhiE(jet_pT[i3], jet_eta[i3], jet_phi[i3], jet_E[i3]);

      if( option.Contains("Hct") || option.Contains("Hut") ){
        if(hbjet1.DeltaR(jetP4s[1]) < 0.4 or hbjet1.DeltaR(jetP4s[2]) < 0.4) match1 = true;
        if(hbjet2.DeltaR(jetP4s[1]) < 0.4 or hbjet2.DeltaR(jetP4s[2]) < 0.4) match2 = true;
      }
    }

    /////Fill histograms
    int Ncuts = 11;
    bool eventSelection[Ncuts];
    for(unsigned int bcut=0; bcut < Ncuts; bcut++) eventSelection[bcut] = false;

    eventSelection[0] = true;
    eventSelection[1] = ( njets >= 4 );
    eventSelection[2] = ( njets >= 4 ) && ( nbjets_m == 2 );
    eventSelection[3] = ( njets >= 4 ) && ( nbjets_m == 3 );
    eventSelection[4] = ( njets >= 4 ) && ( nbjets_m == 4 );
    eventSelection[5] = ( njets >= 4 ) && ( nbjets_m >= 3 );
    eventSelection[6] = ( njets >= 4 ) && ( nbjets_m >= 4 );
    eventSelection[7] = ( njets >= 6 ); 
    eventSelection[8] = ( njets >= 6 ) && ( nbjets_m == 3 );
    eventSelection[9] = ( njets >= 6 ) && ( nbjets_m == 2 || nbjets_m == 3 );
    eventSelection[10] = ( njets >= 6 ) && ( nbjets_m >= 3 );

    for( int cut = 0; cut < 11; cut++){
      if(eventSelection[cut]){
        h_NJet[mode][cut]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][cut]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][cut]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][cut]->Fill(ncjets_m, EventWeight);
        h_LepPt[mode][cut]->Fill(lepton.Pt(), EventWeight);
        h_LepPhi[mode][cut]->Fill(lepton.Phi(), EventWeight);
        h_LepEta[mode][cut]->Fill(lepton.Eta(), EventWeight);
        h_MET[mode][cut]->Fill(*MET, EventWeight);
        h_WMass[mode][cut]->Fill(transverseM, EventWeight);
        h_DPhi[mode][cut]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][cut]->Fill(relIso, EventWeight);

        if( njets >=4 && nbjets_m >=2 ){
          for( int i = 0; i < 4; ++i ){
            const size_t j = jetIdx[i];
            h_csvv2[mode][cut]->Fill(jet_CSV[j],EventWeight);
            h_cvsl[mode][cut]->Fill(jet_CvsL[j],EventWeight);
            h_cvsb[mode][cut]->Fill(jet_CvsB[j],EventWeight);
          }
          h_DRFCNHkinLepWMass[mode][cut]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][cut]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][cut]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][cut]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][cut]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][cut]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_DRFCNHkinHPt[mode][cut]->Fill((jetP4s[1]+jetP4s[2]).Pt(),EventWeight);
          h_DRFCNHkinHdPhi[mode][cut]->Fill(abs(jetP4s[1].DeltaPhi(jetP4s[2])),EventWeight);
          h_DRFCNHkinHdEta[mode][cut]->Fill(abs((jetP4s[1]-jetP4s[2]).Eta()),EventWeight);
          h_DRFCNHkinHb1CSV[mode][cut]->Fill(jet_CSV[jetIdx[1]],EventWeight);
          h_DRFCNHkinHb2CSV[mode][cut]->Fill(jet_CSV[jetIdx[2]],EventWeight);
          h_DRFCNHkinLepTopPt[mode][cut]->Fill((lepton+p4met+jetP4s[0]).Pt(),EventWeight);
          h_DRFCNHkinHadTopPt[mode][cut]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).Pt(),EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][cut]->Fill(cjetPt, EventWeight);
        }

        if(genH.Pt() > 0){
          h_genDR[mode][cut]->Fill(gendR, EventWeight);
          h_genHm[mode][cut]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][cut]->Fill(gendR, EventWeight);
            h_matchHm[mode][cut]->Fill(genHm, EventWeight);
          }
        }
      }
    }//selection loopa
  }
  evtNum++;
  cout << evtNum << '\r';
  return kTRUE;
}

void MyAnalysis::SlaveTerminate()
{
  TString option = GetOption();
}
   

void MyAnalysis::Terminate()
{
  TString option = GetOption();

  TFile * out = TFile::Open(Form("hist_%s.root",option.Data()),"RECREATE");

   TList * l = GetOutputList();
   TIter next(l);
   TObject *object = 0;
   while( ( object = next()) ){
     const char * name = object->GetName();
     std::string str(name);
     if (str.find("h_") !=std::string::npos ){
       object->Write();
     }
   }

  out->Write();
  out->Close();

  //cout << lepcount << endl;
  auto it = unique( dupCheck.begin(), dupCheck.end() );
  cout << ((it == dupCheck.end()) ? "Unique\n" : "Duplicate(s)\n");
  
}

double MyAnalysis::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

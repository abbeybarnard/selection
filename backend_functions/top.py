# important parameters & some top level functions for the xsec analysis 

import math
import warnings
import numpy as np
import pandas as pd
import uproot

import uncertainty_functions 

import matplotlib.pyplot as plt


# FULL SIGNAL DEFINITION 
#### passes software trigger 
#### 'nu_pdg==12 and ccnc==0 
#### 1 proton > 40 MeV 
#### no pions above 40 MeV 
#### within a FV defined by: 10<=true_nu_vtx_x<=246 and -106<=true_nu_vtx_y<=106 and 10<=true_nu_vtx_z<=1026'

# The Pandora reco_in_fv_query is defined here:
# reco_in_fv_query = "10<=reco_nu_vtx_sce_x<=246 and -106<=reco_nu_vtx_sce_y<=106 and 10<=reco_nu_vtx_sce_z<=1026"

# The WireCell reco_in_fv_query is defined here:
reco_in_fv_query = "10<=wc3_reco_nuvtxX<=246 and -106<=wc3_reco_nuvtxY<=106 and 10<=wc3_reco_nuvtxZ<=1026"

### BDT TRAINING VARIABLES ###
# Need to be called by their pandora_ prefix in the dataframe
training_parameters = [
        "pandora_shr_tkfit_dedx_Y", "pandora_shr_tkfit_gap10_dedx_Y", "pandora_shr_tkfit_2cm_dedx_Y", "pandora_trkpid",
        "pandora_pfng2hipfrac", "pandora_pfng2hipavrg", "pandora_ng2hip_r1cm", "pandora_ng2hip_r10cm",
        "pandora_pfng2shrfrac", "pandora_pfng2shravrg", "pandora_shrmoliereavg", "pandora_subcluster",
        "pandora_tksh_distance", "pandora_trkshrhitdist2"]
    
selection_variables = ['slice_orig_pass_id', "reco_nu_vtx_sce_x", "reco_nu_vtx_sce_y", "reco_nu_vtx_sce_z", 
                      "contained_fraction",  'n_tracks_contained', 
                       'trk_energy', 'shrmoliereavg', 'trkpid', 
                      'n_showers_contained', 'n_tracks_contained', 'shr_tkfit_dedx_Y', 'tksh_distance', 
                       'tksh_angle', 'trkshrhitdist2', 'subcluster']

# # quality cuts
# BDT_PRE_QUERY = 'swtrig_pre==1 and nslice==1'
# BDT_PRE_QUERY += ' and ' + reco_in_fv_query
# BDT_PRE_QUERY +=' and contained_fraction>0.9'

# # signal definition - shower constraints
# BDT_PRE_QUERY += ' and n_showers_contained==1'

# # signal definition - track constraints
# BDT_PRE_QUERY += ' and n_tracks_contained>0'
# BDT_PRE_QUERY += ' and trk_energy>0.04' 
    
# BDT_LOOSE_CUTS = BDT_PRE_QUERY

# # loose shower constraints
# BDT_LOOSE_CUTS +=' and shr_score<0.3'
# BDT_LOOSE_CUTS += ' and shrmoliereavg<15'
# BDT_LOOSE_CUTS += ' and shr_tkfit_dedx_Y<7'

# # loose track constraints
# BDT_LOOSE_CUTS += ' and trkpid<0.35'
# BDT_LOOSE_CUTS += ' and tksh_distance<12'

# quality cuts
BDT_PRE_QUERY = 'swtrig_pre == 1'
BDT_PRE_QUERY += ' and slice_orig_pass_id == 1' 
BDT_PRE_QUERY += ' and ' + reco_in_fv_query
BDT_PRE_QUERY +=' and contained_fraction > 0.9'

# signal definition - shower constraints
BDT_PRE_QUERY += ' and n_showers_contained == 1'

# signal definition - michel electron phase space constraint
BDT_PRE_QUERY += ' and shr_energy_tot_cali > 0.07'

# signal definition - track constraints
BDT_PRE_QUERY += ' and n_tracks_contained > 0'
BDT_PRE_QUERY += ' and trk_energy > 0.04'

BDT_LOOSE_CUTS = BDT_PRE_QUERY

# loose shower constraints
BDT_LOOSE_CUTS += ' and shrmoliereavg < 15'

# loose track constraints (including NuGraph variables)
BDT_LOOSE_CUTS += ' and trkpid < 0.35'
BDT_LOOSE_CUTS += ' and tksh_distance < 12'

######################### analysis parameters ##############################
# set the POT & plots_path for plotting
# UPDATE based on the ntuples being used 
def parameters(ISRUN3): 
    
    rho_argon = 1.3836 # g/cm^3
    fv = 236*212*1016
    n_a = 6.022E23
    n_nucleons = 40
    m_mol = 39.95 #g/mol
    
    n_target = (rho_argon * fv * n_a * n_nucleons) / m_mol
    
    ext_tune = 0.98 # validated


   # Below, I've updated all the paths to have /exp at the beginning!  
   # FHC
    if not ISRUN3: 
        
        plots_path = "/Users/abarnard/phd/ccnp/uBNuMI_CC1eNp/plots/fhc/"
        cv_ntuple_path = "/Users/abarnard/phd/pelee_ntuples/run1/slimmed/" 
        full_ntuple_path = "/Users/abarnard/phd/pelee_ntuples/run1/unslimmed/"
        run4b_path = "/pnfs/uboone/persistent/users/uboonepro/surprise/run4b_full_samples/wc_processed/NuMI/"
        
        dirt_tune = 0.65 # validated
        
        beamon_pot = 2.0E20 # v5

        NUE = 'numi_nue_run1'
        
        # OLD INTEGRATED FLUX
        # integrated_flux_per_pot = 1.1864531e-11 # 1.18069E-11 # [ nu / cm^2 / POT]  , includes 60 MeV neutrino energy threshold
        
        # NEW VALUE
        integrated_flux_per_pot = 1.22343e-11 # [ nu / cm^2 / POT]  , includes 60 MeV neutrino energy threshold
        
        bdt_model = 'BDT_models/bdt_FHC_may2022_subset.model' 
        bdt_score_cut = 0.55
        
        detsys = 0.122
    
    # RHC 
    else: 
        
        plots_path = "/Users/abarnard/phd/ccnp/uBNuMI_CC1eNp/plots/rhc/"
        cv_ntuple_path = "/Users/abarnard/phd/pelee_ntuples/run3b/slimmed/"
        full_ntuple_path = "/Users/abarnard/phd/pelee_ntuples/run3b/unslimmed/"
        run4b_path = "/pnfs/uboone/persistent/users/uboonepro/surprise/run4b_full_samples/wc_processed/NuMI/"
        
        dirt_tune = 0.45 # validated
        
        beamon_pot = 5.013E20
        # beamon_pot = 2.527e+20
        
        NUE = 'checkout_MCC9.10_Run4b_NuMI_RHC_nue_overlay_surprise_v10_04_07_09_reco2_hist'

        # OLD INTEGRATED FLUX
        #integrated_flux_per_pot =  8.6283762e-12 #3.2774914e-12 # [ nu / cm^2 / POT]  , includes 60 MeV neutrino energy threshold

        # New integrated flux!
        integrated_flux_per_pot =  9.02463e-12 #3.2774914e-12 # [ nu / cm^2 / POT]  , includes 60 MeV neutrino energy threshold
        
        bdt_model = 'BDT_models/test_run4brhc_scaled_noext_noshrscore.model'

        bdt_score_cut = 0.55 
        
        detsys = 0.129 #0.133
        
    # create a dictionary 
    d = { 
        "plots_path" : plots_path, 
        "cv_ntuple_path" : cv_ntuple_path, 
        "full_ntuple_path" : full_ntuple_path, 
        "run4b_path" : run4b_path,
        "dirt_tune" : dirt_tune, 
        "ext_tune" : ext_tune, 
        "beamon_pot" : beamon_pot, 
        "NUE" : NUE, 
        "integrated_flux_per_pot" : integrated_flux_per_pot, 
        "n_target" : n_target,
        "bdt_model" : bdt_model, 
        "bdt_score_cut" : bdt_score_cut, 
        "detsys_flat" : detsys
    }
    
    return d


######################### plot categories ##############################
# everything must pass software trigger ! 

in_fv_query = "10<=pandora_true_nu_vtx_x<=246 and -106<=pandora_true_nu_vtx_y<=106 and 10<=pandora_true_nu_vtx_z<=1026"
out_fv_query = "((pandora_true_nu_vtx_x<10 or pandora_true_nu_vtx_x>246) or (pandora_true_nu_vtx_y<-106 or pandora_true_nu_vtx_y>106) or (pandora_true_nu_vtx_z<10 or pandora_true_nu_vtx_z>1026))"

numu_CC_Npi0 = '((pandora_nu_pdg==14 or pandora_nu_pdg==-14) and pandora_ccnc==0 and pandora_npi0>=1)'
numu_CC_0pi0 = '((pandora_nu_pdg==14 or pandora_nu_pdg==-14) and pandora_ccnc==0 and pandora_npi0==0)'

numu_NC_Npi0 = '((pandora_nu_pdg==14 or pandora_nu_pdg==-14) and pandora_ccnc==1 and pandora_npi0>=1)'
numu_NC_0pi0 = '((pandora_nu_pdg==14 or pandora_nu_pdg==-14) and pandora_ccnc==1 and pandora_npi0==0)'

nuebar_1eNp = '((pandora_nu_pdg==-12 and pandora_ccnc==0 and pandora_nproton>0 and pandora_npion==0 and pandora_npi0==0))'
nue_NC = '((pandora_nu_pdg==12 or pandora_nu_pdg==-12) and pandora_ccnc==1)'

nue_CCother = '(((pandora_nu_pdg==12 and pandora_ccnc==0) and (pandora_nproton==0 or pandora_npi0>0 or pandora_npion>0)) or (pandora_nu_pdg==-12 and pandora_ccnc==0 and (pandora_nproton==0 or pandora_npion>0 or pandora_npi0>0)))'

# less specific categories 
nue_other = '(((pandora_nu_pdg==12 or pandora_nu_pdg==-12) and pandora_ccnc==1) or (( (pandora_nu_pdg==12 or pandora_nu_pdg==-12) and pandora_ccnc==0) and (pandora_nproton==0 or pandora_npi0>0 or pandora_npion>0)))'
numu_Npi0 = '( (pandora_nu_pdg==14 or pandora_nu_pdg==-14) and pandora_npi0>=1)'
numu_0pi0 = '( (pandora_nu_pdg==14 or pandora_nu_pdg==-14) and pandora_npi0==0)'

# signal vs. not signal 
signal = in_fv_query + ' and (pandora_nu_pdg == 12 and pandora_ccnc == 0 and pandora_nproton > 0 and pandora_npion == 0 and pandora_npi0 == 0 and pandora_elec_e > 0.07)'
not_signal = '(' + out_fv_query + ' or (pandora_nu_pdg != 12) or (pandora_nu_pdg == 12 and pandora_ccnc == 1) or (pandora_nu_pdg == 12 and pandora_ccnc == 0 and (pandora_nproton == 0 or pandora_npi0 > 0 or pandora_npion > 0 or pandora_elec_e <= 0.07)))'

# for replacing nue CC 
in_AV_query = "-1.55<=pandora_true_nu_vtx_x<=254.8 and -116.5<=pandora_true_nu_vtx_y<=116.5 and 0<=pandora_true_nu_vtx_z<=1036.8"
nueCC_query = 'abs(pandora_nu_pdg)==12 and pandora_ccnc==0 and '+in_AV_query

########################################################################
#################### labels ############################################

labels = { 
    'signal' : ['$\\nu_e$ CC0$\pi$Np', 'orange'], 
    'numu_CC_Npi0' : ['$\\nu_\mu$ CC $\pi^{0}$', 'brown'],
    'numu_NC_Npi0' : ['$\\nu_\mu$ NC $\pi^{0}$', 'orangered'],
    'numu_NC_0pi0' : ['$\\nu_\mu$ NC', '#33FCFF'],
    'numu_CC_0pi0' : ['$\\nu_\mu$ CC', '#437ED8'],
    'nue_CCother': ['$\\nu_e$ CC other', '#05B415'], 
    'nue_NC': ['$\\nu_e$ NC', '#B8FF33'], 
    'outfv' : ['Out FV', 'orchid'], 
    'ext' : ['EXT', 'lightpink'],
    'nue_other' : ['$\\nu_e$ / $\\overline{\\nu}_e$  other', '#33db09'], 
    'numu_Npi0' : ['$\\nu_\\mu$ / $\\overline{\\nu}_\\mu$  $\pi^{0}$', '#EE1B1B'], 
    'numu_0pi0' : ['$\\nu_\\mu$ / $\\overline{\\nu}_\\mu$  other', '#437ED8'],
    'nuebar_1eNp' : ['$\\bar{\\nu}_e$ CC0$\pi$Np', 'gold']
}

########################################################################
# get rid of 30 MeV threshold on visible energy 
def vis_e_fix(df): 
    
    elec_e = np.array(df.elec_e)
    elec_ke = [0 for i in range(len(df))]
    
    # if electron energy is filled
    for i in range(len(elec_e)): 
        if elec_e[i] > 0: 
            elec_ke[i] = elec_e - 0.000511
            
    df['elec_ke'] = elec_ke
    
    print('added electron kinetic energy')
    
    E_vis = np.array(df.true_e_visible)
    E_vis_new = [0 for i in range(len(E_vis))]

    for i in range(len(E_vis)): 
    
        # for electrons above the 30 MeV threshold - do nothing 
        if elec_ke[i] > 0.03: 
            E_vis_new[i] = E_vis[i]

        # for electrons below the 30 MeV threshold - add to the visible energy 
        elif 0<elec_ke[i]<=0.03: 
            E_vis_new[i] = E_vis[i] + elec_ke[i] 
            
    df['true_e_visible2'] = E_vis_new
    
    print('added new visible energy')
    
    return df

########################################################################
# function to properly scale the RHC Run 3 (before & after software trigger change)
def pot_scale(df, df_type, ISRUN3, tune=True): 
    
    if tune: 
        print('Adding pot_scale column using dirt & EXT tune....')

        dirt_tune = parameters(ISRUN3)['dirt_tune']
        ext_tune = parameters(ISRUN3)['ext_tune']
    
    else: 
        print('Adding pot_scale column without dirt & EXT tune....')

        dirt_tune = 1
        ext_tune = 1     
    
    if ISRUN3: 

        overlay_pot = 2.33807e+21  # Run 4b
        # dirt_pot = 1.67392E21 # david's file
        beamon_pot = 5.013E20 # Normalizing to 1E20 POT for comparison
        nue_intrinsic_pot = 5.04447e+22 # Run 4b

        beamon_ntrig =  10349610.0 # Triggers of Run 3 data (from Patrick) (think this is an average)
        beamoff_ntrig = 15770854.05  # Triggers of Run 4b beam off 

        df_new = df.copy()
        
        if df_type == 'overlay': 
            df_new['pot_scale'] = beamon_pot/overlay_pot

        elif df_type == 'intrinsic': 
            df_new['pot_scale'] = beamon_pot/nue_intrinsic_pot

        # elif df_type == 'dirt': 
        #     df_new['pot_scale'] = (beamon_pot/dirt_pot)*dirt_tune

        elif df_type == "ext": 
            df_new['pot_scale'] = (beamon_ntrig/beamoff_ntrig)*ext_tune

        ### UNSCALED VERSION

        # overlay_pot = 2.33807e+21
        # # dirt_pot = 1
        # beamon_pot = 2.527e+20
        # nue_intrinsic_pot = 5.04447e+22

        # beamon_ntrig = 5418188.0
        # beamoff_ntrig = 15770854.05

        # df_new = df.copy()
        
        # if df_type == 'overlay': 
        #     df_new['pot_scale'] = beamon_pot/overlay_pot

        # elif df_type == 'intrinsic': 
        #     df_new['pot_scale'] = beamon_pot/nue_intrinsic_pot

        # # elif df_type == 'dirt': 
        # #     df_new['pot_scale'] = (beamon_pot/dirt_pot)*dirt_tune

        # elif df_type == "ext": 
        #     df_new['pot_scale'] = (beamon_ntrig/beamoff_ntrig)*ext_tune
    
    
    else: 
        
        overlay_pot =  2.33652E21  
        dirt_pot = 1.67392E21 # david's file
        beamon_pot = 2.0E20 #v5

        beamon_ntrig =  5268051.0 # v5 (EA9CNT_wcut)
        beamoff_ntrig = 9199232.74  # v5 (EXT_NUMIwin_FEMBeamTriggerAlgo)

        nue_intrinsic_pot = 2.37838E22
        
        df_new = df.copy()
        
        if df_type == 'overlay': 
            df_new['pot_scale'] = beamon_pot/overlay_pot

        elif df_type == 'intrinsic': 
            df_new['pot_scale'] = beamon_pot/nue_intrinsic_pot

        elif df_type == 'dirt': 
            df_new['pot_scale'] = (beamon_pot/dirt_pot)*dirt_tune

        elif df_type == "ext": 
            df_new['pot_scale'] = (beamon_ntrig/beamoff_ntrig)*ext_tune
        

    return df_new

########################################################################
# MC Stat Error Counting -- sum of the weights 
def mc_error(var, bins, xlow, xhigh, datasets): 
    
    # combine the datasets -- cuts should have already been applied 
    selected = pd.concat(datasets, ignore_index=True, sort=True)
    
    mc_stat = []
    
    for i in range(len(bins)-1):

        if i==len(bins)-2: # if the last bin, 
            bin_query = var+' >= '+str(bins[i])+' and '+var+' <= '+str(bins[i+1])
        
        else: 
            bin_query = var+' >= '+str(bins[i])+' and '+var+' < '+str(bins[i+1])

        mc_stat.append( np.sqrt(sum(selected.query(bin_query).totweight_data ** 2)) )
    
    return mc_stat
########################################################################
# Print out counts for each type of neutrino background inside the FV
def check_counts(in_fv, norm, cuts): 
    
    # (outdated)
    
    #################
    # in_fv --> in FV dataframe 
    # norm --> totweight_data
    # cuts --> cuts query for the dataframe 
    #################

    infv = in_fv.query(cuts)
    
    print('numu_Npi0 = '+str(round(sum(infv.query(numu_Npi0)[norm]), 1)))
    print('numu_0pi0 = '+str(round(sum(infv.query(numu_0pi0)[norm]), 1)))
    print('nue_other = '+str(round(sum(infv.query(nue_other)[norm]), 1)))
    print('  ')
    print('numu_NC_Npi0 = '+str(round(sum(infv.query(numu_NC_Npi0)[norm]), 1)))
    print('numu_CC_Npi0 = '+str(round(sum(infv.query(numu_CC_Npi0)[norm]), 1)))
    print('numu_NC_0pi0 = '+str(round(sum(infv.query(numu_NC_0pi0)[norm]), 1)))
    print('numu_CC_0pi0 = '+str(round(sum(infv.query(numu_CC_0pi0)[norm]), 1)))
    print('nue_CCother = '+str(round(sum(infv.query(nue_CCother)[norm]), 1)))
    print('nue_NC = '+str(round(sum(infv.query(nue_NC)[norm]), 1)))
    print('  ')
    print('signal = '+str(round(sum(infv.query(signal)[norm]), 1)))
    print('nuebar 1eNp = '+str(round(sum(infv.query(nuebar_1eNp)[norm]), 1)))
    print('  ')
    print('total nue/nuebar = '+str(round(sum(infv.query('nu_pdg==12 or nu_pdg==-12')[norm]), 1)))
    print('total numu/numubar = '+str(round(sum(infv.query('nu_pdg==14 or nu_pdg==-14')[norm]), 1)))
    print('  ')
    print('total  = '+str(round(sum(infv[norm]), 1)))
    
########################################################################
# detector variation - POT values
detvar_run1_fhc = {
    "LYRayleigh": 7.59732E20, #7.60573E20, 
    "LYDown": 7.43109E20, 
    "SCE": 7.39875E20, 
    "Recomb2": 7.59105E20, 
    "WireModX": 7.64918E20, 
    "WireModYZ": 7.532E20, 
    "WireModThetaXZ": 7.64282E20,
    "WireModThetaYZ_withSigmaSplines": 7.64543E20, 
    "CV": 7.59732E20
}

intrinsic_detvar_run1_fhc = {
    "LYRayleigh_intrinsic": 2.67655E22, #2.38081E22, 
    "LYDown_intrinsic": 2.24505E22, 
    "SCE_intrinsic": 2.60685E22, #2.39023E22, 
    "Recomb2_intrinsic":  2.60657E22, #2.38193E22, 
    "WireModX_intrinsic": 2.66184E22, #2.38318E22, 
    "WireModYZ_intrinsic":  2.62256E22, #2.38416E22,
    "WireModThetaXZ_intrinsic": 2.65175E22, #2.31518E22, 
    "WireModThetaYZ_withSigmaSplines_intrinsic": 2.62256E22, #2.31421E22, 
    "CV_intrinsic": 2.68294E22 #2.37261E22   
}

detvar_run3_rhc = {
    "LYAttenuation": 3.31177E20,
    "LYRayleigh":  3.15492E20, # 2.81E20, 
    "LYDown": 3.2338E20, #2.81E20, 
    "SCE": 3.33283E20, 
    "Recomb2": 3.29539E20, 
    "WireModX": 3.24286E20, 
    "WireModYZ": 3.36399E20, 
    "WireModThetaXZ": 3.20027E20,
    "WireModThetaYZ_withSigmaSplines": 3.35762E20, 
    "CV": 2.87219E20 #2.72E20
    
}

intrinsic_detvar_run3_rhc = {
    "LYAttenuation_intrinsic": 2.5392E22,
    "LYRayleigh_intrinsic": 2.53581E22, 
    "LYDown_intrinsic": 2.53082E22, 
    "SCE_intrinsic": 2.54153E22,  
    "Recomb2_intrinsic": 2.54549E22,  
    "WireModX_intrinsic": 2.50092E22, 
    "WireModYZ_intrinsic": 2.54089E22, 
    "WireModThetaXZ_intrinsic": 2.44365E22, 
    "WireModThetaYZ_withSigmaSplines_intrinsic":2.5992E22, 
    "CV_intrinsic": 2.5392E22
    
}
########################################################################
# corrected visible energy variable - account for electrons below 30 MeV 
def visible_energy_nothres(df): 
    
    # Handle both prefixed and unprefixed column names
    if 'pandora_elec_e' in df.columns:
        # Combined DataFrame with prefixes
        elec_e_col = 'pandora_elec_e'
        true_e_vis_col = 'pandora_true_e_visible'
        elec_ke_col = 'pandora_elec_ke'
    elif 'elec_e' in df.columns:
        # Original DataFrame without prefixes  
        elec_e_col = 'elec_e'
        true_e_vis_col = 'true_e_visible'
        elec_ke_col = 'elec_ke'
    else:
        raise KeyError("Could not find electron energy column. Expected 'pandora_elec_e' or 'elec_e'")
    
    df[elec_ke_col] = df[elec_e_col] - 0.000511
    pandora_elec_ke = list(df[elec_ke_col])

    pandora_E_vis = np.array(df[true_e_vis_col])
    pandora_E_vis_new = [0 for i in range(len(pandora_E_vis))]

    for i in range(len(pandora_E_vis)): 
    
        # for electrons above the 30 MeV threshold - do nothing 
        if pandora_elec_ke[i] > 0.03 or pandora_elec_ke[i] < 0: 
            pandora_E_vis_new[i] = pandora_E_vis[i]

        # for electrons below the 30 MeV threshold - add to the total visible energy 
        elif 0<pandora_elec_ke[i]<=0.03: 
            pandora_E_vis_new[i] = pandora_E_vis[i] + pandora_elec_ke[i] 

    # Use appropriate column name for the output
    if 'pandora_elec_e' in df.columns:
        df['pandora_true_e_visible2'] = pandora_E_vis_new
    else:
        df['true_e_visible2'] = pandora_E_vis_new
    
########################################################################
def flugg_reweight(df, isrun3): 
    
    horn_current = '' 
    if isrun3: 
        horn_current = "RHC"
    else: 
        horn_current = "FHC"
    
    flavor = {14: 'numu', -14: 'numubar', 12: 'nue', -12:'nuebar'}
    
    energy = list(df.nu_e)
    angle = list(df.thbeam)
    pdg = list(df.nu_pdg)
    
    file = uproot.open('/Users/abarnard/phd/ccnp/flux/flugg/flugg_2dratios_ppfx.root')
    
    flugg_weights = []
    
    for i in range(len(pdg)): 
        
        #print("i = ", i)
        #print('energy = ', energy[i])
        #print('flavor = ', flavor[pdg[i]])
        
        #hist = file["ratio_" + flavor[pdg[i]] + "_" + horn_current]
        hist = file["flugg_" + flavor[pdg[i]] + "_" + horn_current + "_" + "ratio"]
        
        # x-axis
        energy_bins = (hist.edges)[0]
        
        #y-axis
        theta_bins = (hist.edges)[1]
        
        weights = hist.values
        
        x = None
        y = None
        
        #print('hist name = ', "flugg_" + flavor[pdg[i]] + "_" + horn_current + "_" + "ratio")
        #print('energy bins = ', energy_bins)
        #print('theta bins = ', theta_bins)
        #print('weights = ', weights)
        
        # ENERGY LOOP OVER BINS 
        for j in range(len(energy_bins)-1): 

            # if energy > 10 GeV 
            if energy[i] > 10.0: 
                break 
            
            # for all other bins except last bin 
            if j != len(energy_bins)-2:
                if energy[i] >= energy_bins[j] and energy[i] < energy_bins[j+1]: # if the energy is inside the energy bin 
                    #print('testing energy bins:', theta_bins[j], theta_bins[j+1])
                    x = j
                    break
                else: 
                    continue

            # need inclusive top edge for the last bin 
            elif j == len(energy_bins)-2: 
                
                if energy[i] >= energy_bins[j] and energy[i] <= energy_bins[j+1]: 
                    #print('testing energy bins:', theta_bins[j], theta_bins[j+1])
                    x = j
                    break
                    #flugg_weights.append(weights[j])
                    #print('energy bin ', energy_bins[j], " to ", energy_bins[j+1], "; weight = ", weights[j])
                    
                else: 
                    print(i, "No weights exist for this neutrino energy!", energy[i])
                    break 
                    
            
        for k in range(len(theta_bins)-1): 

            # if energy > 10 GeV 
            if energy[i] > 10.0: 
                break 

            #print('angle = ', angle[i])
            
            # for all bins except the last bin 
            if k != len(theta_bins)-2: 
                if angle[i] >= theta_bins[k] and angle[i] < theta_bins[k+1]: 
                    #print('testing theta bins:', theta_bins[k], theta_bins[k+1])
                    y = k 
                    break 
                else: 
                    continue
                    
            # last bin - inclusive top edge 
            elif k == len(theta_bins)-2: 
                if angle[i] >= theta_bins[k] and angle[i] <= theta_bins[k+1]: 
                    #print('testing theta bins:', theta_bins[k], theta_bins[k+1])
                    y = k
                    break
            
                else: 
                    print(i, "No weights exist for this angle!", angle[i])
                    break 
                
        if energy[i] > 10.0: 
            flugg_weights.append(1.0)
        else: 
            flugg_weights.append(weights[x][y])
            
        x = None
        y = None
                    
    df['flugg_reweight'] = flugg_weights
    return df 
    
    
########################################################################
# scales to standard overlay 
def generated_signal(ISRUN3, var, bins, xlow, xhigh, cuts=None, weight='totweight_data', genie_sys=None, isFlugg=False):
    
    # print('WARNING: generated_signal now scales to beam on POT unless otherwise specified! --> make sure to update functions using this!')
        
    fold = "nuselection"
    tree = "NeutrinoSelectionFilter"
    
    variables = ["swtrig_pre", 'run', "nu_pdg", "ccnc", "nproton", "npion", "npi0", 
                "true_nu_vtx_x", "true_nu_vtx_y", "true_nu_vtx_z", "ppfx_cv", "weightSplineTimesTune", "weightTune",
                "nslice", 
                 "elec_e", "shr_energy_cali", 
                 "NeutrinoEnergy2", "true_e_visible", "tksh_angle", "nu_e", "true_nu_px", "true_nu_py", "true_nu_pz"] # "opening_angle", 
    
    
    if var not in variables: 
        if var != "true_e_visible2": 
            if var != "NeutrinoEnergy2_GeV": 
                variables.append(var)
    
    if genie_sys: 
        if isinstance(genie_sys, list): 
            variables = variables + genie_sys
            
        else: 
            variables.append(genie_sys)

    
    # This needs to be full_ntuple_path!
    f = uproot.open(parameters(ISRUN3)['run4b_path']+parameters(ISRUN3)['NUE']+".root")[fold][tree]
    df = pd.DataFrame(f.arrays(variables, library="np"))

    # Added in the ppfx_cv cleaning here 
    df.loc[ df['ppfx_cv'] <= 0, 'ppfx_cv' ] = 1.
    df.loc[ df['ppfx_cv'] == np.inf, 'ppfx_cv' ] = 1.
    df.loc[ df['ppfx_cv'] > 30, 'ppfx_cv' ] = 1.
    df.loc[ np.isnan(df['ppfx_cv']) == True, 'ppfx_cv' ] = 1.
    
    df.loc[ df['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
    df.loc[ df['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
    df.loc[ df['weightSplineTimesTune'] > 30, 'weightSplineTimesTune' ] = 1.
    df.loc[ np.isnan(df['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
    
    df.loc[ df['weightTune'] <= 0, 'weightTune' ] = 1.
    df.loc[ df['weightTune'] == np.inf, 'weightTune' ] = 1.
    df.loc[ df['weightTune'] > 30, 'weightTune' ] = 1.
    df.loc[ np.isnan(df['weightTune']) == True, 'weightTune' ] = 1.

    df['is_signal'] = np.where((df.nu_pdg==12) & (df.ccnc==0) & (df.nproton>0) & (df.npion==0) & (df.npi0==0)
                             & (10 <= df.true_nu_vtx_x) & (df.true_nu_vtx_x <= 246)
                             & (-106 <= df.true_nu_vtx_y) & (df.true_nu_vtx_y <= 106)
                             & (10 <= df.true_nu_vtx_z) & (df.true_nu_vtx_z <= 1026)
                             & (df.elec_e>0.07), True, False) # Set the Michel electron phase space veto here
    
    # Defining the reconstructed neutrino energy in GeV (Pandora)
    df['NeutrinoEnergy2_GeV'] = df['NeutrinoEnergy2']/1000

    visible_energy_nothres(df)
    
    if isFlugg: 
        print('Reweighting with flugg ratios....')
        df = addAngles(df)
        df = flugg_reweight(df, ISRUN3)
    
    
    df_signal = df.query('is_signal==True').copy()
    df_signal = pot_scale(df_signal, 'intrinsic', ISRUN3)
    
    #print('Tune weight is off!')
    #df_signal['weightSplineTimesTune'] = [1 for x in range(len(df_signal))]
    #df_signal['weightTune'] = [1 for x in range(len(df_signal))]

    df_signal['totweight_data'] = df_signal['ppfx_cv']*df_signal['pot_scale']*df_signal['weightSplineTimesTune']
    df_signal['totweight_intrinsic'] = df_signal['ppfx_cv']*df_signal['weightSplineTimesTune'
    ]
    
    if isFlugg:
        df_signal['totweight_data_flugg'] = df_signal['ppfx_cv']*df_signal['pot_scale']*df_signal['weightSplineTimesTune']*df_signal['flugg_reweight']
    
    
    if genie_sys=='weightsGenie': 
        df_signal[genie_sys] = df_signal[genie_sys]/1000
        
        for ievt in range(df_signal.shape[0]):
            # check for NaNs separately        
            if np.isnan(df_signal['weightsGenie'].iloc[ievt]).any() == True: 
                df_signal['weightsGenie'].iloc[ievt][ np.isnan(df_signal['weightsGenie'].iloc[ievt]) ] = 1.

            # This was originally 60, but I think it should be 30.
            reweightCondition = ((df_signal['weightsGenie'].iloc[ievt] > 30) | (df_signal['weightsGenie'].iloc[ievt] < 0)  | 
                                 (df_signal['weightsGenie'].iloc[ievt] == np.inf) | (df_signal['weightsGenie'].iloc[ievt] == np.nan))
            df_signal['weightsGenie'].iloc[ievt][ reweightCondition ] = 1.

            # if no variations exist for the event
            if not list(df_signal['weightsGenie'].iloc[ievt]): 
                df_signal['weightsGenie'].iloc[ievt] = [1.0 for k in range(600)]
                

    if cuts: 
        df_signal = df_signal.query(cuts)
        
    # make a histogram of the generated SIGNAL ONLY events (CV)
    n, b, p = plt.hist(df_signal[var], bins, histtype='bar', range=[xlow, xhigh], weights=df_signal[weight])
    plt.close()
    
    if genie_sys: 
        if isinstance(genie_sys, list): 
           df_weights = df_signal[genie_sys+['weightTune', 'totweight_data', var]].copy() 
            
        else: 
            df_weights = df_signal[[genie_sys, 'weightTune', 'totweight_data', var]].copy()
        
    else: 
        df_weights = None
       
    generated_sumw2 = []
    if weight=='totweight_data': 
        
        for i in range(len(bins)-1): 
            
            if i==len(bins)-2: 
                bin_query = var+'>='+str(bins[i])+' and '+var+'<='+str(bins[i+1])
            else: 
                bin_query = var+'>='+str(bins[i])+' and '+var+'<'+str(bins[i+1])

            generated_sumw2.append( sum(df_signal.query(bin_query).totweight_data ** 2) )
    
    elif weight=='totweight_intrinsic': 
        
        for i in range(len(bins)-1): 
            
            if i==len(bins)-2: 
                bin_query = var+'>='+str(bins[i])+' and '+var+'<='+str(bins[i+1])
            else: 
                bin_query = var+'>='+str(bins[i])+' and '+var+'<'+str(bins[i+1])

            generated_sumw2.append( sum(df_signal.query(bin_query).totweight_intrinsic ** 2) )

    return n.tolist(), df_weights, generated_sumw2
    
########################################################################
# add angles in beam & detector coordinates
def addAngles(df): 
    
    ## rotation matrix -- convert detector to beam coordinates
    R = [
        [0.921,   4.625e-05,     -0.3895],
        [0.02271,    0.9983,     0.05383],
        [0.3888,   -0.05843,      0.9195]
    ]
    det_origin_beamcoor = [5502.0, 7259.0,  67270.0]
     
    # angles in detector coordinates
    df['pandora_thdet'] = np.arctan2(((df['pandora_true_nu_px']*df['pandora_true_nu_px'])+(df['pandora_true_nu_py']*df['pandora_true_nu_py']))**(1/2), df['pandora_true_nu_pz'])*(180/math.pi)
    df['pandora_phidet'] = np.arctan2(df['pandora_true_nu_py'], df['pandora_true_nu_px'])*(180/math.pi)
        
    # get true momentum in beam coordinates
    df['pandora_true_nu_px_beam'] = R[0][0]*df['pandora_true_nu_px'] + R[0][1]*df['pandora_true_nu_py'] + R[0][2]*df['pandora_true_nu_pz']
    df['pandora_true_nu_py_beam'] = R[1][0]*df['pandora_true_nu_px'] + R[1][1]*df['pandora_true_nu_py'] + R[1][2]*df['pandora_true_nu_pz']
    df['pandora_true_nu_pz_beam'] = R[2][0]*df['pandora_true_nu_px'] + R[2][1]*df['pandora_true_nu_py'] + R[2][2]*df['pandora_true_nu_pz']
    
    # angles in beam coordinates
    df['pandora_thbeam'] = np.arctan2(((df['pandora_true_nu_px_beam']*df['pandora_true_nu_px_beam'])+(df['pandora_true_nu_py_beam']*df['pandora_true_nu_py_beam']))**(1/2), df['pandora_true_nu_pz_beam'])*(180/math.pi)
    df['pandora_phibeam'] = np.arctan2(df['pandora_true_nu_py_beam'], df['pandora_true_nu_px_beam'])*(180/math.pi)
        
    return df


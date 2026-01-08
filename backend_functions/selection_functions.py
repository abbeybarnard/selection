## UPDATE THE PLOTS_PATH FOR SAVING DISTRIBUTIONS ## 

import sys

import math
import warnings
import importlib 

#import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib.pylab as pylab
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
import csv
import uproot

sys.path.insert(0, 'backend_functions')

import NuMIDetSys

importlib.reload(NuMIDetSys)
NuMIDetSysWeights = NuMIDetSys.NuMIDetSys()

import top 
importlib.reload(top)
from top import *

import uncertainty_functions
from uncertainty_functions import * 

#import ROOT
#from ROOT import TH1F, TH2F, TDirectory, TH1D

from IPython.display import display


########################################################################
################## inputs to functions: ################################
# datasets: [infv, outfv, cosmic, ext, data]

# infv: overlay & dirt events with truth vtx in FV 
# outfv: overlay & dirt events with truth vtx in FV that are classified as neutrinos
# cosmic: overlay & dirt events with true vtx in FV that get misclassified as cosmic 
# ext: beam OFF data
# data:  beam ON data 

########################################################################
######################## selection functions ###########################

# use offline flux weights
def offline_flux_weights(df, ISRUN3): 
    
    if ISRUN3: 
        print("No ppfx maps for RHC!")
    
    else: 
        f = ROOT.TFile.Open("/Users/abarnard/phd/ccnp/uBNuMI_CC1eNp/ppfx_maps.root", "READ")
    
    numu_map = f.Get("numu_ratio")
    numubar_map = f.Get("numubar_ratio")
    nue_map = f.Get("nue_ratio")
    nuebar_map = f.Get("nuebar_ratio")
    
    nu_flav = list(df['nu_pdg'])
    angle = list(df['thbeam'])
    true_energy = list(df['nu_e'])

    fluxweights = []

    for i in range(len(nu_flav)): 
        if nu_flav[i]==14: 
            h = numu_map
        elif nu_flav[i]==-14: 
            h = numubar_map
        elif nu_flav[i]==12: 
            h = nue_map
        elif nu_flav[i]==-12: 
            h = nuebar_map
        else: 
            print("No map to match PDG code!")
        
        fluxweights.append( h.GetBinContent(h.FindBin(true_energy[i], angle[i])) )

    df['ppfx_cv'] = fluxweights
    #mc_df[0]['weightFlux'] = fluxweights
    #mc_df[1]['weightFlux'] = [1 for i in range(len(mc_df[1]))] # for now 
    
    f.Close()
    
    return df
    
########################################################################
# error on the data/MC ratio 
def get_ratio_err(n_data, n_mc): 
    
    err = []
    for i in range(len(n_data)): 
        
        # divide the counting error by n_mc - the MC error is handled by the systematics band 
        
        if n_data[i]>=0: 
            err.append( math.sqrt(n_data[i])/n_mc[i] )
        else: 
            err.append(0)

   # err = []
   # for i in range(len(n_data)): 
   #     err.append( n_data[i]/n_mc[i] * math.sqrt( (math.sqrt(n_data[i]) / n_data[i])**2))# + (math.sqrt(n_mc[i]) / n_mc[i])**2 )) 
   # print(err)
    return err
########################################################################
# get event counts for plotting 
def event_counts(datasets, xvar, xmin, xmax, cuts, ext_norm, mc_norm, plot_data=False, bdt_scale=None):
    
    q = (xvar+">="+str(xmin)+" and "+xvar+"<="+str(xmax))
    
    if cuts: 
        q = q + " and " + cuts
        
    counts = {
        'outfv' : round(np.nansum(datasets['outfv'].query(q)[mc_norm]), 1), 
        'numu_NC_Npi0' : round(np.nansum(datasets['infv'].query(numu_NC_Npi0+" and "+ q)[mc_norm]), 1),
        'numu_CC_Npi0' : round(np.nansum(datasets['infv'].query(numu_CC_Npi0+" and "+ q)[mc_norm]), 1), 
        'numu_NC_0pi0' : round(np.nansum(datasets['infv'].query(numu_NC_0pi0+" and "+ q)[mc_norm]), 1),
        'numu_CC_0pi0' : round(np.nansum(datasets['infv'].query(numu_CC_0pi0+" and "+ q)[mc_norm]), 1), 
        'nue_NC' : round(np.nansum(datasets['infv'].query(nue_NC+" and "+ q)[mc_norm]), 1), 
        'nue_CCother' : round(np.nansum(datasets['infv'].query(nue_CCother+" and "+ q)[mc_norm]), 1), 
        'numu_Npi0' : round(np.nansum(datasets['infv'].query(numu_Npi0+" and "+ q)[mc_norm]), 1), 
        'numu_0pi0' : round(np.nansum(datasets['infv'].query(numu_0pi0+" and "+ q)[mc_norm]), 1), 
        'nue_other' : round(np.nansum(datasets['infv'].query(nue_other+" and "+ q)[mc_norm]), 1),
        'nuebar_1eNp' : round(np.nansum(datasets['infv'].query(nuebar_1eNp+" and "+ q)[mc_norm]), 1),
        'signal' : round(np.nansum(datasets['infv'].query(signal+" and "+ q)[mc_norm]), 1), 
        'ext' : round(np.nansum(datasets['ext'].query(q)[ext_norm]), 1)
    }

    if bdt_scale: 
        for category in counts.keys(): 
            counts[category] = counts[category]/bdt_scale
    
    return counts

########################################################################
# Plot MC, normalized to beam on, overlay, OR projected 
# NEED TO ADD: GENIE UNISIMS, NON-nueCC DET SYS
def plot_mc(var, nbins, xlow, xhigh, cuts, datasets, isrun3, norm='overlay', save=False, save_label=None, log=False, x_label=None, xmax=None, y_label=None, ymax=None, bdt_scale=None, text=None, xtext=None, ytext=None, osc=None, plot_bkgd=False, sys=None, x_ticks=None, is_flugg_reweight=False, bin_norm=1.0):
    
    
    # set the POT & plots_path for plotting
    plots_path = parameters(isrun3)['plots_path']

    if (cuts==""): 
        infv = datasets['infv']
        outfv = datasets['outfv']
        ext = datasets['ext']
        
    else: 
        infv = datasets['infv'].query(cuts)
        outfv = datasets['outfv'].query(cuts)
        ext = datasets['ext'].query(cuts)
    
    ## MC weights
    categories = {'ext' : ext, 
                  'outfv' : outfv, 
                  'numu_NC_Npi0' : infv.query(numu_NC_Npi0), 
                  'numu_CC_Npi0' : infv.query(numu_CC_Npi0), 
                  'numu_NC_0pi0' : infv.query(numu_NC_0pi0), 
                  "numu_CC_0pi0" : infv.query(numu_CC_0pi0), 
                  "numu_Npi0" : infv.query(numu_Npi0), 
                  "numu_0pi0" : infv.query(numu_0pi0), 
                  'nue_NC' : infv.query(nue_NC), 
                  'nue_CCother' : infv.query(nue_CCother), 
                  'nue_other' : infv.query(nue_other), 
                  'nuebar_1eNp' : infv.query(nuebar_1eNp), 
                  'signal' : infv.query(signal),
                  }
    
    mc_norm = ''
    ext_norm = ''

    if (norm=='data'): 
        
        if is_flugg_reweight: 
            mc_norm = 'totweight_data_flugg'
        else: 
            mc_norm = 'totweight_data'
        ext_norm = 'pot_scale'
        
    else: 
        print("update!")
        
    mc_weights = {}
    if bdt_scale: 
        print("Accounting for BDT test/train split....")
        for category in categories.keys(): 
            if category=='ext': 
                mc_weights['ext'] = [ x/(bdt_scale) for x in categories[category][ext_norm]]
            else: 
                mc_weights[category] = [ x/(bdt_scale) for x in categories[category][mc_norm]]
              
    else:
        for category in categories.keys(): 
            if category=='ext': 
                mc_weights['ext'] = categories[category][ext_norm]
            else: 
                mc_weights[category] = categories[category][mc_norm]
        
    # event counts
    # this needs to be across all the bins, not just xhigh and xlow because it doesn't capture them all!
    counts = event_counts(datasets, var, nbins[0], nbins[-1], cuts, ext_norm, mc_norm, plot_data=False, bdt_scale=bdt_scale)
     
    # legend 
    leg = {
        'ext' : labels['ext'][0]+': '+str(counts['ext']),
        'outfv' : labels['outfv'][0]+': '+str(counts['outfv']), 
        'numu_NC_Npi0' : labels['numu_NC_Npi0'][0]+': '+str(counts['numu_NC_Npi0']), 
        'numu_CC_Npi0' : labels['numu_CC_Npi0'][0]+': '+str(counts['numu_CC_Npi0']), 
        'numu_NC_0pi0' : labels['numu_NC_0pi0'][0]+': '+str(counts['numu_NC_0pi0']), 
        'numu_CC_0pi0' : labels['numu_CC_0pi0'][0]+': '+str(counts['numu_CC_0pi0']), 
        'nue_NC' : labels['nue_NC'][0]+': '+str(counts['nue_NC']), 
        'nue_CCother' : labels['nue_CCother'][0]+': '+str(counts['nue_CCother']),
        "numu_Npi0" : labels['numu_Npi0'][0]+': '+str(counts['numu_Npi0']), 
        "numu_0pi0" : labels['numu_0pi0'][0]+': '+str(counts['numu_0pi0']), 
        "nue_other" : labels['nue_other'][0]+': '+str(counts['nue_other']), 
        'nuebar_1eNp' : labels['nuebar_1eNp'][0]+': '+str(counts['nuebar_1eNp']), 
        'signal' : labels['signal'][0]+': '+str(counts['signal'])
    }
        
    
    ################### oscillated event rate #########################
    
    if osc:
        
        # plot signal only 
        n_sig, b_sig, p_sig = plt.hist(infv.query(signal)[var], nbins, histtype='bar', range=[xlow, xhigh], weights=mc_weights[-2])
        plt.close()
        #print(n_sig)
        
        osc_weight = []
        
        with open(osc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0: 
                    osc_weight.append(float(row[0]))
                    #bin_centers.append(float(row[1]))
                    
                line_count += 1

        osc_counts = [ a*b for a, b in zip(n_sig,osc_weight) ]
        
    ############### Error calculation pt. 1 (pre-plotting) #######################
    
    if sys is None: 
        mc_err = mc_error(var, nbins, xlow, xhigh, [infv, outfv]) 
    
        # quick plot of ext 
        ext_counts = plt.hist(ext[var], nbins, range=[xlow, xhigh], weights=ext[ext_norm])[0]
        plt.close()
    
    ############################ PLOT ####################################### 
     
    fig = plt.figure(figsize=(8, 5))
    n, b, p = plt.hist([ext[var], outfv[var], 
                       infv.query(numu_NC_Npi0)[var],
                       infv.query(numu_CC_Npi0)[var],
                       infv.query(numu_NC_0pi0)[var],
                       infv.query(numu_CC_0pi0)[var],
                       infv.query(nue_NC)[var],
                       infv.query(nue_CCother)[var],
                       #infv.query(numu_Npi0)[var], 
                       #infv.query(numu_0pi0)[var], 
                       #infv.query(nue_other)[var], 
                       infv.query(nuebar_1eNp)[var], 
                       infv.query(signal)[var]],
            nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
            color=[labels['ext'][1], labels['outfv'][1], 
                       labels['numu_NC_Npi0'][1], 
                       labels['numu_CC_Npi0'][1], 
                       labels['numu_NC_0pi0'][1], 
                       labels['numu_CC_0pi0'][1], 
                       labels['nue_NC'][1], 
                       labels['nue_CCother'][1],
                       #labels['numu_Npi0'][1], 
                       #labels['numu_0pi0'][1], 
                       #labels['nue_other'][1], 
                       labels['nuebar_1eNp'][1], 
                       labels['signal'][1]], 
            label=[leg['ext'],
                   leg['outfv'], 
                   leg['numu_NC_Npi0'], 
                   leg['numu_CC_Npi0'], 
                   leg['numu_NC_0pi0'], 
                   leg['numu_CC_0pi0'], 
                   leg['nue_NC'], 
                   leg['nue_CCother'], 
                   #leg['numu_Npi0'], 
                   #leg['numu_0pi0'], 
                   #leg['nue_other'], 
                   leg['nuebar_1eNp'], 
                   leg['signal']
                  ],
            weights=[mc_weights['ext'], 
                     mc_weights['outfv'], 
                     mc_weights['numu_NC_Npi0'], 
                     mc_weights['numu_CC_Npi0'], 
                     mc_weights['numu_NC_0pi0'], 
                     mc_weights['numu_CC_0pi0'], 
                     mc_weights['nue_NC'], 
                     mc_weights['nue_CCother'], 
                     #mc_weights['numu_Npi0'], 
                     #mc_weights['numu_0pi0'], 
                     #mc_weights['nue_other'], 
                     mc_weights['nuebar_1eNp'], 
                     mc_weights['signal'] 
                     ])
    
    # total selected 
    print('total selected = '+str(np.nansum(n[-1])))
    
    
    ############### Error calculation pt. 2 (post-plotting) #######################
    
    if sys is not None: 
        
        err_label = 'MC+EXT Stat.\n& Sys. Uncertainty'
        tot_percent_err = sys
        tot_err = [x*y for x,y in zip(n[-1],sys)]
        
    else: 
        ext_percent_err = np.sqrt(ext_counts)/n[-1]
        mc_percent_err = mc_err/n[-1]
    
        # add in quadrature 
        sim_percent_err = np.array([x**2+y**2 for x,y in zip(mc_percent_err, ext_percent_err)])
        sim_percent_err = np.sqrt(sim_percent_err)
    
        sim_err = [x*y for x, y in zip(n[-1], sim_percent_err)]
        
        err_label = 'MC+EXT Stat.\nUncertainty'
        
        tot_err = sim_err
        tot_percent_err =  sim_percent_err
        
    
    # uncertainty band 
    low_err = [ x-y for x,y in zip(n[-1], tot_err) ]
    low_err.insert(0, low_err[0])

    high_err = [ x+y for x,y in zip(n[-1], tot_err)]
    high_err.insert(0, high_err[0])
    
    error_handle = plt.fill_between(nbins, low_err, high_err, step="pre", facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray', 
                     hatch='.....', 
                     linewidth=0.0, zorder=2, 
                     label=err_label)
    
    bincenters = 0.5*(b[1:]+b[:-1])
    plt.errorbar(bincenters, n[-1], yerr=sim_err, fmt='none', color='black', linewidth=1)
    
    # simulation outline 
    tot = list([0, n[-1][0]])+list(n[-1])+[0]
    b_step = list([b[0]])+list(b)+list([b[-1]])
    #plt.step(b_step, tot, color='saddlebrown', linewidth=2)
    plt.step(b_step, tot, color='black', linewidth=1)
      
    ##################### Add in oscillated event rate #############################
    
    if osc:    
        # add in unoscillated background 
        osc_counts = list([0, osc_counts[0]])+osc_counts+[0]
        sig_counts = list([0, n_sig[0]])+list(n_sig)+[0]
        bkgd_counts = [y-z for y, z in zip(tot,sig_counts)]
        osc_counts = [a+b for a,b in zip(osc_counts, bkgd_counts)]
        
        plt.step(b_step, osc_counts, color='darkblue', linestyle='dashed')
    
    ############################################################################## 
   
    # plot format stuff
    # flip legend order to match plot_data style
    label_order_main = [
        leg['ext'],
        leg['outfv'], 
        leg['numu_NC_Npi0'], 
        leg['numu_CC_Npi0'], 
        leg['numu_NC_0pi0'], 
        leg['numu_CC_0pi0'], 
        leg['nue_NC'], 
        leg['nue_CCother'], 
        leg['nuebar_1eNp'], 
        leg['signal']
    ]
    plt.legend(handles=p[::-1] + [error_handle], labels=label_order_main[::-1] + [err_label], loc='upper right', prop={"size":10}, ncol=2, frameon=False)

    plt.text(0.03, 0.95, "MicroBooNE Run 4b RHC", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    
    if y_label: 
        plt.ylabel(y_label, fontsize=15, labelpad=8)
    
    if x_label:
        plt.xlabel(x_label, fontsize=15, labelpad=8)
    else: 
        plt.xlabel(var, fontsize=15, labelpad=8)
    
    if x_ticks: 
        plt.xticks(x_ticks, fontsize=14)
    else: 
        plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if log: 
        plt.yscale('log')
        
    if ymax: 
        if log: 
            plt.ylim(1, ymax)
        else: 
            plt.ylim(0, ymax)
            
    if xmax: 
        plt.xlim(xlow, xmax)
    else: 
        plt.xlim(xlow, xhigh)
            
    if text: 
        plt.text(xtext, ytext, text, fontsize='xx-large', horizontalalignment='right')
    
    if save: 
        plt.savefig("BDTPlot_Run4bRHC_Comp.svg", transparent=False, bbox_inches='tight') 
        #plt.savefig(plots_path+var+"_"+save_label+".pdf", transparent=True, bbox_inches='tight') 
        print('saving to: '+plots_path)
        
    plt.show()
    
    ######################### plot background only #################################
    
    mc_bkgd_err = mc_error(var, nbins, xlow, xhigh, [outfv, infv.query(not_signal)]) 

    fig = plt.figure(figsize=(8, 5))

    n2, b2, p2 = plt.hist([outfv[var], 
                           infv.query(numu_NC_Npi0)[var],
                           infv.query(numu_CC_Npi0)[var],
                           infv.query(numu_NC_0pi0)[var],
                           infv.query(numu_CC_0pi0)[var],
                           infv.query(nue_NC)[var],
                           infv.query(nue_CCother)[var],
                           infv.query(nuebar_1eNp)[var], 
                           ext[var]],
                nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
                color=[labels['outfv'][1], 
                           labels['numu_NC_Npi0'][1], 
                           labels['numu_CC_Npi0'][1], 
                           labels['numu_NC_0pi0'][1], 
                           labels['numu_CC_0pi0'][1], 
                           labels['nue_NC'][1], 
                           labels['nue_CCother'][1],
                           labels['nuebar_1eNp'][1], 
                           labels['ext'][1]], 
                label=[leg['outfv'], 
                       leg['numu_NC_Npi0'], 
                       leg['numu_CC_Npi0'], 
                       leg['numu_NC_0pi0'], 
                       leg['numu_CC_0pi0'], 
                       leg['nue_NC'], 
                       leg['nue_CCother'], 
                       leg['nuebar_1eNp'], 
                       leg['ext']],
                weights=[mc_weights['outfv'], 
                       mc_weights['numu_NC_Npi0'], 
                       mc_weights['numu_CC_Npi0'], 
                       mc_weights['numu_NC_0pi0'], 
                       mc_weights['numu_CC_0pi0'], 
                       mc_weights['nue_NC'], 
                       mc_weights['nue_CCother'], 
                       mc_weights['nuebar_1eNp'], 
                       mc_weights['ext']
                      ])
    
    if plot_bkgd: 

        plt.legend(loc='best', prop={"size":10}, ncol=3, frameon=False)

        mc_bkgd_percent_err = mc_bkgd_err/n2[-1]

        # add in quadrature 
        sim_bkgd_percent_err = np.array([x**2+y**2 for x,y in zip(mc_bkgd_percent_err, ext_percent_err)])
        sim_bkgd_percent_err = np.sqrt(sim_bkgd_percent_err)

        sim_bkgd_err = [x*y for x, y in zip(n2[-1], sim_bkgd_percent_err)]

        tot2 = list([0, n2[-1][0]])+list(n2[-1])+[0]
        b_step2 = list([b2[0]])+list(b2)+list([b2[-1]])
        plt.step(b_step2, tot2, color='black', linewidth=.7)
        
        # ERRORS
        plt.errorbar(bincenters, n2[-1], yerr=sim_bkgd_err, fmt='none', color='black', linewidth=1)
    
        # FORMATTING STUFF
        
        #if pot is not None: 
            #plt.ylabel("$\\nu$ / "+pot+" POT", fontsize=15, labelpad=8)
    
        if x_label:
            plt.xlabel(x_label, fontsize=15, labelpad=8)
        else: 
            plt.xlabel(var, fontsize=15, labelpad=8)
    
        plt.xlim(xlow, xhigh)
    
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        if ymax: 
            if log: 
                plt.ylim(1, ymax)
            else: 
                plt.ylim(0, ymax)
    
        plt.title('Background Distribution', fontsize=15) 
        plt.show()
        
    else: 
        plt.close()
    
    
    ######################### Create dictionary ##################################
    # return python dictionary with bins, CV, & fractional uncertainties 
    d = { 
       "bins" : nbins, 
        "CV" : list(n[-1]), 
        "background_counts" : list(n2[-1])
    }

    return d

########################################################################
# Data/MC comparisons
def plot_data(var, nbins, xlow, xhigh, cuts, datasets, isrun3, bdt_scale=None, save=False, save_label=None, log=False, x_label=None, ymax=None, sys=None, text=None, xtext=None, ytext=None, ncol=None, x_ticks=None, norm='pot'): 
    
    # set the POT & plots_path for plotting
    plots_path = parameters(isrun3)['plots_path']
    
    mc_norm = 'totweight_data'
    ext_norm = 'pot_scale'

    if (cuts==""): 
        infv = datasets['infv']
        outfv = datasets['outfv']
        ext = datasets['ext']
        data = datasets['data']
        
    else: 
        infv = datasets['infv'].query(cuts)
        outfv = datasets['outfv'].query(cuts)
        ext = datasets['ext'].query(cuts)
        data = datasets['data'].query(cuts)
    
    
    ####### get beam on histogram info #######
    n_data, b_data, p_data = plt.hist(data[var], nbins, range=[xlow, xhigh])
    integral_data = np.nansum(n_data)
    plt.close()

    if var=='tksh_angle': 
        bincenters = 0.5*(np.array(nbins)[1:]+np.array(nbins)[:-1])
    else: 
        bincenters = 0.5*(np.array(nbins[:-1]+[xhigh])[1:]+np.array(nbins[:-1]+[xhigh])[:-1])

    ####### get integral for simulated event spectrum #######
    n_sim, b_sim, p_sim = plt.hist([outfv[var], infv[var], ext[var]], 
                                  nbins, range=[xlow, xhigh], stacked=True, 
                                  weights=[outfv[mc_norm], infv[mc_norm], ext[ext_norm]])
    integral_mc = np.nansum(n_sim[-1])
    plt.close()
    
    ####### weights for the MC plot #######
    mc_weights = []
    mc_weights_pot = [ext[ext_norm], 
                      outfv[mc_norm], 
                      infv.query(numu_NC_Npi0)[mc_norm], 
                      infv.query(numu_CC_Npi0)[mc_norm], 
                      infv.query(numu_NC_0pi0)[mc_norm], 
                      infv.query(numu_CC_0pi0)[mc_norm], 
                      infv.query(nue_NC)[mc_norm], 
                      infv.query(nue_CCother)[mc_norm], 
                      infv.query(nuebar_1eNp)[mc_norm], 
                      infv.query(signal)[mc_norm]]
    

    # mc_weights_pot = [
    # infv.query(signal)[mc_norm],
    # infv.query(nuebar_1eNp)[mc_norm],
    # infv.query(nue_CCother)[mc_norm],
    # infv.query(nue_NC)[mc_norm],
    # infv.query(numu_CC_0pi0)[mc_norm],
    # infv.query(numu_NC_0pi0)[mc_norm],
    # infv.query(numu_CC_Npi0)[mc_norm],
    # infv.query(numu_NC_Npi0)[mc_norm],
    # outfv[mc_norm],
    # ext[ext_norm]
    # ]


    ####### account for POT change in the test/train splitting #######
    if bdt_scale: 
        print('Accounting for test/train split....')
        mc_weights_pot = [[x/bdt_scale for x in y] for y in mc_weights_pot]

    if norm=='area':         
        area_scale = integral_data/integral_mc  
        
        for l in mc_weights_pot: 
            mc_weights.append([ k*area_scale for k in l ])
            
    else: 
        mc_weights = mc_weights_pot


    ######## event counts ########
    counts = event_counts(datasets, var, xlow, xhigh, cuts, ext_norm, mc_norm, plot_data=True, bdt_scale=bdt_scale)

    
    ######## legend ########
    leg = [labels['ext'][0]+': '+str(counts['ext']),
                        labels['outfv'][0]+': '+str(counts['outfv']), 
                        labels['numu_NC_Npi0'][0]+': '+str(counts['numu_NC_Npi0']), 
                        labels['numu_CC_Npi0'][0]+': '+str(counts['numu_CC_Npi0']), 
                        labels['numu_NC_0pi0'][0]+': '+str(counts['numu_NC_0pi0']), 
                        labels['numu_CC_0pi0'][0]+': '+str(counts['numu_CC_0pi0']), 
                        labels['nue_NC'][0]+': '+str(counts['nue_NC']), 
                        labels['nue_CCother'][0]+': '+str(counts['nue_CCother']), 
                        labels['nuebar_1eNp'][0]+': '+str(counts['nuebar_1eNp']),
                        labels['signal'][0]+': '+str(counts['signal'])
                        ]

    # leg = [
    # labels['signal'][0]+': '+str(counts['signal']),
    # labels['nuebar_1eNp'][0]+': '+str(counts['nuebar_1eNp']),
    # labels['nue_CCother'][0]+': '+str(counts['nue_CCother']),
    # labels['nue_NC'][0]+': '+str(counts['nue_NC']),
    # labels['numu_CC_0pi0'][0]+': '+str(counts['numu_CC_0pi0']),
    # labels['numu_NC_0pi0'][0]+': '+str(counts['numu_NC_0pi0']),
    # labels['numu_CC_Npi0'][0]+': '+str(counts['numu_CC_Npi0']),
    # labels['numu_NC_Npi0'][0]+': '+str(counts['numu_NC_Npi0']),
    # labels['outfv'][0]+': '+str(counts['outfv']),
    # labels['ext'][0]+': '+str(counts['ext'])
    # ]   

    ############### error calculation pt. 1 (pre-plotting) #######################
    
    print(f"sys value: {sys}, type: {type(sys)}")

    if sys is None: # then only plot the stat error 
        
        mc_err = mc_error(var, nbins, xlow, xhigh, [infv, outfv]) 
        
        # quick plot of ext 
        ext_counts = plt.hist(ext[var], nbins, range=[xlow, xhigh], weights=ext[ext_norm])[0]
        plt.close()
        

    ##############################################################################

    # plot 
    #fig = plt.figure(figsize=(12, 10))
    fig = plt.figure(figsize=(8, 7))

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
    
    ax2.yaxis.grid(linestyle="--", color='black', alpha=0.2)
    ax2.xaxis.grid(linestyle="--", color='black', alpha=0.2)
    
    if x_ticks: 
        ax1.set_xticks(x_ticks)
        ax2.set_xticks(x_ticks)


    n, b, p = ax1.hist([ext[var], 
                        outfv[var], 
                        infv.query(numu_NC_Npi0)[var],
                        infv.query(numu_CC_Npi0)[var],
                        infv.query(numu_NC_0pi0)[var],
                        infv.query(numu_CC_0pi0)[var],
                        infv.query(nue_NC)[var], 
                        infv.query(nue_CCother)[var], 
                        infv.query(nuebar_1eNp)[var], 
                        infv.query(signal)[var]], 
            nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
            color=[labels['ext'][1], 
                            labels['outfv'][1], 
                            labels['numu_NC_Npi0'][1],
                            labels['numu_CC_Npi0'][1],
                            labels['numu_NC_0pi0'][1],
                            labels['numu_CC_0pi0'][1],
                            labels['nue_NC'][1], 
                            labels['nue_CCother'][1], 
                            labels['nuebar_1eNp'][1], 
                            labels['signal'][1] 
                            ],      
            label=leg, 
            weights=mc_weights_pot, zorder=1)

    # n, b, p = ax1.hist([infv.query(signal)[var], 
                    # infv.query(nuebar_1eNp)[var], 
                    # infv.query(nue_CCother)[var], 
                    # infv.query(nue_NC)[var], 
                    # infv.query(numu_CC_0pi0)[var], 
                    # infv.query(numu_NC_0pi0)[var], 
                    # infv.query(numu_CC_Npi0)[var], 
                    # infv.query(numu_NC_Npi0)[var], 
                    # outfv[var], 
                    # ext[var]], 
        # nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
        # color=[labels['signal'][1], 
                    # labels['nuebar_1eNp'][1], 
                    # labels['nue_CCother'][1], 
                    # labels['nue_NC'][1], 
                    # labels['numu_CC_0pi0'][1], 
                    # labels['numu_NC_0pi0'][1], 
                    # labels['numu_CC_Npi0'][1], 
                    # labels['numu_NC_Npi0'][1], 
                    # labels['outfv'][1], 
                    # labels['ext'][1] 
                #    ], 
        # label=leg, 
        # weights=mc_weights, zorder=1)


    ############################ PLOT THE BEAM-ON DATA ############################
     
    # calculate the width of each bin 
    #x_err = [ (b[i+1]-b[i])/2 for i in range(len(b)-1) ]
    
    x_err = []
    for x in range(len(bincenters)):
        if var=='tksh_angle': 
            x_err.append(round(abs((nbins)[x+1]-(nbins)[x])/2, 3))
        
        else: 
            x_err.append(round(abs((nbins[:-1]+[xhigh])[x+1]-(nbins[:-1]+[xhigh])[x])/2, 3))

    data_handle = ax1.errorbar(bincenters, n_data, yerr=np.sqrt(n_data), xerr=x_err, 
             color="black", fmt='o', markersize=3, label='NuMI Data: '+str(int(sum(n_data))), zorder=4) # 4
    
    ax1.set_ylabel("Events / Bin", fontsize=15, labelpad=7)
    ax1.set_xlim(xlow, xhigh)
    
    if ymax:
        if log: 
            ax1.set_ylim(1, ymax)
        else: 
            ax1.set_ylim(0, ymax)    
    
    ############### error calculation pt. 2 (post-plotting)#######################
    
    if sys is not None: 
        
        err_label = 'MC+EXT Stat.\n& Sys. Uncertainty'
        tot_percent_err =  sys
        tot_err = [x*y for x, y in zip(n[-1], sys)]
    
    else: 
        ext_percent_err = np.sqrt(ext_counts)/n[-1]
        mc_percent_err = mc_err/n[-1]

        # add in quadrature 
        sim_percent_err = np.array([x**2+y**2 for x,y in zip(mc_percent_err, ext_percent_err)])
        sim_percent_err = np.sqrt(sim_percent_err)

        sim_err = [x*y for x, y in zip(n[-1], sim_percent_err)]
        
        err_label = 'MC+EXT Stat.\nUncertainty'
        
        tot_err = sim_err
        tot_percent_err =  sim_percent_err

    
    low_err = [ x-y for x,y in zip(n[-1], tot_err) ]
    low_err.insert(0, low_err[0])

    high_err = [ x+y for x,y in zip(n[-1], tot_err)]
    high_err.insert(0, high_err[0])

    print(f"tot_err shape: {len(tot_err)}")
    print(f"nbins shape: {len(nbins)}")
    print(f"low_err shape: {len(low_err)}")
    print(f"high_err shape: {len(high_err)}")
    
    error_handle = ax1.fill_between(nbins, low_err, high_err, step="pre",
                    facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray', #(.8627, .8627, .8627, 1),  
                     hatch='.....', 
                     linewidth=0.0, zorder=2, 
                     label=err_label)
    
    # simulation outline 
    tot = list([0, n[-1][0]])+list(n[-1])+[0]
    b_step = list([b[0]])+list(b)+list([b[-1]])
    ax1.step(b_step, tot, color='saddlebrown', linewidth=2, zorder=3, alpha=0.85)
    
            
    ############################ PLOT THE RATIO ############################
            
    # ratio plot  
    ax2.errorbar(bincenters, n_data/n[-1], yerr=get_ratio_err(n_data, n[-1]), xerr=x_err, color="black", fmt='o')
    ax2.set_xlim(xlow, xhigh)
    # ax2.set_ylim(-.3, 2.3) # Best for main BDT 
    # ax2.set_ylim(-.4, 2.4)
    ax2.set_ylim(-0.8, 2.8)
    # ax2.set_ylim(-1.0, 3.0)
    # ax2.set_ylim(-0.5, 2.5) # Best for visible energy and opening angle! 
    # ax2.set_ylim(-2.0, 4)
    # ax2.set_ylim(-1.8, 3.8) # Best for dE/dx
    # ax2.set_ylim(0.5, 1.5)
    # ax2.set_ylim(0.7, 1.3)
    # ax2.set_ylim(0.2, 1.8)
    # ax2.set_ylim(0, 2) 
    
    # horizontal line at 1 
    ax2.axhline(1.0, color='black', lw=1, linestyle='--')
    
    # MC ratio error - stat + sys 
    low_err_ratio = [ 1 - x for x in tot_percent_err ]
    low_err_ratio.insert(0, low_err_ratio[0])
    
    high_err_ratio = [ 1 + x for x in tot_percent_err ]
    high_err_ratio.insert(0, high_err_ratio[0])

    ax2.fill_between(nbins, low_err_ratio, high_err_ratio, step="pre", facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray',  
                     hatch='.....', 
                     linewidth=0.0, zorder=1)
    
    if x_label: 
        ax2.set_xlabel(x_label, fontsize=15, labelpad=7)
    else: 
        ax2.set_xlabel(var, fontsize=15, labelpad=7)
        
    ax2.set_ylabel("Data / Prediction", fontsize=15, labelpad=7)
    
    #ax2.set_yticks([0.5, 0.75, 1, 1.25, 1.5])
    #ax1.set_xticks([0, 1])
    #ax2.set_xticks([0, 1])
    
    if ncol: 
        #ax1.legend(prop={"size":10}, ncol=ncol, handles=p[::-1], labels=leg[::-1], frameon=False, loc='upper right') #, bbox_to_anchor=(.945, 0.99))
        # ax1.legend(handles=p[::-1] + [error_handle] + [data_handle], labels=leg[::-1] + [err_label] + ['NuMI Data: '+str(int(sum(n_data)))], 
            # prop={"size":10}, ncol=ncol, frameon=False, loc='upper right')
            ax1.legend(handles=p[::-1] + [error_handle] + [data_handle], 
            labels=leg[::-1] + [err_label] + ['NuMI Data: '+str(int(sum(n_data)))], 
            prop={"size": 10}, 
            ncol=ncol, 
            frameon=False,
            # loc='upper right') 
            loc='upper right', bbox_to_anchor=(.955, 0.99)) # This is for the plots where it's right-heavy

        
    else:
        ax1.legend(prop={"size":10}, handles=p[::-1], labels=leg[::-1], ncol=3, frameon=False)

    print(ax1.get_legend_handles_labels())

    if log: 
        ax1.set_yscale('log')
        
    ############################ FINAL PLOTTING DETAILS ############################
        
    ## chi2 calculation ## 
    #chi2 = 0 
    
    #for i in range(len(n[-1])): 
    #    if tot_err[i]==0 or np.isnan(((n_data[i] - n[-1][i] )**2 / tot_err[i]**2)): 
    #        continue 
    #    else: 
            
    #        chi2 = chi2 + ((n_data[i] - n[-1][i] )**2 / tot_err[i]**2)  #((i-j)*(i-j))/i stat only chi2
        #print('bin', i, 'chi2 addition', chi2 + ((n_data[i] - n[-1][i] )**2 / tot_err[i]**2))

    # KATRINAS
    #if text: 
        #ax1.text(xtext, ytext, text, #text+"\n$\\chi^{2}$/n = "+str(round(chi2, 2))+"/"+str(len(b)-1), 
                 #fontsize=13.5, horizontalalignment='left')
    if text: 
        #ax1.text(0.023, 0.75, text, fontsize=13.5, transform=ax1.transAxes, horizontalalignment='left')
        # ax1.text(0.023, 0.77, text, fontsize=13.5, transform=ax1.transAxes, horizontalalignment='left') # ORIGINAL
        ax1.text(0.020, 0.79, text, fontsize=12, transform=ax1.transAxes, horizontalalignment='left') # FOR TRACK PID
        #ax1.text(0.023, 0.4, text, fontsize=12, transform=ax1.transAxes, horizontalalignment='left') # NO DATA

    
    
    if norm=='area': 
        ax1.set_title("Area Normalized", fontsize=15)
    
    if save: 
        print('saving to: ', plots_path)
        # plt.savefig("/Users/abarnard/Desktop/analysis_plots/variables/"+var+"_"+save_label+".svg", bbox_inches='tight')#, dpi=1000) 
        #plt.savefig("/Users/abarnard/Desktop/analysis_plots/variables/Variable.svg", bbox_inches='tight')
        plt.savefig("/Users/abarnard/Downloads/"+var+"_"+save_label+".svg", bbox_inches='tight')#, dpi=1000) 

    plt.show()
    
    #display(fig)
    
    d = {
        #'percent_errors': percent_errors, 
        #'tot_err_percent' : tot_err_percent, 
        'mc_counts' : n[-1], 
        'data_counts': n_data, 
        'data_error' : np.sqrt(n_data), 
        'data_mc_ratio' : n_data/n[-1], 
        'data_mc_ratio_err' : get_ratio_err(n_data, n[-1])
    }
    
    return d
    
########################################################################
# Plot blinded variables for talks 

def blinded_plot(var, nbins, xlow, xhigh, cuts, datasets, isrun3, bdt_scale=None, save=False, save_label=None, log=False, x_label=None, ymax=None, sys=None, text=None, xtext=None, ytext=None, ncol=None, x_ticks=None, norm='pot'): 
    
    # set the POT & plots_path for plotting
    plots_path = parameters(isrun3)['plots_path']
    
    mc_norm = 'totweight_data'
    ext_norm = 'pot_scale'

    if (cuts==""): 
        infv = datasets['infv']
        outfv = datasets['outfv']
        ext = datasets['ext']
        
    else: 
        infv = datasets['infv'].query(cuts)
        outfv = datasets['outfv'].query(cuts)
        ext = datasets['ext'].query(cuts)
    
    ####### weights for the MC plot #######
    mc_weights = []
    mc_weights_pot = [
        infv.query(signal)[mc_norm],
        infv.query(nuebar_1eNp)[mc_norm],
        infv.query(nue_CCother)[mc_norm],
        infv.query(nue_NC)[mc_norm],
        infv.query(numu_CC_0pi0)[mc_norm],
        infv.query(numu_NC_0pi0)[mc_norm],
        infv.query(numu_CC_Npi0)[mc_norm],
        infv.query(numu_NC_Npi0)[mc_norm],
        outfv[mc_norm],
        ext[ext_norm]
    ]

    ####### account for POT change in the test/train splitting #######
    if bdt_scale: 
        print('Accounting for test/train split....')
        mc_weights_pot = [[x/bdt_scale for x in y] for y in mc_weights_pot]

    if norm=='area':         
        area_scale = integral_data/integral_mc  
        
        for l in mc_weights_pot: 
            mc_weights.append([ k*area_scale for k in l ])
            
    else: 
        mc_weights = mc_weights_pot

    ######## event counts ########
    counts = event_counts(datasets, var, xlow, xhigh, cuts, ext_norm, mc_norm, plot_data=False, bdt_scale=bdt_scale)

    ######## legend ########
    leg = [
        labels['signal'][0]+': '+str(counts['signal']),
        labels['nuebar_1eNp'][0]+': '+str(counts['nuebar_1eNp']),
        labels['nue_CCother'][0]+': '+str(counts['nue_CCother']),
        labels['nue_NC'][0]+': '+str(counts['nue_NC']),
        labels['numu_CC_0pi0'][0]+': '+str(counts['numu_CC_0pi0']),
        labels['numu_NC_0pi0'][0]+': '+str(counts['numu_NC_0pi0']),
        labels['numu_CC_Npi0'][0]+': '+str(counts['numu_CC_Npi0']),
        labels['numu_NC_Npi0'][0]+': '+str(counts['numu_NC_Npi0']),
        labels['outfv'][0]+': '+str(counts['outfv']),
        labels['ext'][0]+': '+str(counts['ext'])
    ]   

    ############### error calculation pt. 1 (pre-plotting) #######################
    if sys is None: # then only plot the stat error 
        mc_err = mc_error(var, nbins, xlow, xhigh, [infv, outfv]) 
        
        # quick plot of ext 
        ext_counts = plt.hist(ext[var], nbins, range=[xlow, xhigh], weights=ext[ext_norm])[0]
        plt.close()

    ##############################################################################
    # plot 
    fig = plt.figure(figsize=(8, 5))

    ax1 = plt.subplot(111)
    
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
    
    if x_ticks: 
        ax1.set_xticks(x_ticks)

    n, b, p = ax1.hist([infv.query(signal)[var], 
                        infv.query(nuebar_1eNp)[var], 
                        infv.query(nue_CCother)[var], 
                        infv.query(nue_NC)[var], 
                        infv.query(numu_CC_0pi0)[var], 
                        infv.query(numu_NC_0pi0)[var], 
                        infv.query(numu_CC_Npi0)[var], 
                        infv.query(numu_NC_Npi0)[var], 
                        outfv[var], 
                        ext[var]], 
        nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
        color=[labels['signal'][1], 
               labels['nuebar_1eNp'][1], 
               labels['nue_CCother'][1], 
               labels['nue_NC'][1], 
               labels['numu_CC_0pi0'][1], 
               labels['numu_NC_0pi0'][1], 
               labels['numu_CC_Npi0'][1], 
               labels['numu_NC_Npi0'][1], 
               labels['outfv'][1], 
               labels['ext'][1] 
              ], 
        label=leg, 
        weights=mc_weights, zorder=1)
    
    ax1.set_ylabel("Events / Bin", fontsize=15, labelpad=7)
    ax1.set_xlim(xlow, xhigh)
    
    if ymax:
        if log: 
            ax1.set_ylim(1, ymax)
        else: 
            ax1.set_ylim(0, ymax)    
    
    ############### error calculation pt. 2 (post-plotting)#######################
    if sys is not None: 
        err_label = 'MC+EXT Stat.\n& Sys. Uncertainty'
        tot_percent_err =  sys
        tot_err = [x*y for x, y in zip(n[-1], sys)]
    else: 
        ext_percent_err = np.sqrt(ext_counts)/n[-1]
        mc_percent_err = mc_err/n[-1]

        # add in quadrature 
        sim_percent_err = np.array([x**2+y**2 for x,y in zip(mc_percent_err, ext_percent_err)])
        sim_percent_err = np.sqrt(sim_percent_err)

        sim_err = [x*y for x, y in zip(n[-1], sim_percent_err)]
        
        err_label = 'MC+EXT Stat.\nUncertainty'
        
        tot_err = sim_err
        tot_percent_err =  sim_percent_err

    low_err = [ x-y for x,y in zip(n[-1], tot_err) ]
    low_err.insert(0, low_err[0])

    high_err = [ x+y for x,y in zip(n[-1], tot_err)]
    high_err.insert(0, high_err[0])

    ax1.fill_between(nbins, low_err, high_err, step="pre",
                    facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray', 
                     hatch='.....', 
                     linewidth=0.0, zorder=2, 
                     label=err_label)
    
    # simulation outline 
    tot = list([0, n[-1][0]])+list(n[-1])+[0]
    b_step = list([b[0]])+list(b)+list([b[-1]])
    ax1.step(b_step, tot, color='saddlebrown', linewidth=2, zorder=3, alpha=0.85)
    
    if x_label: 
        ax1.set_xlabel(x_label, fontsize=15, labelpad=7)
    else: 
        ax1.set_xlabel(var, fontsize=15, labelpad=7)
        
    if ncol: 
        ax1.legend(prop={"size":10}, ncol=ncol, frameon=False, loc='upper right')
    else:
        ax1.legend(prop={"size":10}, ncol=3, frameon=False)
        
    if log: 
        ax1.set_yscale('log')
        
    ############################ FINAL PLOTTING DETAILS ############################
    if text: 
        ax1.text(0.023, 0.82, text, fontsize=13.5, transform=ax1.transAxes, horizontalalignment='left')
    
    if norm=='area': 
        ax1.set_title("Area Normalized", fontsize=15)
    
    if save: 
        print('saving to: ', plots_path)
        plt.savefig("/Users/abarnard/Desktop/analysis_plots/variables/"+var+"_"+save_label+".svg", bbox_inches='tight')

    plt.show()
    
    d = {
        'mc_counts' : n[-1]
    }
    
    return d
    
########################################################################
# Return a table of the selection performance 
# normalized to data POT
def selection_performance(cuts, datasets, gen, ISRUN3):

    #################
    # cuts --> list of strings of the cuts applied
    # datasets --> list of dataframes [df_infv, df_outfv, df_cosmic, df_ext, df_data]
    # norm --> normalize to beam ON or overlay? 
    #################
    
    # no cuts on these yet, only separated into their truth categories 
    infv = datasets['infv']
    outfv = datasets['outfv']
    ext = datasets['ext']
    
    norm = 'totweight_data'
    ext_norm = 'pot_scale'
            
    df_out = pd.DataFrame(columns=['cut', '# signal after cut',  'efficiency (%)', 'rel. eff. (%)', 
                                'purity (%)', 'purity (MC only, %)'])
    
    sig_gen_norm = np.nansum(gen)
    print("total # of signal generated in FV (normalized to DATA): "+ str(sig_gen_norm))
    
    num_signal = []
    pur = []
    pur_mconly = []
    eff = []
    rel_eff = []
    cut_list = []
    
    # start with the number of signal events 
    sig_last = sig_gen_norm #round( np.nansum(infv.query(signal)[norm]), 1 )
    
    slimmed_variables = ['nslice==1', reco_in_fv_query, 'contained_fraction>0.9']
    
    q = ''
    n=0
    
    for cut in cuts: 
        
        if cut in slimmed_variables: 
            
            if q == '': 
                q = cut
                
            else: 
                q = q + ' and ' + cut

            sig_sel_norm = np.nansum(generated_signal(ISRUN3, 'nu_e', 1, 0, 20, q)[0])
            
            num_signal.append(round(sig_sel_norm, 1))
            
            eff.append(round(sig_sel_norm/sig_gen_norm * 100, 1))
            rel_eff.append(round(sig_sel_norm/sig_last * 100, 1))
            
            pur.append(np.nan)
            pur_mconly.append(np.nan)
            
            if (n==2): 
                cut_list.append("reco'd in FV")
            else: 
                cut_list.append(cut)
            
            sig_last = sig_sel_norm
            
            n = n+1
            
        else: 
        
            infv = infv.query(cut)
            outfv = outfv.query(cut)
            ext = ext.query(cut)
        
            # how many true signal gets selected?  
            sig_sel_norm = np.nansum(infv.query(signal)[norm]) 

            tot_sel_norm = np.nansum(infv[norm])+np.nansum(outfv[norm])+np.nansum(ext[ext_norm])
            tot_sel_norm_mconly = np.nansum(infv[norm])+np.nansum(outfv[norm]) # do not include EXT
        
            num_signal.append(round(sig_sel_norm, 1))
        
            eff.append(round(sig_sel_norm/sig_gen_norm * 100, 1))
            rel_eff.append(round(sig_sel_norm/sig_last * 100, 1))
        
            pur.append(round(sig_sel_norm/tot_sel_norm * 100, 1))
            pur_mconly.append(round(sig_sel_norm/tot_sel_norm_mconly * 100, 1))
        
            if (n==2): 
                cut_list.append("reco'd in FV")
            else: 
                cut_list.append(cut)
        
            sig_last = sig_sel_norm
            n = n+1
        
    df_out['cut'] = cut_list
    df_out['# signal after cut'] = num_signal
    df_out['efficiency (%)'] = eff
    df_out['rel. eff. (%)'] = rel_eff
    df_out['purity (%)'] = pur
    df_out['purity (MC only, %)'] = pur_mconly
         
    return df_out
########################################################################
# Plot the efficiency - with binomial error bars 
def plot_eff(var, nbins, xlower, xupper, cut, datasets, isrun3, save=False, x_label=None, ymax=None, text=None, xtext=None, ytext=None, x_ticks=None): 

    infv = datasets['infv']
 
    
    ############################ Generated signal ############################

    v_sig_gen = generated_signal(isrun3, var, nbins, xlower, xupper, weight='totweight_intrinsic')[0]

    print("# of generated signal in FV: "+str( np.nansum(v_sig_gen ) ) )

    
    ############################ Selected signal ############################
    
    # apply cuts
    infv_selected = infv.query(cut)
    signal_sel = infv_selected.query('is_signal==True')
    
    print("# of selected signal in FV: "+str( np.sum( signal_sel['ppfx_cv']*signal_sel['weightSplineTimesTune'] ) ) )

    v_sig_sel, b_sig_sel, p_sig_sel = plt.hist(signal_sel[var], 
                                                              nbins, 
                                                              histtype='step', range=[xlower, xupper], 
                                                              label='signal selected in FV',
                                                              weights=signal_sel['ppfx_cv']*signal_sel['weightSplineTimesTune'])
    plt.close()
    
    b_sig_sel[-1] = xupper

   ############################ Efficiency & stat error #######################
    
    #eff = [i/j for i, j in zip(v_sig_sel, v_sig_gen)]
    eff = []
    for i, j in zip(v_sig_sel, v_sig_gen): 
        e = i/j
        if np.isnan(e): 
            eff.append(0)
        else: 
            eff.append(e)
        
    eff_err = []
    for i in range(len(eff)): 
        if eff[i]==0: 
            eff_err.append(0)
        else: 
            eff_err.append(math.sqrt( (eff[i]*(1-eff[i]))/v_sig_gen[i] ))
    
    
    bincenters = 0.5*(b_sig_sel[1:]+b_sig_sel[:-1])
    binwidth = []
    
    for x in range(len(bincenters)): 
        binwidth.append(abs(b_sig_sel[x+1]-b_sig_sel[x])/2)
    
    fig = plt.figure(figsize=(8, 5))
    plt.errorbar(bincenters, eff, xerr=binwidth, yerr=eff_err, fmt='o', 
             color='seagreen', ecolor='seagreen', markersize=3) 
    
    plt.xlim(xlower, xupper)
    plt.grid(linestyle=':')
    
    if x_label: 
        plt.xlabel(x_label, fontsize=15)
    else: 
        plt.xlabel(var, fontsize=15)
    
    if ymax: 
        plt.ylim(0, ymax)
        
    if text: 
        plt.text(xtext, ytext, text, fontsize='xx-large', horizontalalignment='center')
        
    if x_ticks: 
        plt.xticks(x_ticks, fontsize=14)
        
    else: 
        plt.xticks(fontsize=14)
        
    plt.ylabel("Efficiency", fontsize=15)
    plt.tight_layout()

    plt.yticks(fontsize=14)
    
    plots_path = parameters(isrun3)['plots_path']
    
    if save: 
        plt.savefig(plots_path+"eff_"+var+".pdf", transparent=True)
        print("saving to "+plots_path)
    
    plt.show()

        
    
########################################################################
# BDT FUNCTIONS
########################################################################
# makes a copy of MC and EXT dataframes with extra columns 
# updated for modified signal
def addRelevantColumns(datasets): 
    
    mc_bdt = pd.concat([datasets['infv'], datasets['outfv']], ignore_index=True, sort=True)
    ext_bdt = datasets['ext']
    
    mc_bdt['is_mc'] = True 
    ext_bdt['is_mc'] = False
        
    mc_bdt['weight'] = mc_bdt['totweight_data']
    ext_bdt['weight'] = ext_bdt['pot_scale']
            
    df_pre = pd.concat([mc_bdt, ext_bdt], ignore_index=True, sort=True)

    
    return df_pre

########################################################################

def addRelevantColumns_flexible(datasets, USE_EXT_IN_BDT=True): 
    """
    Combine MC and optionally EXT datasets with additional columns needed for BDT analysis.
    
    Parameters:
    - datasets: dictionary containing 'infv', 'outfv', and 'ext'
    - USE_EXT_IN_BDT: boolean flag to include EXT data in BDT training
    
    Returns:
    - df_pre: combined dataframe ready for BDT training
    """
    
    mc_bdt = pd.concat([datasets['infv'], datasets['outfv']], ignore_index=True, sort=True)
    mc_bdt['is_mc'] = True 
    mc_bdt['weight'] = mc_bdt['totweight_data']
    
    if USE_EXT_IN_BDT:
        ext_bdt = datasets['ext']
        ext_bdt['is_mc'] = False
        ext_bdt['weight'] = ext_bdt['pot_scale']
        df_pre = pd.concat([mc_bdt, ext_bdt], ignore_index=True, sort=True)
    else:
        df_pre = mc_bdt
    
    return df_pre

########################################################################
def prep_sets(train, test, train_query, test_query, varlist):
    
    train_query = train.query(train_query)
    test_query = test.query(test_query)
    
    # Last column will be signal definition for training ('is_signal')
    X_train, y_train = train_query.iloc[:,:-1], train_query['is_signal']
    
    # Signal definition for testing will always be 'is_signal' or true signal definition
    X_test, y_test = test_query.iloc[:,:-1], test_query['is_signal']

    # Cleaning dataframe
    # Note that data for testing is also cleaned
    for column in varlist:
        X_train.loc[(X_train[column] < -1.0e37) | (X_train[column] > 1.0e37), column] = np.nan
        X_test.loc[(X_test[column] < -1.0e37) | (X_test[column] > 1.0e37), column] = np.nan
    
    # Training and Testing DMatrices are only comprised of training variable list
    dtrain = xgb.DMatrix(data=X_train[varlist], label=y_train)
    dtest = xgb.DMatrix(data=X_test[varlist], label=y_test)
    
    d = {
        'X_train': X_train, 
        'X_test': X_test, 
        'dtrain': dtrain, 
        'dtest' : dtest
    }
    
    return d
########################################################################
def bdt_raw_results(train, test, train_query, test_query, varlist, params, rounds):
    
    d = prep_sets(train, test, train_query, test_query, varlist)
    
    queried_train_df = d['X_train']
    queried_test_df = d['X_test']
    dtrain = d['dtrain']
    dtest = d['dtest']
    
    model = xgb.train(params, dtrain, rounds)
    preds = model.predict(dtest)
    
    queried_test_df['is_signal'] = dtest.get_label()
    queried_test_df['BDT_score'] = preds
    
    # Add weight column back from the original test dataframe
    # Get the queried test indices
    test_queried = test.query(test_query)
    if 'weight' in test_queried.columns:
        queried_test_df['weight'] = test_queried['weight'].values
    
    return queried_test_df, model
########################################################################
def main_BDT(datasets, train_query, test_query, rounds, training_parameters, isrun3, test_size=0.5, USE_EXT_IN_BDT=True):
    
    # combine MC & EXT datasets with additional columns needed for BDT analysis
    df_pre = addRelevantColumns_flexible(datasets, USE_EXT_IN_BDT=USE_EXT_IN_BDT)
    
    # compute the scale weight for model parameters 
    scale_weight = len(df_pre.query(train_query + ' and is_signal == False')) / len(df_pre.query(train_query + ' and is_signal == True'))
    print("scale pos weight (ratio of negative to positive) = "+str(scale_weight))
    
    # Split arrays or matrices into random train and test subsets
    # stratify keeps the same signal/background ratio 
    df_pre_train, df_pre_test = train_test_split(df_pre, test_size=test_size, random_state=17, stratify=df_pre['is_signal'])

    varlist = training_parameters
    
    #model params
    params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eta': 0.02,
        'tree_method': 'exact',
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 1,
        'silent': 1,
        'min_child_weight': 1,
        'seed': 2002,
        'gamma': 1,
        'max_delta_step': 0,
        'scale_pos_weight': scale_weight,
        'eval_metric': ['error', 'auc', 'aucpr']
    }
    
    # datasets get cleaned in bdt_raw_results (prep_sets)
    bdt_results_df, bdt_model  = bdt_raw_results(df_pre_train, df_pre_test, train_query, test_query, training_parameters, params, rounds)
    
    d = {
        'bdt_results_df': bdt_results_df, 
        'bdt_model': bdt_model, 
        'df_pre_train': df_pre_train, 
        'df_pre_test': df_pre_test, 
        'df_pre': df_pre
    }
    
    return d
########################################################################    
# BDT Metric evaluation
def bdt_metrics(train, test, train_query, test_query, training_parameters, isrun3, save=False, verbose=False): 
    
    scale_weight = len(train.query(train_query+' and is_signal==True')) / len(train.query(train_query+' and is_signal==False'))
    
    #model params
    params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eta': 0.02,
        'tree_method': 'exact',
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 1,
        'silent': 1,
        'min_child_weight': 1,
        'seed': 2002,
        'gamma': 1,
        'max_delta_step': 0,
        'scale_pos_weight': scale_weight,
        'eval_metric': ['error', 'auc', 'aucpr']
    }
    
    dtrain = prep_sets(train, test, train_query, test_query, training_parameters)['dtrain']
    dtest = prep_sets(train, test, train_query, test_query, training_parameters)['dtest']

    watchlist = [(dtrain, 'train'), (dtest, 'valid')]

    progress = dict()

    model = xgb.train(params, dtrain, 1000, watchlist, early_stopping_rounds=50, evals_result=progress, verbose_eval=verbose)

    # AUC
    plt.figure(figsize=(10, 5))
    
    plt.plot(progress['train']['auc'], color='orange', label='AUC (Training Sample)', markersize=3)
    plt.plot(progress['valid']['auc'], color='blue', label='AUC (Test Sample)', markersize=3)
    
    plt.grid(linestyle=":")
    plt.legend(loc='best', prop={"size":13})
    
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    
    if isrun3: 
        plt.title('Run 4b RHC BDT AUC', fontsize=15)
        plt.ylim(0.62, 0.79)
    else: 
        plt.title('FHC Run 1 BDT AUC', fontsize=15)
    
    plt.xlabel('Number of Boosting Rounds', fontsize=14)
    #plt.ylim(0.75, 0.8)
    
    if save: 
        plt.savefig("BDT_AUC.pdf", transparent=True, bbox_inches='tight') 
    plt.show()
    
    
    # AUC PR
    plt.figure(figsize=(10, 5))
    
    plt.plot(progress['train']['aucpr'], color='orange', label='AUC PR (Training Sample)', markersize=3)
    plt.plot(progress['valid']['aucpr'], color='blue', label='AUC PR (Test Sample)', markersize=3)
    
    plt.grid(linestyle=":")
    plt.legend(loc='upper left', prop={"size":13})
    
    if isrun3: 
        plt.title('Run 4b RHC BDT AUCPR', fontsize=15)
    else: 
        plt.title('FHC Run 1 BDT AUCPR', fontsize=15)
        
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.xlabel('Number of Boosting Rounds', fontsize=14)
    # plt.ylim(0.7, 0.9)
    
    if save: 
        plt.savefig("BDT_AUCPR.pdf", transparent=True, bbox_inches='tight') 
    plt.show()
    
    
    #for metric in progress['train'].keys():
    #    plt.figure(figsize=(10, 5))
    #    plt.plot(progress['train'][metric], color='orange', label='train '+metric, markersize=3)
    #    plt.plot(progress['valid'][metric], color='blue', label='test '+metric, markersize=3)  
    #    plt.grid(linestyle=":")
    #    plt.legend(loc='upper right', prop={"size":13})
    #    plt.xlabel('# of rounds')
    #    plt.show()
        
########################################################################
# BDT Purity/Efficiency 
def bdt_pe(df, xvals, gen_data, gen_intrinsic, split):
    
    ########################
    # df --> dataframe with evaluated BDT_score added 
    # xvals --> x axis array
    # full_test_df --> df used for testing (before pre/loose cuts) 
    ########################
    
    purity=[]
    purErr=[]
    eff=[]
    effErr=[]
    

    for cut_val in xvals:
    
        
        cut_val = round(cut_val, 3)
        q = BDT_LOOSE_CUTS+' and BDT_score > '+str(cut_val)
        
        # total signal selected
        tot_sel_sig = np.nansum(df.query(q+' and is_signal == True').weight)
        
        # total events selected
        tot_sel = np.nansum(df.query(q).weight)
        
        # total signal generated
        tot_sig = np.nansum(gen_data)*split # only include the amount of dataset used for TESTING
        tot_sig_intrinsic = np.nansum(gen_intrinsic)*split # for error computation, use the intrinsic event count
        
        p = tot_sel_sig / tot_sel
        purity.append(p * 100)
        purErr.append( p * np.sqrt( sum(df.query(q+' and is_signal==True').weight**2)/sum(df.query(q+' and is_signal==True').weight)**2 + sum(df.query(q).weight**2)/sum(df.query(q).weight)**2 )  *100)
        
        e = tot_sel_sig / tot_sig
        eff.append(e * 100)
        effErr.append(np.sqrt( (e * (1-e)) / tot_sig_intrinsic ) * 100)
        
        
    d = {
        'purity': purity, 
        'purErr': purErr, 
        'eff': eff, 
        'effErr': effErr
    }
    
    return d

########################################################################
def split_events(df):

    # Remember to turn back on EXT stuff if you want to do BDT with EXT included!
    
    #separate by in/out FV & cosmic 
    # ext_bdt = df.query('is_mc==False')
    outfv_bdt = df.query(out_fv_query+' and is_mc==True')
    #cosmic_bdt = df.query(in_fv_query+' and nu_purity_from_pfp<=0.5 and is_mc==True')
    infv_bdt = df.query(in_fv_query+' and is_mc==True')
    
    # checks 
    print('split_events check:', len(df) == len(outfv_bdt)+len(infv_bdt))#+len(cosmic))
    
    d = {
        'infv': infv_bdt, 
        'outfv': outfv_bdt, 
        #'cosmic': cosmic_bdt, 
        # 'ext': ext_bdt
    }
    
    return d
########################################################################
def bdt_svb_plot(df, is_log=False):
    
    plt.hist([df.query('is_signal == True')['BDT_score'], df.query('is_signal == False')['BDT_score']], 
             50, histtype='bar', range=[0, 1.0], stacked=True, 
             color=['orange','cornflowerblue'],
             label=['signal','background'],
             log=is_log)
    
    plt.legend(loc='upper right')
    plt.xlabel('BDT score')
    
    plt.show()
########################################################################  
def bdt_pe_plot(perf, xvals, isrun3, split, save=False):
    
    
    ########################
    # df --> dataframe with evaluated BDT_score added 
    # xvals --> x axis array
    ########################
    
    plots_path = parameters(isrun3)['plots_path']
    
    #plot pur/eff as function of bdt score
    plt.figure(figsize=(7, 5))
    
    pur, purErr, eff, effErr = perf['purity'], perf['purErr'], perf['eff'], perf['effErr']
    
    plt.errorbar(xvals, pur, yerr=purErr, marker='o', color='firebrick', label='Purity', markersize=3)
    plt.errorbar(xvals, eff, yerr=effErr, marker='o', color='seagreen', label='Efficiency', markersize=3)  

    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xlabel('BDT_score > #', fontsize=14)
    plt.grid(linestyle=":")
    plt.xticks(fontsize=12)
    plt.yticks(np.arange(0,105,5), fontsize=12)
    plt.legend(loc='upper left', prop={"size":13})
    plt.ylim(0, 100)
    plt.tight_layout()
    if save: 
        plt.savefig(plots_path+"BDT_performance.pdf", transparent=True, bbox_inches='tight') 

    plt.show()
    
########################################################################    
def bdt_box_plot(results_bdt, xvals, isrun3, second_results_bdt=None, results_box=None, results_box_err=None, save=False, 
                save_label=None, title=None):
    
    ###################
    # results_bdt --> BDT performance (output of bdt_pe function)
    # results_box --> [ linear sel. purity, lineear sel. efficiency ]
    # xvals --> values on x-axis
    # second_results_bdt --> BDT performance using preselection cuts ONLY (as opposed to loose cuts)
    ###################
    
    plots_path = parameters(isrun3)['plots_path']
    
    #plot pur/eff as function of bdt score
    plt.figure(figsize=(7, 5))
    
    # Loose cut BDT results
    pur, purErr, eff, effErr = results_bdt
    
    plt.errorbar(xvals, pur, yerr=purErr, marker='o', color='maroon', label='BDT Purity', markersize=3)
    plt.errorbar(xvals, eff, yerr=effErr, marker='o', color='green', label='BDT Efficiency', markersize=3)  
    

    # Box cut results 
    if results_box: 
        plt.axhline(results_box[0], color='red', 
                    linestyle='dashed', label='Lin. Sel. Purity ('+str(round(results_box[0], 1))+'%)', linewidth=2)
        plt.axhline(results_box[1], color='limegreen', 
                    linestyle='dashed', label='Lin. Sel. Eff. ('+str(round(results_box[1], 1))+'%)', linewidth=2)
    
    if results_box_err:
        
        # purity stat error - poisson  
        plt.fill_between(xvals, results_box[0]-results_box_err[0], results_box[0]+results_box_err[0], color='red', 
                        alpha=0.15)
        
        # eff stat error - binomial 
        plt.fill_between(xvals, results_box[1]-results_box_err[1], results_box[1]+results_box_err[1], color='limegreen', 
                        alpha=0.15)
    
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xlabel('BDT score > #', fontsize=14)
    plt.grid(linestyle=":")
    plt.xticks(fontsize=12)
    plt.xlim(0, xvals[-1])
    plt.yticks(np.arange(0,105,5), fontsize=12)
    plt.legend(prop={"size":11}, loc='upper left')
    plt.ylim(0, 100)
    if title: 
        plt.title(title, fontsize=15)
    #plt.tight_layout()
    if save: 
        plt.savefig("BDT_performance_"+save_label+".pdf", transparent=True, bbox_inches='tight') 

    
    plt.show()  

########################################################################
def plot_mc_no_ext(var, nbins, xlow, xhigh, cuts, datasets, isrun3, norm='overlay', save=False, save_label=None, log=False, x_label=None, xmax=None, y_label=None, ymax=None, bdt_scale=None, text=None, xtext=None, ytext=None, osc=None, plot_bkgd=False, sys=None, x_ticks=None, is_flugg_reweight=False, bin_norm=1.0):
    """
    Modified version of plot_mc that excludes EXT (beam-off data) from the plots.
    This function plots only Monte Carlo samples (infv and outfv) without any beam-off background.
    
    Parameters are the same as plot_mc, but EXT data is completely excluded from the analysis.
    """
    
    # set the POT & plots_path for plotting
    plots_path = parameters(isrun3)['plots_path']

    if (cuts==""): 
        infv = datasets['infv']
        outfv = datasets['outfv']
        
    else: 
        infv = datasets['infv'].query(cuts)
        outfv = datasets['outfv'].query(cuts)
    
    ## MC weights - NO EXT
    categories = {'outfv' : outfv, 
                  'numu_NC_Npi0' : infv.query(numu_NC_Npi0), 
                  'numu_CC_Npi0' : infv.query(numu_CC_Npi0), 
                  'numu_NC_0pi0' : infv.query(numu_NC_0pi0), 
                  "numu_CC_0pi0" : infv.query(numu_CC_0pi0), 
                  "numu_Npi0" : infv.query(numu_Npi0), 
                  "numu_0pi0" : infv.query(numu_0pi0), 
                  'nue_NC' : infv.query(nue_NC), 
                  'nue_CCother' : infv.query(nue_CCother), 
                  'nue_other' : infv.query(nue_other), 
                  'nuebar_1eNp' : infv.query(nuebar_1eNp), 
                  'signal' : infv.query(signal),
                  }
    
    mc_norm = ''

    if (norm=='data'): 
        
        if is_flugg_reweight: 
            mc_norm = 'totweight_data_flugg'
        else: 
            mc_norm = 'totweight_data'
        
    else: 
        print("update!")
        
    mc_weights = {}
    if bdt_scale: 
        print("Accounting for BDT test/train split....")
        for category in categories.keys(): 
            mc_weights[category] = [ x/(bdt_scale) for x in categories[category][mc_norm]]
              
    else:
        for category in categories.keys(): 
            mc_weights[category] = categories[category][mc_norm]
        
    # event counts - NO EXT
    counts = {}
    for category in categories.keys():
        if len(categories[category]) > 0:
            counts[category] = np.nansum(mc_weights[category])
        else:
            counts[category] = 0.0
     
     # legend - NO EXT (signal displayed to one decimal place)
    leg = {
        'outfv' : labels['outfv'][0]+': '+format(counts['outfv'], '.1f'),
        'numu_NC_Npi0' : labels['numu_NC_Npi0'][0]+': '+format(counts['numu_NC_Npi0'], '.1f'),
        'numu_CC_Npi0' : labels['numu_CC_Npi0'][0]+': '+format(counts['numu_CC_Npi0'], '.1f'),
        'numu_NC_0pi0' : labels['numu_NC_0pi0'][0]+': '+format(counts['numu_NC_0pi0'], '.1f'),
        'numu_CC_0pi0' : labels['numu_CC_0pi0'][0]+': '+format(counts['numu_CC_0pi0'], '.1f'),
        'nue_NC' : labels['nue_NC'][0]+': '+format(counts['nue_NC'], '.1f'),
        'nue_CCother' : labels['nue_CCother'][0]+': '+format(counts['nue_CCother'], '.1f'),
        "numu_Npi0" : labels['numu_Npi0'][0]+': '+format(counts['numu_Npi0'], '.1f'),
        "numu_0pi0" : labels['numu_0pi0'][0]+': '+format(counts['numu_0pi0'], '.1f'),
        "nue_other" : labels['nue_other'][0]+': '+format(counts['nue_other'], '.1f'),
        'nuebar_1eNp' : labels['nuebar_1eNp'][0]+': '+format(counts['nuebar_1eNp'], '.1f'),
        # format signal to one decimal place
        'signal' : labels['signal'][0]+': '+format(counts['signal'], '.1f')
    }
        
    
    ################### oscillated event rate #########################
    
    if osc:
        
        # plot signal only 
        n_sig, b_sig, p_sig = plt.hist(infv.query(signal)[var], nbins, histtype='bar', range=[xlow, xhigh], weights=mc_weights['signal'])
        plt.close()
        
        osc_weight = []
        
        with open(osc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0: 
                    osc_weight.append(float(row[0]))
                    
                line_count += 1

        osc_counts = [ a*b for a, b in zip(n_sig,osc_weight) ]
        
    ############### Error calculation pt. 1 (pre-plotting) #######################
    
    if sys is None: 
        mc_err = mc_error(var, nbins, xlow, xhigh, [infv, outfv]) 
    
    ############################ PLOT - NO EXT ####################################### 
     
    fig = plt.figure(figsize=(8, 5))
    n, b, p = plt.hist([outfv[var], 
                       infv.query(numu_NC_Npi0)[var],
                       infv.query(numu_CC_Npi0)[var],
                       infv.query(numu_NC_0pi0)[var],
                       infv.query(numu_CC_0pi0)[var],
                       infv.query(nue_NC)[var],
                       infv.query(nue_CCother)[var],
                       infv.query(nuebar_1eNp)[var], 
                       infv.query(signal)[var]],
            nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
            color=[labels['outfv'][1], 
                       labels['numu_NC_Npi0'][1], 
                       labels['numu_CC_Npi0'][1], 
                       labels['numu_NC_0pi0'][1], 
                       labels['numu_CC_0pi0'][1], 
                       labels['nue_NC'][1], 
                       labels['nue_CCother'][1],
                       labels['nuebar_1eNp'][1], 
                       labels['signal'][1]], 
            label=[leg['outfv'], 
                   leg['numu_NC_Npi0'], 
                   leg['numu_CC_Npi0'], 
                   leg['numu_NC_0pi0'], 
                   leg['numu_CC_0pi0'], 
                   leg['nue_NC'], 
                   leg['nue_CCother'], 
                   leg['nuebar_1eNp'], 
                   leg['signal']
                  ],
            weights=[mc_weights['outfv'], 
                     mc_weights['numu_NC_Npi0'], 
                     mc_weights['numu_CC_Npi0'], 
                     mc_weights['numu_NC_0pi0'], 
                     mc_weights['numu_CC_0pi0'], 
                     mc_weights['nue_NC'], 
                     mc_weights['nue_CCother'], 
                     mc_weights['nuebar_1eNp'], 
                     mc_weights['signal'] 
                     ])
    
    # total selected (MC only - no EXT)
    print('total selected (MC only, no EXT) = '+str(np.nansum(n[-1])))
    
    
    ############### Error calculation pt. 2 (post-plotting) #######################
    
    if sys is not None: 
        
        err_label = 'MC Stat.\n& Sys. Uncertainty'
        tot_percent_err = sys
        tot_err = [x*y for x,y in zip(n[-1],sys)]
        
    else: 
        mc_percent_err = mc_err/n[-1]
    
        sim_err = [x*y for x, y in zip(n[-1], mc_percent_err)]
        
        err_label = 'MC Stat.\nUncertainty'
        
        tot_err = sim_err
        tot_percent_err = mc_percent_err
        
    
    # uncertainty band 
    low_err = [ x-y for x,y in zip(n[-1], tot_err) ]
    low_err.insert(0, low_err[0])

    high_err = [ x+y for x,y in zip(n[-1], tot_err)]
    high_err.insert(0, high_err[0])
    
    plt.fill_between(nbins, low_err, high_err, step="pre", facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray', 
                     hatch='.....', 
                     linewidth=0.0, zorder=2, 
                     label=err_label)
    
    bincenters = 0.5*(b[1:]+b[:-1])
    plt.errorbar(bincenters, n[-1], yerr=sim_err, fmt='none', color='black', linewidth=1)

    # simulation outline 
    tot = list([0, n[-1][0]])+list(n[-1])+[0]
    b_step = list([b[0]])+list(b)+list([b[-1]])
    plt.step(b_step, tot, color='black', linewidth=1)
      
    ##################### Add in oscillated event rate #############################
    
    if osc:    
        # add in unoscillated background 
        osc_counts = list([0, osc_counts[0]])+osc_counts+[0]
        sig_counts = list([0, n_sig[0]])+list(n_sig)+[0]
        bkgd_counts = [y-z for y, z in zip(tot,sig_counts)]
        osc_counts = [a+b for a,b in zip(osc_counts, bkgd_counts)]
        
        plt.step(b_step, osc_counts, color='darkblue', linestyle='dashed')
    
    ############################################################################## 
   
    # plot format stuff
    # flip legend order to match plot_data style
    label_order_main = [
        leg['outfv'], 
        leg['numu_NC_Npi0'], 
        leg['numu_CC_Npi0'], 
        leg['numu_NC_0pi0'], 
        leg['numu_CC_0pi0'], 
        leg['nue_NC'], 
        leg['nue_CCother'], 
        leg['nuebar_1eNp'], 
        leg['signal']
    ]
    plt.legend(handles=p[::-1], labels=label_order_main[::-1], loc='upper right', prop={"size":10}, ncol=2, frameon=False)
    
    # Add top-left label
    plt.text(0.03, 0.95, "MicroBooNE Run 4b RHC", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        
    if y_label: 
        plt.ylabel(y_label, fontsize=15, labelpad=8)
    
    if x_label:
        plt.xlabel(x_label, fontsize=15, labelpad=8)
    else: 
        plt.xlabel(var, fontsize=15, labelpad=8)
    
    if x_ticks: 
        plt.xticks(x_ticks, fontsize=14)
    else: 
        plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if log: 
        plt.yscale('log')
        
    if ymax: 
        if log: 
            plt.ylim(1, ymax)
        else: 
            plt.ylim(0, ymax)
            
    if xmax: 
        plt.xlim(xlow, xmax)
    else: 
        plt.xlim(xlow, xhigh)
            
    if text: 
        plt.text(xtext, ytext, text, fontsize='xx-large', horizontalalignment='right')
    
    if save: 
        plt.savefig(plots_path+var+"_"+save_label+"_no_ext.svg", transparent=False, bbox_inches='tight') 
        print('saving to: '+plots_path)
        
    plt.show()
    
    ######################### plot background only (no EXT) #################################
    
    mc_bkgd_err = mc_error(var, nbins, xlow, xhigh, [outfv, infv.query(not_signal)]) 

    fig = plt.figure(figsize=(8, 5))

    n2, b2, p2 = plt.hist([outfv[var], 
                           infv.query(numu_NC_Npi0)[var],
                           infv.query(numu_CC_Npi0)[var],
                           infv.query(numu_NC_0pi0)[var],
                           infv.query(numu_CC_0pi0)[var],
                           infv.query(nue_NC)[var],
                           infv.query(nue_CCother)[var],
                           infv.query(nuebar_1eNp)[var]],
                nbins, histtype='bar', range=[xlow, xhigh], stacked=True, 
                color=[labels['outfv'][1], 
                           labels['numu_NC_Npi0'][1], 
                           labels['numu_CC_Npi0'][1], 
                           labels['numu_NC_0pi0'][1], 
                           labels['numu_CC_0pi0'][1], 
                           labels['nue_NC'][1], 
                           labels['nue_CCother'][1],
                           labels['nuebar_1eNp'][1]], 
                label=[leg['outfv'], 
                       leg['numu_NC_Npi0'], 
                       leg['numu_CC_Npi0'], 
                       leg['numu_NC_0pi0'], 
                       leg['numu_CC_0pi0'], 
                       leg['nue_NC'], 
                       leg['nue_CCother'], 
                       leg['nuebar_1eNp']], 
                weights=[mc_weights['outfv'], 
                         mc_weights['numu_NC_Npi0'], 
                         mc_weights['numu_CC_Npi0'], 
                         mc_weights['numu_NC_0pi0'], 
                         mc_weights['numu_CC_0pi0'], 
                         mc_weights['nue_NC'], 
                         mc_weights['nue_CCother'], 
                         mc_weights['nuebar_1eNp']])
    
    ############### Error calculation (background only) #######################
    
    mc_percent_err_bkgd = mc_bkgd_err/n2[-1]
    
    sim_err_bkgd = [x*y for x, y in zip(n2[-1], mc_percent_err_bkgd)]
        
    # uncertainty band 
    low_err_bkgd = [ x-y for x,y in zip(n2[-1], sim_err_bkgd) ]
    low_err_bkgd.insert(0, low_err_bkgd[0])

    high_err_bkgd = [ x+y for x,y in zip(n2[-1], sim_err_bkgd)]
    high_err_bkgd.insert(0, high_err_bkgd[0])
    
    plt.fill_between(nbins, low_err_bkgd, high_err_bkgd, step="pre", facecolor=(.25, .25, .25, 0), 
                     edgecolor='darkgray', 
                     hatch='.....', 
                     linewidth=0.0, zorder=2, 
                     label='MC Stat.\nUncertainty')
    
    bincenters2 = 0.5*(b2[1:]+b2[:-1])
    plt.errorbar(bincenters2, n2[-1], yerr=sim_err_bkgd, fmt='none', color='black', linewidth=1)

    # simulation outline 
    tot2 = list([0, n2[-1][0]])+list(n2[-1])+[0]
    b2_step = list([b2[0]])+list(b2)+list([b2[-1]])
    plt.step(b2_step, tot2, color='black', linewidth=1)
    
    # plot format stuff
    # flip legend order here as well
    label_order_bkgd = [
        leg['outfv'], 
        leg['numu_NC_Npi0'], 
        leg['numu_CC_Npi0'], 
        leg['numu_NC_0pi0'], 
        leg['numu_CC_0pi0'], 
        leg['nue_NC'], 
        leg['nue_CCother'], 
        leg['nuebar_1eNp']
    ]
    plt.legend(handles=p2[::-1], labels=label_order_bkgd[::-1], loc='upper right', prop={"size":10}, ncol=2, frameon=False)
    
    # Add top-left label to background-only plot
    plt.text(0.03, 0.95, "MicroBooNE Run 4b RHC", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        
    if y_label: 
        plt.ylabel(y_label, fontsize=15, labelpad=8)
    
    if x_label:
        plt.xlabel(x_label, fontsize=15, labelpad=8)
    else: 
        plt.xlabel(var, fontsize=15, labelpad=8)
    
    if x_ticks: 
        plt.xticks(x_ticks, fontsize=14)
    else: 
        plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if log: 
        plt.yscale('log')
        
    if ymax: 
        if log: 
            plt.ylim(1, ymax)
        else: 
            plt.ylim(0, ymax)
            
    if xmax: 
        plt.xlim(xlow, xmax)
    else: 
        plt.xlim(xlow, xhigh)
            
    if text: 
        plt.text(xtext, ytext, text, fontsize='xx-large', horizontalalignment='right')
    
    if save: 
        plt.savefig(plots_path+var+"_"+save_label+"_bkgd_only_no_ext.svg", transparent=False, bbox_inches='tight') 
        print('saving to: '+plots_path)
        
    if plot_bkgd:
        plt.show()
    else:
        plt.close()
    
    ######################### Return data dictionary #################################
    
    return {
        'bins': b,
        'CV': [np.nansum(n[-1])],
        'background_counts': [np.nansum(n2[-1])]
    }

########################################################################
######################### Training Data Functions ######################
########################################################################

def load_training_data(filename='BDT_training_data/training_data_run4brhc_scaled_noext.pkl'):
    """
    Load training data that was saved during BDT training.
    Returns all the components needed for SHAP analysis.
    """
    import pickle
    import os
    
    if not os.path.exists(filename):
        print(f"❌ Training data file not found: {filename}")
        print("   Run BDT training with save_bdt=True to create this file")
        return None
    
    try:
        with open(filename, 'rb') as f:
            data_package = pickle.load(f)
        
        print("✅ Training data loaded successfully!")
        print(f"   - Saved on: {data_package.get('timestamp', 'Unknown')}")
        print(f"   - Training events: {len(data_package['train_df']):,}")
        print(f"   - Test events: {len(data_package['test_df']):,}")
        print(f"   - Features: {len(data_package['training_parameters'])}")
        print(f"   - Split ratio: {data_package.get('split_ratio', 'Unknown')}")
        print(f"   - Model path: {data_package.get('model_path', 'Unknown')}")
        
        return data_package
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

########################################################################   

def load_training_data_for_shap(filename='BDT_training_data/training_data_run4brhc_scaled_noext.pkl'):
    """
    Load training data and format it for SHAP analysis.
    
    Parameters:
    - filename: Path to the training data pickle file
    
    Returns:
    - X_train, y_train, X_test, y_test, feature_names: Formatted training data
    - data_package: Original data package for additional metadata
    """
    import os
    
    # First check what training data files are available
    training_dir = 'BDT_training_data'
    if os.path.exists(training_dir):
        available_files = [f for f in os.listdir(training_dir) if f.endswith('.pkl')]
        if available_files:
            print(f"📁 Available training data files:")
            for i, f in enumerate(available_files, 1):
                file_path = os.path.join(training_dir, f)
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                mod_time = pd.Timestamp.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')
                print(f"   {i}. {f} ({file_size:.1f} MB, modified {mod_time})")
        else:
            print("📁 No training data files found in BDT_training_data/")
    else:
        print("📁 BDT_training_data directory does not exist")
    
    # Load the specified file
    data_package = load_training_data(filename)
    if data_package is None:
        return None, None, None, None, None, None
    
    try:
        # Extract and format components for SHAP
        train_df = data_package['train_df']
        test_df = data_package['test_df']
        training_parameters = data_package['training_parameters']
        
        # Prepare data in SHAP format
        X_train = train_df[training_parameters]
        y_train = train_df['is_signal']
        X_test = test_df[training_parameters]
        y_test = test_df['is_signal']
        feature_names = training_parameters
        
        print(f"✅ Data formatted for SHAP analysis:")
        print(f"   - X_train shape: {X_train.shape}")
        print(f"   - X_test shape: {X_test.shape}")
        print(f"   - Features: {len(feature_names)}")
        
        return X_train, y_train, X_test, y_test, feature_names, data_package
        
    except Exception as e:
        print(f"❌ Error formatting data for SHAP: {e}")
        return None, None, None, None, None, None

########################################################################   

def reconstruct_training_data(datasets, train_query, test_query, split_ratio=0.35, random_state=17, USE_EXT_IN_BDT=False):
    """
    Recreate the training data using the same parameters as the original BDT training.
    This is useful when you have a trained model but no saved training data.
    
    Parameters:
    - datasets: your original datasets dictionary
    - train_query, test_query: the queries used for training
    - split_ratio: test size used in train_test_split
    - random_state: same random state used in original training
    - use_ext: whether EXT was included in training
    
    Returns:
    - train_df, test_df: reconstructed training and test sets
    """
    try:
        # Import required function
        from sklearn.model_selection import train_test_split
        
        # Recreate the pre-training dataset using the same logic
        df_pre = addRelevantColumns_flexible(datasets, USE_EXT_IN_BDT=USE_EXT_IN_BDT)
        
        print(f"📊 Reconstructing training data...")
        print(f"   - Total events: {len(df_pre):,}")
        print(f"   - Using split ratio: {split_ratio}")
        print(f"   - Random state: {random_state}")
        print(f"   - Include EXT: {USE_EXT_IN_BDT}")
        
        # Recreate the train/test split with the same parameters
        df_pre_train, df_pre_test = train_test_split(
            df_pre, 
            test_size=split_ratio, 
            random_state=random_state, 
            stratify=df_pre['is_signal']
        )
        
        print(f"✅ Training data reconstructed!")
        print(f"   - Training events: {len(df_pre_train):,}")
        print(f"   - Test events: {len(df_pre_test):,}")
        print(f"   - Signal in training: {len(df_pre_train[df_pre_train['is_signal']==True]):,}")
        print(f"   - Background in training: {len(df_pre_train[df_pre_train['is_signal']==False]):,}")
        
        return df_pre_train, df_pre_test
        
    except Exception as e:
        print(f"❌ Error reconstructing training data: {e}")
        return None, None

########################################################################

def draw_root_like(x, bins='auto', weights=None, title=None, xlabel=None, ylabel=None,
                   density=False, logy=False, show_stats=True, range=None, ax=None,
                   histtype='step', color='black', nbins=50, max_value=None):
    """
    Simple ROOT-like Draw for a 1D array-like `x`.
    
    Parameters:
    - x: array-like (numpy, pandas Series, awkward flattened array)
    - bins: 'auto', int, or array of bin edges
    - weights: same length as x or None
    - density: if True, plot pdf
    - logy: if True, use log scale on y
    - show_stats: draws basic statistics box (entries, mean, rms)
    - range: tuple (xmin,xmax)
    - ax: matplotlib Axes
    - histtype: 'step' by default  
    - color: line color
    - nbins: number of bins if bins='auto'
    - max_value: maximum y-axis value
    
    Returns:
    - ax: matplotlib Axes object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,5))
    else:
        fig = ax.figure

    # Convert awkward arrays to numpy if needed
    try:
        import awkward as ak
        if ak.is_awkward(x):
            x = ak.to_numpy(ak.flatten(x))
    except Exception:
        pass

    x = np.asarray(x)
    if range is not None:
        # restrict values to the requested range for auto binning / stats
        mask = np.ones_like(x, dtype=bool)
        mask &= np.isfinite(x)
        if range[0] is not None:
            mask &= (x >= range[0])
        if range[1] is not None:
            mask &= (x <= range[1])
        x = x[mask]
        if weights is not None:
            weights = np.asarray(weights)[mask]

    # choose bins
    if bins == 'auto':
        # Freedman–Diaconis rule as a decent automatic choice
        q75, q25 = np.nanpercentile(x, [75, 25]) if len(x)>0 else (0,0)
        iqr = q75 - q25
        if iqr == 0:
            bin_width = (np.nanmax(x) - np.nanmin(x)) / nbins if np.nanmax(x) != np.nanmin(x) else 1.0
        else:
            bin_width = 2 * iqr * (len(x) ** (-1/3))
        if bin_width == 0 or np.isnan(bin_width) or np.isinf(bin_width):
            nb = nbins
        else:
            nb = max(1, int(np.ceil((np.nanmax(x) - np.nanmin(x)) / bin_width)))
        if nb <= 0:
            nb = nbins
        bins = nb
            
    # plot
    counts, edges, patches = ax.hist(x, bins=bins, weights=weights, histtype=histtype,
                                     density=density, range=range, color=color)
    ax.set_xlabel(xlabel if xlabel else '')
    ax.set_ylabel(ylabel if ylabel else ('Counts (normalized)' if density else 'Counts'))
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    if max_value is not None:
        ax.set_ylim(0, max_value)

    if show_stats:
        # compute basic stats with/without weights
        if weights is None:
            n = np.sum(np.isfinite(x))
            mean = np.nanmean(x) if n>0 else np.nan
            rms = np.nanstd(x) if n>0 else np.nan
        else:
            w = np.asarray(weights)
            # align lengths (already masked above)
            mask = np.isfinite(x) & np.isfinite(w)
            if np.sum(mask) == 0:
                n = 0
                mean = np.nan
                rms = np.nan
            else:
                xw = x[mask]
                ww = w[mask]
                n = np.sum(ww)
                mean = np.sum(xw * ww) / np.sum(ww)
                rms = np.sqrt(np.sum(ww * (xw - mean)**2) / np.sum(ww))
        statbox = f"Entries = {int(np.nansum(np.ones_like(x)))}\nMean = {mean:.3g}\nRMS = {rms:.3g}"
        # put a small box on the top-right of the plot
        ax.text(0.98, 0.95, statbox, transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8), fontsize=9)

    plt.tight_layout()
    return ax

########################################################################
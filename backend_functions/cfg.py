# Required for the arrays 
import numpy as np

# Creating a function 'get_variable_info' which contains the info for each variable 
def get_variable_info(name_of_variable, ISRUN3, cut_level):
  # OPENING ANGLE
  if name_of_variable == "Opening Angle":
    xvar = "tksh_angle"
    true_var = "opening_angle"
    bins = [-1.000, -0.500, -0.174, 0.174, 0.500, 1.000]
    xlow = bins[0]
    xhigh = bins[-1] 
    x_label = r'$\cos{\theta_{\mathrm{ep}}}$'
    x_ticks = [-1.000, -0.500, -0.174, 0.174, 0.500, 1.000] 

  # PROTON MULTIPLICITY
  elif name_of_variable == "Proton Multiplicity":
    xvar = "reco_Nproton"
    true_var = "nproton" # n_protons
    bins = [1, 2, 3, 12]
    xlow = bins[0]
    xhigh = 5 # Was originally bins[-1] 
    x_label = name_of_variable
    x_ticks = [1, 2, 3] 

  # ELECTRON ENERGY
  elif name_of_variable == "Electron Energy":
    xvar = "shr_energy_cali"
    true_var = "elec_e"
    bins = [0.02, 0.22, 0.42, 0.62, 0.82, 1.22, 7.00]
    xlow = bins[0]
    xhigh = 2.5 # Previously 3, wants to be changed to 2.5! 
    x_label = "Electron Energy [GeV]"
    x_ticks = [0.02, 0.22, 0.42, 0.62, 0.82, 1.22]

  # ELECTRON ENERGY LITE
  elif name_of_variable == "Electron Energy Lite":
    xvar = "shr_energy_cali"
    true_var = "elec_e"
    bins = [0.02, 0.42, 0.62, 0.82, 1.22, 7.00]
    xlow = bins[0]
    xhigh = 2.5 # Previously 3, wants to be changed to 2.5! 
    x_label = "Electron Energy [GeV]"
    x_ticks = [0.02, 0.42, 0.62, 0.82, 1.22]

    if cut_level == "signal definition":
      xvar = "shr_energy_cali"
      true_var = "elec_e"
      bins = [0.02, 0.42, 0.62, 0.82, 1.22, 7.00]
      xlow = bins[0]
      xhigh = 2.5 # Previously 3, wants to be changed to 2.5! 
      x_label = "Electron Energy [GeV]"
      x_ticks = [0.02, 0.42, 0.62, 0.82, 1.22]

    else:
      raise ValueError(f"Unsupported cut_level for {name_of_variable}: {cut_level}") 

  # ELECTRON ENERGY TEST
  elif name_of_variable == "Electron Energy Test":
    xvar = "shr_energy_cali"
    true_var = "elec_e"
    bins = [0.015, 0.02, 0.025, 0.03, 0.04]
    xlow = bins[0]
    xhigh = bins[-1] # Previously 3, wants to be changed to 2.5! 
    x_label = "Electron Energy [GeV]"
    x_ticks = bins

  # VISIBLE ENERGY
  elif name_of_variable == "Visible Energy":
    xvar = "NeutrinoEnergy2_GeV"
    true_var = "true_e_visible2"
    bins = [0.05, 0.45, 0.65, 0.85, 1.05, 1.35, 1.85, 7.00]
    xlow = bins[0]
    xhigh = 3 # Previously 4, wants to be changed to 3! 
    x_label = "Visible Energy [GeV]"
    x_ticks = [0.05, 0.45, 0.65, 0.85, 1.05, 1.35, 1.85]

  # TOTAL CROSS SECTION
  elif name_of_variable == "Total Cross Section":
    xvar = "tksh_angle"
    true_var = "opening_angle"
    bins = [-1.00, 1.00]
    xlow = -1.00
    xhigh = 1.00
    x_label = "Total Cross Section"
    x_ticks = [-1.00, 1.00]

##########################################################################################################################################################

  # 2D DISTANCE BETWEEN TRACK AND SHOWER
  elif name_of_variable == "2D Distance":
    xvar = "trkshrhitdist2"
    true_var = "trkshrhitdist2"
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    xlow = bins[0]
    xhigh = bins[-1]
    x_label = "2D Distance Between Track & Shower [cm]"
    x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  
  # NUMBER OF SHOWER SUBCLUSTERS
  elif name_of_variable == "Subclusters":
    xvar = "subcluster"
    true_var = "subcluster"
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    xlow = bins[0]
    xhigh = bins[-1]
    x_label = "Number of Shower Subclusters"
    x_ticks = [0, 10, 20, 30, 40, 50, 60]

  # TRACK PID SCORE
  elif name_of_variable == "Track PID Score":
    xvar = "trkpid"
    true_var = "trkpid"

    if cut_level == "pre-selection":
      bins = [-1.00, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
      xlow = bins[0]
      xhigh = bins[-1]
      x_label = name_of_variable
      x_ticks = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]

    elif cut_level == "bdt train": 
      bins = [-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35]
      xlow = -0.95
      xhigh = 0.35
      x_label = name_of_variable
      x_ticks = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2]

    else:
      raise ValueError(f"Unsupported cut_level for {name_of_variable}: {cut_level}")  

  elif name_of_variable == "Pandora Score":
    xvar = "shr_score"
    true_var = "shr_score"

    if cut_level == "pre-selection":  
      bins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
      xlow = bins[0]
      xhigh = bins[-1]
      x_label = name_of_variable
      x_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    elif cut_level == "bdt train": 
      bins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
      xlow = bins[0]
      xhigh = bins[-1]
      x_label = name_of_variable
      x_ticks = bins

    else: 
      raise ValueError(f"Unsupported cut_level for {name_of_variable}: {cut_level}")

  # 3D DISTANCE BETWEEN TRACK AND SHOWER
  elif name_of_variable == "3D Distance":
    xvar = "tksh_distance"
    true_var = "tksh_distance"

    if cut_level == "numu":
      bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
      xlow = bins[0]
      xhigh = bins[-1]
      x_label = "3D Distance Between Track & Shower [cm]"
      x_ticks = [0, 5, 10, 15, 20, 25]
    
    elif cut_level == "bdt train":
      bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      xlow = bins[0]
      xhigh = bins[-1]
      x_label = "3D Distance Between Track & Shower [cm]"
      x_ticks = [0, 2, 4, 6, 8, 10, 12]  

    else: 
      raise ValueError(f"Unsupported cut_level for {name_of_variable}: {cut_level}")

  # AVERAGE SHOWER MOLIÈRE ANGLE
  elif name_of_variable == "Moliere Angle":
    xvar = "shrmoliereavg"
    true_var = "shrmoliereavg"

    if cut_level == "numu":
      bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
      xlow = bins[0]
      xhigh = bins[-1]
      x_label = "Average Shower Molière Angle [deg]"
      x_ticks = [0, 5, 10, 15, 20, 25, 30, 25, 40]

    elif cut_level == "bdt train":
      bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      xlow = bins[0]
      xhigh = 15
      x_label = "Average Shower Molière Angle [deg]"
      x_ticks = [0, 2, 4, 6, 8, 10, 12, 14]

    else:
      raise ValueError(f"Unsupported cut_level for {name_of_variable}: {cut_level}")

  # DE/DX ON THE COLLECTION PLANE
  elif name_of_variable == "dE/dx":
    xvar = "shr_tkfit_dedx_Y"
    true_var = "shr_tkfit_dedx_Y"

    if cut_level == "numu":
      bins = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
      xlow = bins[0]
      xhigh = bins[-1]
      x_label = r"$\mathrm{d}\mathit{E}/\mathrm{d}\mathit{x}$ on the Collection Plane [MeV/cm]"
      x_ticks = [0, 2, 4, 6, 8, 10, 12]

    elif cut_level == "bdt train":
      bins = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
      xlow = bins[0]
      xhigh = bins[-1]
      x_label = r"$\mathrm{d}\mathit{E}/\mathrm{d}\mathit{x}$ on the Collection Plane [MeV/cm]"
      x_ticks = [0, 1, 2, 3, 4, 5, 6, 7]

    else: 
      raise ValueError(f"Unsupported cut_level for {name_of_variable}: {cut_level}") 

  # BDT SCORE
  elif name_of_variable == "BDT Score":
    xvar = "BDT_score"
    true_var = "BDT_score"
    
    # Check if ISRUN3 is True
    if ISRUN3:
        # Binning and ticks for ISRUN3 = True (to 0.8!)
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        xlow = bins[0]
        xhigh = bins[-1] 
        x_label = "BDT Score"
        x_ticks = bins 
        
    else:  # For ISRUN3 = False
        # Default binning and ticks for ISRUN3 = False
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        xlow = bins[0]
        xhigh = bins[-1]
        x_label = "BDT Score"
        x_ticks = bins
    
  else:
    raise ValueError(f"Unknown variable {name_of_variable} in cfg.py")

  # Printing out the variable info to the screen when the function is run
  print("From cfg.py...")
  print(f"Variable definition: {name_of_variable}")
  print(f"Reconstructed variable = {xvar}")
  print(f"True variable = {true_var}")
  print(f"Bins = {bins}")
  print(f"Lowest x-value on plot = {xlow}")
  print(f"Highest x-value on plot = {xhigh}")
  print(f"x-axis title = {x_label}")
  print(f"x-axis labels = {x_ticks}")

  return xvar, true_var, bins, xlow, xhigh, x_label, x_ticks # Order dependent! 

#### EXAMPLE USAGE ####
# Variables = "Opening Angle", "Proton Multiplicity", "Electron Energy", "Visible Energy", "BDT Score"
# name_of_variable = "Opening Angle" 
# xvar, true_var, bins, xlow, xhigh, x_label, x_ticks = get_variable_info(name_of_variable, ISRUN3) # Order dependent! 
# The output will be the variables printed out. 
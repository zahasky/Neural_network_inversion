# -*- coding: utf-8 -*-
"""
Three-dimensional transport in a uniform flow field 
This is based on MT3DMS Example problem 7 available here: https://github.com/modflowpy/flopy/blob/develop/examples/Notebooks/flopy3_MT3DMS_examples.ipynb

Created on Fri May 22 16:50:00 2020

@author: Revised by Christopher Zahasky to add timing and much more commenting

"""

# All packages called by functions should be imported
import sys
import os
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import math
import time

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy
    
# from flopy.utils.util_array import read1d

# # set figure size, call mpl.rcParams to see all variables that can be changed
# mpl.rcParams['figure.figsize'] = (8, 8)

# names of executable with path IF NOT IN CURRENT DIRECTORY
# exe_name_mf = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
# exe_name_mf = 'mf2005'
# exe_name_mt = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# # directory to save data
# directory_name = 'data_3D_model'
# datadir = os.path.join('..', directory_name, 'mt3d_test', 'mt3dms')
# workdir = os.path.join('.', directory_name)

# # uncomment if you want the information about numpy, matplotlib, and flopy printed    
# # print(sys.version)
# # print('numpy version: {}'.format(np.__version__))
# # print('matplotlib version: {}'.format(mpl.__version__))
# # print('flopy version: {}'.format(flopy.__version__))


def mt3d_pulse_injection_sim(dirname, model_ws, raw_hk, grid_size, perlen_mf, nprs, mixelm, exe_name_mf, exe_name_mt):
    # # Model workspace and new sub-directory
    # model_ws = os.path.join(workdir, dirname)
    
    # Call function and time it
    start = time.time() # start a timer
# =============================================================================
#     UNIT INFORMATION
# =============================================================================
    # units must be set for both MODFLOW and MT3D, they have different variable names for each
    # time units (itmuni in MODFLOW discretization package)
    # 1 = seconds, 2 = minutes, 3 = hours, 4 = days, 5 = years
    itmuni = 2 # MODFLOW length units
    mt_tunit = 'M' # MT3D units
    # length units (lenuniint in MODFLOW discretization package)
    # 0 = undefined, 1 = feet, 2 = meters, 3 = centimeters
    lenuni = 3 # MODFLOW units
    mt_lunit = 'CM' # MT3D units
    
# =============================================================================
#     MODEL DIMENSION AND MATERIAL PROPERTY INFORMATION
# =============================================================================
    # Make model dimensions the same size as the hydraulic conductivity field input 
    # NOTE: that the there are two additional columns added as dummy slices (representing coreholder faces)
    hk_size = raw_hk.shape
    # determine dummy slice perm based on maximum hydraulic conductivity
    dummy_slice_hk = 10*raw_hk.max()
    # define dummy slice
    dummy_array = dummy_slice_hk * np.ones((hk_size[0], hk_size[1], 1), dtype=np.float)
    # concantenate dummy slice on hydraulic conductivity array
    hk = np.concatenate((dummy_array, raw_hk, dummy_array), axis=2)
    # Model information (true in all models called by 'p01')
    nlay = hk_size[0] # number of layers / grid cells
    nrow = hk_size[1] # number of rows / grid cells
    ncol = hk_size[2]+2 # number of columns (parallel to axis of core)
    delr = grid_size[0] # grid size in direction of Lx
    delc = grid_size[1] # grid size in direction of Ly
    delv = grid_size[2] # grid size in direction of Lz
    
    laytyp = 0
    # cell elevations are specified in variable BOTM. A flat layer is indicated
    # by simply specifying the same value for the bottom elevation of all cells in the layer
    botm = [-delv * k for k in range(1, nlay + 1)]
    
    # ADDITIONAL MATERIAL PROPERTIES
    prsity = 0.3 # porosity. float or array of floats (nlay, nrow, ncol)
    al = 0.1 # longitudental dispersivity
    trpt = 0.3 # ratio of horizontal transverse dispersivity to longitudenal dispersivity
    trpv = 0.3 # ratio of vertical transverse dispersivity to longitudenal dispersivity
    
# =============================================================================
#     BOUNDARY AND INTIAL CONDITIONS
# =============================================================================
    # backpressure, give this in kPa for conversion
    bp_kpa = 70
    # Injection rate in defined units 
    injection_rate = 0.5 # [cm^3/min]
    # Core radius
    core_radius = 2.54 # [cm]
    # Calculation of core area
    core_area = math.pi*core_radius**2
    # specific discharge or injection rate/area
    q = injection_rate/core_area # [cm/min]
    
    # number of stress periods (MF input), calculated from period length input
    nper = len(perlen_mf)    
    # Initial concentration (MT input)
    c0 = 1.
    # Stress period 2 concentration
    c1 = 0.0
    # period length in selected units (provided as input, but could be defined here)
    # perlen_mf = [1., 50]
    
    # MODFLOW head boundary conditions, <0 = specified head, 0 = no flow, >0 variable head
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int)
    # inlet conditions (currently set with well so inlet is zero)
    # ibound[:, :, 0] = -1
    # outlet conditions
    ibound[:, :, -1] = -1
    
    # MODFLOW constant initial head conditions
    strt = np.zeros((nlay, nrow, ncol), dtype=np.float)
    # convert backpressure to head units
    if lenuni == 3: # centimeters
        hout = bp_kpa*1000/(1000*9.81)*100 
    else: # elseif meters
        if lenuni == 2: 
            hout = bp_kpa*1000/(1000*9.81)
    # assign outlet pressure as head converted from 'bp_kpa' input variable
    strt[:, :, -1] = hout
    
    # Stress period well data for MODFLOW. Each well is defined through defintition
    # of layer (int), row (int), column (int), flux (float). The first number corresponds to the stress period
    # Example for 1 stress period: spd_mf = {0:[[0, 0, 1, q],[0, 5, 1, q]]}
    well_info = np.zeros((nlay*nrow, 4), dtype=np.float)
    # Nested loop to define every inlet face grid cell as a well
    for layer in range(0, nlay):
        for row in range(0, nrow):
            index_n = layer*nrow + row
            # print(index_n)
            well_info[index_n] = [layer, row, 0, q]   
    # Now insert well information into stress period data 
    # *** TO DO: Generalize this for multiple stress periods
    # This has the form: spd_mf = {0:[[0, 0, 0, q],[0, 5, 1, q]], 1:[[0, 1, 1, q]]}
    spd_mf={0:well_info}
    
    # MT3D stress period data, note that the indices between 'spd_mt' must exist in 'spd_mf' 
    # This is used as input for the source and sink mixing package
    # Itype is an integer indicating the type of point source, 2=well, 3=drain, -1=constant concentration
    itype = 2
    cwell_info = np.zeros((nlay*nrow, 5), dtype=np.float)
    # Nested loop to define every inlet face grid cell as a well
    for layer in range(0, nlay):
        for row in range(0, nrow):
            index_n = layer*nrow + row
            cwell_info[index_n] = [layer, row, 0, c0, itype]
            
    # Second stress period        
    cwell_info2 = cwell_info.copy()   
    cwell_info2[:,3] = c1 
    # Now apply stress period info    
    spd_mt = {0:cwell_info, 1:cwell_info2}

    # Concentration boundary conditions set with well so this is commented out
    # c0 = 1.
    # icbund = np.ones((nlay, nrow, ncol), dtype=np.int)
    # icbund[0, 0, 0] = -1
    # Initial concentration conditions, currently set to zero everywhere
    # sconc = np.zeros((nlay, nrow, ncol), dtype=np.float)
    # sconc[0, 0, 0] = c0
    
# =============================================================================
# MT3D OUTPUT CONTROL 
# =============================================================================
    # nprs (int): A flag indicating (i) the frequency of the output and (ii) whether 
    #     the output frequency is specified in terms of total elapsed simulation 
    #     time or the transport step number. If nprs > 0 results will be saved at 
    #     the times as specified in timprs; if nprs = 0, results will not be saved 
    #     except at the end of simulation; if NPRS < 0, simulation results will be 
    #     saved whenever the number of transport steps is an even multiple of nprs. (default is 0).
    # nprs = 20
    
    # timprs (list of float): The total elapsed time at which the simulation 
    #     results are saved. The number of entries in timprs must equal nprs. (default is None).
    timprs = np.linspace(0, np.sum(perlen_mf), nprs, endpoint=False)
    # obs (array of int): An array with the cell indices (layer, row, column) 
    #     for which the concentration is to be printed at every transport step. 
    #     (default is None). obs indices must be entered as zero-based numbers as 
    #     a 1 is added to them before writing to the btn file.
    # nprobs (int): An integer indicating how frequently the concentration at 
    #     the specified observation points should be saved. (default is 1).
    
    # Particle output control
    dceps = 1.e-5
    nplane = 1
    npl = 0 # number of initial particles per cell to be placed in cells with a concentration less than 'dceps'
    nph = 16
    npmin = 2
    npmax = 32 # maximum number of particles allowed per cell
    dchmoc = 1.e-3
    nlsink = nplane
    npsink = nph
    
# =============================================================================
# START CALLING MODFLOW PACKAGES AND RUN MODEL
# =============================================================================
    # Start callingwriting files
    modelname_mf = dirname + '_mf'
    # same as 1D model
    mf = flopy.modflow.Modflow(modelname=modelname_mf, model_ws=model_ws, exe_name=exe_name_mf)
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper,
                                   delr=delr, delc=delc, top=0., botm=botm,
                                   perlen=perlen_mf, steady=True, itmuni=itmuni, lenuni=lenuni)
    
    # MODFLOW basic package class
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    # MODFLOW layer properties flow package class
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, laytyp=laytyp)
    # MODFLOW well package class
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=spd_mf)
    # MODFLOW preconditioned conjugate-gradient package class
    pcg = flopy.modflow.ModflowPcg(mf)
    # MODFLOW Link-MT3DMS Package Class (this is the package for solute transport)
    lmt = flopy.modflow.ModflowLmt(mf)
    # # MODFLOW output control package
    oc = flopy.modflow.ModflowOc(mf)
    
    mf.write_input()
    # RUN MODFLOW MODEL, set to silent=False to see output in terminal
    mf.run_model(silent=True)
    
# =============================================================================
# START CALLING MT3D PACKAGES AND RUN MODEL
# =============================================================================
    # RUN MT3dms solute tranport 
    modelname_mt = dirname + '_mt'

    # MT3DMS Model Class
    # Input: modelname = 'string', namefile_ext = 'string' (Extension for the namefile (the default is 'nam'))
    # modflowmodelflopy.modflow.mf.Modflow = This is a flopy Modflow model object upon which this Mt3dms model is based. (the default is None)
    mt = flopy.mt3d.Mt3dms(modelname=modelname_mt, model_ws=model_ws, 
                           exe_name=exe_name_mt, modflowmodel=mf)
    
    # Basic transport package class
    btn = flopy.mt3d.Mt3dBtn(mt, ncomp=1, icbund=1, prsity=prsity, sconc=0, 
                             tunit=mt_tunit, lunit=mt_lunit, nprs=nprs, timprs=timprs)
    
    # mixelm is an integer flag for the advection solution option, 
    # mixelm = 0 is the standard finite difference method with upstream or 
    # central in space weighting.
    # mixelm = 1 is the forward tracking method of characteristics, this seems to result in minimal numerical dispersion.
    # mixelm = 2 is the backward tracking
    # mixelm = 3 is the hybrid method
    # mixelm = -1 is the third-ord TVD scheme (ULTIMATE)
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm, dceps=dceps, nplane=nplane, 
                             npl=npl, nph=nph, npmin=npmin, npmax=npmax,
                             nlsink=nlsink, npsink=npsink, percel=0.5)

    
    dsp = flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt)
    # Reactions package (optional)
    # rct = flopy.mt3d.Mt3dRct(mt, isothm=1, ireact=1, igetsc=0, rhob=rhob, sp1=kd, 
    #                          rc1=lambda1, rc2=lambda1)
    # source and sink mixing package
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=spd_mt)
    gcg = flopy.mt3d.Mt3dGcg(mt)
    
    mt.write_input()
    fname = os.path.join(model_ws, 'MT3D001.UCN')
    if os.path.isfile(fname):
        os.remove(fname)
    mt.run_model(silent=True)
    
    # Extract concentration information
    fname = os.path.join(model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.UcnFile(fname)
    timearray = ucnobj.get_times()
    # print(times)
    conc = ucnobj.get_alldata()
    
    # Extract head information
    fname = os.path.join(model_ws, modelname_mf+'.hds')
    hdobj = flopy.utils.HeadFile(fname)
    heads = hdobj.get_data()
    
    # convert heads to pascals
    if lenuni == 3: # centimeters
        pressures = heads/100*(1000*9.81) 
    else: # elseif meters
        if lenuni == 2: 
            pressures = heads*(1000*9.81)
            
    
    # crop off extra concentration slices
    conc = conc[:,:,:,1:-1]
    
    # crop off extra pressure slices
    pressures = pressures[:,:,1:-1]

    # Print final run time
    end = time.time() # end timer
    print('Model run time: ', end - start) # show run time

    return mf, mt, conc, timearray, pressures


# Function to calculate the mean breakthrough time
def mean_bt_map(conc, grid_size, perlen_mf, timearray):
    # also output the mean breakthrough time map
    conc_size = conc.shape
    # Append the stress period end time as this is also recorded in addition to 
    # time specified in 'timprs'
    # sp_end_times = np.cumsum(perlen_mf)
    # # append times to end of timprs array
    # t_appended = np.append(timprs, sp_end_times) 
    # # sort chronologically
    # t_sorted = np.sort(t_appended)
    # # remove repeats
    # timearray = np.unique(t_sorted)  
    # print(timearray)
    # print(timearray)
    
    # Preallocate breakthrough time array
    bt_array = np.zeros((conc_size[1], conc_size[2], conc_size[3]), dtype=np.float)

    for layer in range(0, conc_size[1]):
        for row in range(0, conc_size[2]):
            for col in range(0, conc_size[3]):
                cell_btc = conc[:, layer, row, col]
            
                # check to make sure tracer is in grid cell
                if cell_btc.sum() > 0:
                    # calculate zero moment
                    m0 = np.trapz(cell_btc, timearray)
                    # calculate first moment of grid cell 
                    m1 = np.trapz(cell_btc*timearray, timearray);
                    
                    # calculate center of mass in time (mean breakthrough time)
                    bt_array[layer, row, col] = m1/m0

    # shift breakthrough time so that mean of first slice is half-way through the pulse injection
    mean_bt_inlet = np.mean(bt_array[:,:,0])
    half_inj_time = (perlen_mf[0]/2)
    bt_array -= mean_bt_inlet 
    # bt_array += half_inj_time

    # Arrival time difference map calculation
# % new mean breakthrough time at the inlet
    mean_bt_inlet = np.mean(bt_array[:,:,0])
    # calculate mean arrival time at the outlet
    mean_bt_outlet = np.mean(bt_array[:,:,-1])
    # Calculate velocity from difference in mean arrival time between inlet and outlet
    model_length = grid_size[2]*(conc_size[3]-1)
    v = model_length/(mean_bt_outlet - mean_bt_inlet)
    # Now calculate what the mean arrival would be if the velocity was constant everywhere
    xg = np.arange(0, (model_length+grid_size[2]), grid_size[2]) #+ (grid_size[2]/2)
    # vector of ideal mean arrival time based average v
    bt_ideal = xg/v
# % shifted
# mean_xt = mean_xt -(mean_xt(1) - (inj_t/2));
    # Turn this vector into a matrix so that it can simple be subtracted from
    bt_ideal_array = np.tile(bt_ideal, (conc_size[1], conc_size[2], 1))

# % the arrive time map calculated from the PET data
# mean_xt3d(1,1,1:PET_size(3)) = mean_xt;
# Xth = repmat(mean_xt3d, PET_size(1),PET_size(1),1);
    # % Arrival time difference map
    bt_diff = (bt_ideal_array - bt_array)
    # normalized
    bt_diff_norm = bt_diff*(v/model_length)

    return bt_array, bt_diff, bt_diff_norm

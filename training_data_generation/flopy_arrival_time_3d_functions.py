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
from scipy import integrate
import time
import matplotlib.pyplot as plt

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy

# # uncomment if you want the information about numpy, matplotlib, and flopy printed    
# # print(sys.version)
# # print('numpy version: {}'.format(np.__version__))
# # print('matplotlib version: {}'.format(mpl.__version__))
# # print('flopy version: {}'.format(flopy.__version__))


def mt3d_pulse_injection_sim(dirname, model_ws, raw_hk, prsity_field, grid_size, 
                             ndummy_in, perlen_mt, nprs, mixelm, exe_name_mf, exe_name_mt):
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
#     STRESS PERIOD INFO
# =============================================================================
    perlen_mf = [np.sum(perlen_mt)]
    # number of stress periods (MF input), calculated from period length input
    nper_mf = len(perlen_mf)
    # number of stress periods (MT input), calculated from period length input
    nper = len(perlen_mt)
    
# =============================================================================
#     MODEL DIMENSION AND MATERIAL PROPERTY INFORMATION
# =============================================================================
    # Make model dimensions the same size as the hydraulic conductivity field input 
    # NOTE: that the there are two additional columns added as dummy slices (representing coreholder faces)
    hk_size = raw_hk.shape
    # determine dummy slice perm based on maximum hydraulic conductivity
    dummy_slice_hk = raw_hk.max()*10
    # define area with hk values above zero
    core_mask = np.ones((hk_size[0], hk_size[1]))
    core_mask = np.multiply(core_mask, raw_hk[:,:,0])
    core_mask[np.nonzero(core_mask)] = 1
    # define hk in cells with nonzero hk to be equal to 10x the max hk
    # This represents the 'coreholder' slices
    dummy_ch = core_mask[:,:, np.newaxis]*dummy_slice_hk
    dummy_ch_por = core_mask[:,:, np.newaxis]*0.15
    
    # Additional dummy inlet to replicate imperfect boundary
    dummy_slice_in =raw_hk[:,:,0]*core_mask
    dummy_slice_in = dummy_slice_in.reshape(hk_size[0], hk_size[1], 1)
    # concantenate dummy slice on hydraulic conductivity array
    ndummy_in = int(ndummy_in)
    if ndummy_in > 0:
        dummy_block = np.repeat(dummy_slice_in, ndummy_in, axis=2)
        hk = np.concatenate((dummy_ch, dummy_block, raw_hk, dummy_ch), axis=2)
        
        # do the same with porosity slices
        dummy_slice_in =prsity_field[:,:,0]*core_mask
        dummy_slice_in = dummy_slice_in.reshape(hk_size[0], hk_size[1], 1)
        dummy_block = np.repeat(dummy_slice_in, ndummy_in, axis=2)
        prsity = np.concatenate((dummy_ch_por, dummy_block, prsity_field, dummy_ch_por), axis=2)
        
    else:
        hk = np.concatenate((dummy_ch, raw_hk, dummy_ch), axis=2)
        prsity = np.concatenate((dummy_ch_por, prsity_field, dummy_ch_por), axis=2)
        
    # hk = raw_hk
    # Model information (true in all models called by 'p01')
    nlay = int(hk_size[0]) # number of layers / grid cells
    nrow = int(hk_size[1]) # number of rows / grid cells
    # ncol = hk_size[2]+2+ndummy_in # number of columns (along to axis of core)
    ncol = int(hk_size[2]+2+ndummy_in) # number of columns (along to axis of core)
    # ncol = hk_size[2] # number of columns (along to axis of core)
    delv = grid_size[0] # grid size in direction of Lx (nlay)
    delr = grid_size[1] # grid size in direction of Ly (nrow)
    delc = grid_size[2] # grid size in direction of Lz (ncol)
    
    laytyp = 0
    # cell elevations are specified in variable BOTM. A flat layer is indicated
    # by simply specifying the same value for the bottom elevation of all cells in the layer
    botm = [-delv * k for k in range(1, nlay + 1)]
    
    # ADDITIONAL MATERIAL PROPERTIES
    # prsity = 0.25 # porosity. float or array of floats (nlay, nrow, ncol)
    prsity = prsity
    al = 0.1 # longitudental dispersivity
    trpt = 0.3 # ratio of horizontal transverse dispersivity to longitudenal dispersivity
    trpv = 0.3 # ratio of vertical transverse dispersivity to longitudenal dispersivity
    
# =============================================================================
#     BOUNDARY AND INTIAL CONDITIONS
# =============================================================================
    # backpressure, give this in kPa for conversion
    bp_kpa = 70
    # Injection rate in defined units 
    injection_rate = 2 # [cm^3/min]
    
     # Initial concentration (MT input)
    c0 = 0.
    # Stress period 2 concentration
    c1 = 1.0
    
    # Core radius
    core_radius = 2.54 # [cm]
    # Calculation of core area
    core_area = 3.1415*core_radius**2
    # Calculation of mask area
    mask_area = np.sum(core_mask)*grid_size[0]*grid_size[1]
    # total specific discharge or injection flux (rate/area)
    # q = injection_rate*(mask_area/core_area)/np.sum(core_mask)
    # scale injection rate locally by inlet permeability
    q_total = injection_rate*(mask_area/core_area)
    # q_total = injection_rate/core_area
    q = q_total/np.sum(dummy_slice_in)
    
    # MODFLOW head boundary conditions, <0 = specified head, 0 = no flow, >0 variable head
    # ibound = np.ones((nlay, nrow, ncol), dtype=np.int)
    ibound = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)
    
    # inlet conditions (currently set with well so inlet is zero)
    # ibound[:, :, 0] = ibound[:, :, -1]*-1
    # outlet conditions
    # ibound[5:15, 5:15, -1] = -1
    ibound[:, :, -1] = ibound[:, :, -1]*-1
    
    # MODFLOW constant initial head conditions
    strt = np.zeros((nlay, nrow, ncol), dtype=float)
    # Lx = (hk_size[2]*delc)
    # Q = injection_rate/(core_area)
    # geo_mean_k = np.exp(np.sum(np.log(hk[hk>0]))/len(hk[hk>0]))
    # h1 = Q * Lx/geo_mean_k
    # print(h1)
    
    # convert backpressure to head units
    if lenuni == 3: # centimeters
        hout = bp_kpa*1000/(1000*9.81)*100 
    else: # elseif meters
        if lenuni == 2: 
            hout = bp_kpa*1000/(1000*9.81)
    # assign outlet pressure as head converted from 'bp_kpa' input variable
    # index the inlet cell
    # strt[:, :, 0] = h1+hout
    strt[:, :, -1] = core_mask*hout
    # strt[:, :, -1] = hout
    
    # Stress period well data for MODFLOW. Each well is defined through defintition
    # of layer (int), row (int), column (int), flux (float). The first number corresponds to the stress period
    # Example for 1 stress period: spd_mf = {0:[[0, 0, 1, q],[0, 5, 1, q]]}
    well_info = np.zeros((int(np.sum(core_mask)), 4), dtype=float)
    # Nested loop to define every inlet face grid cell as a well
    index_n = 0
    for layer in range(0, nlay):
        for row in range(0, nrow):
            # index_n = layer*nrow + row
            # index_n +=1
            # print(index_n)
            if core_mask[layer, row] > 0:
                well_info[index_n] = [layer, row, 0, q*dummy_slice_in[layer,row]]   
                index_n +=1
    
    
    # geo_mean_k = np.exp(np.sum(np.log(hk[hk>0]))/len(hk[hk>0]))
    
    # # MODFLOW head boundary conditions, <0 = specified head, 0 = no flow, >0 variable head
    # # ibound = np.ones((nlay, nrow, ncol), dtype=np.int)
    # ibound = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)
    
    # # inlet conditions (currently set with well so inlet is zero)
    # ibound[:, :, 0] = ibound[:, :, -1]*-1
    # # outlet conditions
    # # ibound[5:15, 5:15, -1] = -1
    # ibound[:, :, -1] = ibound[:, :, -1]*-1
    
    # # MODFLOW constant initial head conditions
    # strt = np.zeros((nlay, nrow, ncol), dtype=np.float)
    # Lx = (hk_size[2]*delc)
    # q = injection_rate/(core_area)
    # h1 = q * Lx/geo_mean_k
    # print(h1)
    
    # # convert backpressure to head units
    # if lenuni == 3: # centimeters
    #     hout = bp_kpa*1000/(1000*9.81)*100 
    # else: # elseif meters
    #     if lenuni == 2: 
    #         hout = bp_kpa*1000/(1000*9.81)
    # # assign outlet pressure as head converted from 'bp_kpa' input variable
    # # index the inlet cell
    # strt[:, :, 0] = h1+hout
    # # strt[:, :, -1] = core_mask*hout
    # strt[:, :, -1] = hout
    
    # # Stress period well data for MODFLOW. Each well is defined through defintition
    # # of layer (int), row (int), column (int), flux (float). The first number corresponds to the stress period
    # # Example for 1 stress period: spd_mf = {0:[[0, 0, 1, q],[0, 5, 1, q]]}
    # well_info = np.zeros((int(np.sum(core_mask)), 4), dtype=np.float)
    # # Nested loop to define every inlet face grid cell as a well
    # index_n = 0
    # for layer in range(0, nlay):
    #     for row in range(0, nrow):
    #         # index_n = layer*nrow + row
    #         # index_n +=1
    #         # print(index_n)
    #         if core_mask[layer, row] > 0:
    #             well_info[index_n] = [layer, row, 0, q]   
    #             index_n +=1
                
    # Now insert well information into stress period data 
    # *** TO DO: Generalize this for multiple stress periods
    # This has the form: spd_mf = {0:[[0, 0, 0, q],[0, 5, 1, q]], 1:[[0, 1, 1, q]]}
    spd_mf={0:well_info}
    
    # MT3D stress period data, note that the indices between 'spd_mt' must exist in 'spd_mf' 
    # This is used as input for the source and sink mixing package
    # Itype is an integer indicating the type of point source, 2=well, 3=drain, -1=constant concentration
    itype = 2
    cwell_info = np.zeros((int(np.sum(core_mask)), 5), dtype=float)
    # cwell_info = np.zeros((nrow*nlay, 5), dtype=np.float)
    # Nested loop to define every inlet face grid cell as a well
    index_n = 0
    for layer in range(0, nlay):
        for row in range(0, nrow):
            # index_n = layer*nrow + row
            if core_mask[layer, row] > 0:
                cwell_info[index_n] = [layer, row, 0, c0, itype] 
                index_n +=1
            # cwell_info[index_n] = [layer, row, 0, c0, itype]
            
    # Second stress period        
    cwell_info2 = cwell_info.copy()   
    cwell_info2[:,3] = c1 
    # Second stress period        
    cwell_info3 = cwell_info.copy()   
    cwell_info3[:,3] = c0 
    # Now apply stress period info    
    spd_mt = {0:cwell_info, 1:cwell_info2, 2:cwell_info3}

    # Concentration boundary conditions, this is neccessary to indicate 
    # inactive concentration cells outside of the more
    # If icbund = 0, the cell is an inactive concentration cell; 
    # If icbund < 0, the cell is a constant-concentration cell; 
    # If icbund > 0, the cell is an active concentration cell where the 
    # concentration value will be calculated. (default is 1).
    icbund = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)
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
    timprs = np.linspace(0, np.sum(perlen_mt), nprs, endpoint=False)
    # obs (array of int): An array with the cell indices (layer, row, column) 
    #     for which the concentration is to be printed at every transport step. 
    #     (default is None). obs indices must be entered as zero-based numbers as 
    #     a 1 is added to them before writing to the btn file.
    # nprobs (int): An integer indicating how frequently the concentration at 
    #     the specified observation points should be saved. (default is 1).
    
# =============================================================================
# START CALLING MODFLOW PACKAGES AND RUN MODEL
# =============================================================================
    # Start callingwriting files
    modelname_mf = dirname + '_mf'
    # same as 1D model
    mf = flopy.modflow.Modflow(modelname=modelname_mf, model_ws=model_ws, exe_name=exe_name_mf)
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper_mf,
                                   delr=delr, delc=delc, top=0., botm=botm,
                                   perlen=perlen_mf, itmuni=itmuni, lenuni=lenuni)
    
    # MODFLOW basic package class
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    # MODFLOW layer properties flow package class
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, laytyp=laytyp)
    # MODFLOW well package class
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=spd_mf)
    # MODFLOW preconditioned conjugate-gradient package class
    pcg = flopy.modflow.ModflowPcg(mf, mxiter=100, rclose=1e-5, relax=0.97)
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
    btn = flopy.mt3d.Mt3dBtn(mt, icbund=icbund, prsity=prsity, sconc=0, 
                             tunit=mt_tunit, lunit=mt_lunit, nper=nper, perlen=perlen_mt, 
                             nprs=nprs, timprs=timprs)
    
    # mixelm is an integer flag for the advection solution option, 
    # mixelm = 0 is the standard finite difference method with upstream or 
    # central in space weighting.
    # mixelm = 1 is the forward tracking method of characteristics, this seems to result in minimal numerical dispersion.
    # mixelm = 2 is the backward tracking
    # mixelm = 3 is the hybrid method
    # mixelm = -1 is the third-ord TVD scheme (ULTIMATE)
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm)

    dsp = flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt)
    # Reactions package (optional)
    # rct = flopy.mt3d.Mt3dRct(mt, isothm=1, ireact=1, igetsc=0, rhob=rhob, sp1=kd, 
    #                           rc1=lambda1, rc2=lambda1)
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
    
    # set inactive cell pressures to zero, by default inactive cells have a pressure of -999
    # heads[heads < -990] = 0
    
    # convert heads to pascals
    if lenuni == 3: # centimeters
        pressures = heads/100*(1000*9.81) 
    else: # elseif meters
        if lenuni == 2: 
            pressures = heads*(1000*9.81)
            
    
    # crop off extra concentration slices
    conc = conc[:,:,:,ndummy_in+1:-1]
    
    # calculate pressure drop
    # crop off extra pressure slices
    # pressures = pressures[:,:,ndummy_in:-2] # commented out since not returned
    # calculate 
    p_inlet = pressures[:, :, ndummy_in+1]*core_mask
    p_inlet = np.mean(p_inlet[p_inlet>1])
    # print(p_inlet)
    p_outlet = pressures[:, :, -1]*core_mask
    p_outlet = np.mean(p_outlet[p_outlet>1])
    dp = p_inlet-p_outlet
    # print('Pressure drop: '+ str(dp/1000) + ' kPa')
    
    # calculate mean permeability from pressure drop
    # water viscosity
    mu_water = 0.00089 # Pa.s
    L = hk_size[2]*delc
    km2_mean = (q_total/mask_area)*L*mu_water/dp /(60*100**2)
    
    print('Core average perm: '+ str(km2_mean/9.869233E-13*1000) + ' mD')
    
    # Option to plot and calculate geometric mean to double check that core average perm in close
    geo_mean_K = np.exp(np.sum(np.log(raw_hk[raw_hk>0]))/len(raw_hk[raw_hk>0]))
    geo_mean_km2 = geo_mean_K/(1000*9.81*100*60/8.9E-4)
    print('Geometric mean perm: ' + str(geo_mean_km2/9.869233E-13*1000) + ' mD')

    # Print final run time
    end_time = time.time() # end timer
    # print('Model run time: ', end - start) # show run time
    print(f"Model run time: {end_time - start:0.4f} seconds")
    
    return mf, mt, conc, timearray, km2_mean


# Much faster quantile calculation 
def quantile_calc(btc_1d, timearray, quantile):
    # calculate cumulative amount of solute passing by location
    M0i = integrate.cumtrapz(btc_1d, timearray)
    # normalize by total to get CDF
    quant = M0i/M0i[-1]
    # calculate midtimes
    mid_time = (timearray[1:] + timearray[:-1]) / 2.0
    
    # now linearly interpolate to find quantile
    gind = np.argmax(quant > quantile)
    m = (quant[gind] - quant[gind-1])/(mid_time[gind] - mid_time[gind-1])
    b = quant[gind-1] - m*mid_time[gind-1]
    
    tau = (quantile-b)/m
    
    # plot check
    # xp = [mid_time[gind-1], mid_time[gind]]
    # plt.plot(mid_time, quant, '-o')
    # plt.plot(xp, m*xp+b, '-r')
    # plt.plot(tau, quantile, 'ok')
    return tau
    

# Function to calculate the quantile arrival time map
def flopy_arrival_map_function(conc, timearray, grid_size, quantile):
    # determine the size of the data
    conc_size = conc.shape
    
    # define area with hk values above zero
    core_mask = np.copy(conc[0,:,:,0])
    core_mask[core_mask<2] = 1
    core_mask[core_mask>2] = 0
    
    # MT3D sets the values of all concentrations in cells outside of the model 
    # to 1E30, this sets them to 0
    conc[conc>2]=0
    
    # sum of slice concentrations for calculating inlet and outlet breakthrough
    oned = np.nansum(np.nansum(conc, 1), 1)
    
    # arrival time calculation in inlet slice
    tau_in = quantile_calc(oned[:,0], timearray, quantile)
    
    # arrival time calculation in outlet slice
    tau_out = quantile_calc(oned[:,-1], timearray, quantile)

    # core length
    model_length = grid_size[2]*conc_size[3]
    # array of grid cell centers before interpolation
    z_coord_model = np.arange(grid_size[2]/2, model_length, grid_size[2])
    
    # Preallocate arrival time array
    at_array = np.zeros((conc_size[1], conc_size[2], conc_size[3]), dtype=np.float)
    
    for layer in range(0, conc_size[1]):
        for row in range(0, conc_size[2]):
            for col in range(0, conc_size[3]):
                # Check if outside core
                if core_mask[layer, row] > 0:
                    cell_btc = conc[:, layer, row, col]
                    # check to make sure tracer is in grid cell
                    if cell_btc.sum() > 0:
                        # call function to find quantile of interest
                        tau_vox = quantile_calc(cell_btc, timearray, quantile)
                        if tau_vox > 0:
                            at_array[layer, row, col] = tau_vox
                        else:
                            break # if tau can't be calculated then break the nested for loop and run a different model
            else:
                continue
            break
        else:
            continue
        break
    
    if tau_vox == 0: # these set a flag that is used to regenerate training realization
        at_array = 0 
        at_array_norm = 0 
        at_diff_norm = 0
        
    else:
        # v = (model_length-grid_size[2])/(tau_out - tau_in)
        # print('advection velocity: ' + str(v))
    
        # Normalize arrival times
        at_array_norm = (at_array-tau_in)/(tau_out - tau_in)
    
        # vector of ideal mean arrival time based average v
        at_ideal = z_coord_model/z_coord_model[-1]

        # Turn this vector into a matrix so that it can simply be subtracted from
        at_ideal_array = np.tile(at_ideal, (conc_size[1], conc_size[2], 1))

        # Arrival time difference map
        at_diff_norm = (at_ideal_array - at_array_norm)
    
        # Replace values outside model with zeros
        for col in range(0, conc_size[3]):
            at_array[:,:,col] = np.multiply(at_array[:,:,col], core_mask)
            at_array_norm[:,:,col] = np.multiply(at_array_norm[:,:,col], core_mask)
            at_diff_norm[:,:,col] = np.multiply(at_diff_norm[:,:,col], core_mask)

    return at_array, at_array_norm, at_diff_norm


def plot_2d(map_data, dx, dy, colorbar_label, cmap):
    # fontsize
    fs = 18
    hfont = {'fontname':'Arial'}
    r, c = np.shape(map_data)
    # Define grid
    # Lx = c * dx   # length of model in selected units 
    # Ly = r * dy   # length of model in selected units
    # x, y = np.mgrid[slice(0, Lx + dx, dx), slice(0, Ly + dy, dy)]
    
  
    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    
    X, Y = np.meshgrid(x_coord, y_coord)
    # print(slice(0, Ly + dy, dy))
    # print(c)
    # print(slice(0, Lx + dx, dx))
    # print(r)
    
    # fig, ax = plt.figure(figsize=(10, 10) # adjust these numbers to change the size of your figure
    # ax.axis('equal')          
    # fig2.add_subplot(1, 1, 1, aspect='equal')
    # Use 'pcolor' function to plot 2d map of concentration
    # Note that we are flipping map_data and the yaxis to so that y increases downward
    plt.figure(figsize=(12, 4), dpi=200)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'nearest', edgecolor ='k', linewidth = 0.01)
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    # plt.clim(cmin, cmax) 
    # label the colorbar
    cbar.set_label(colorbar_label, fontsize=fs, **hfont)
    # make colorbar font bigger
    cbar.ax.tick_params(labelsize= (fs-2)) 
    # make axis fontsize bigger!
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 
    # Label x-axis
    plt.gca().invert_yaxis()
# All packages called by functions should be imported
import sys
import os
import numpy as np
import time
import flopy

def mt3d_pulse_injection_sim(p, dirname, model_ws, raw_hk, grid_size, perlen_mf, nprs, mixelm, exe_name_mf, exe_name_mt):
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
    # define area with hk values above zero
    core_mask = np.ones((hk_size[0], hk_size[1]))
    core_mask = np.multiply(core_mask, raw_hk[:,:,0])
    core_mask[np.nonzero(core_mask)] = 1
    # define hk in cells with nonzero hk to be equal to 10x the max hk
    dummy_array = core_mask[:,:, np.newaxis]*dummy_slice_hk
    # concantenate dummy slice on hydraulic conductivity array
    hk = np.concatenate((dummy_array, raw_hk, dummy_array), axis=2)
    # Model information (true in all models called by 'p01')
    nlay = hk_size[0] # number of layers / grid cells
    nrow = hk_size[1] # number of rows / grid cells
    ncol = hk_size[2]+2 # number of columns (along to axis of core)
    delv = grid_size[0] # grid size in direction of Lx (nlay)
    delr = grid_size[1] # grid size in direction of Ly (nrow)
    delc = grid_size[2] # grid size in direction of Lz (ncol)

    laytyp = 0
    # cell elevations are specified in variable BOTM. A flat layer is indicated
    # by simply specifying the same value for the bottom elevation of all cells in the layer
    botm = [-delv * k for k in range(1, nlay + 1)]
    
    # ADDITIONAL MATERIAL PROPERTIES
    #dummy_array_prsity = core_mask[:,:, np.newaxis]*0.15 
    # Loading the Porosity data (A np array for the heterogenous porosity case and nothing for the homogeneous porosity case)
    # porosity. float or array of floats (nlay, nrow, ncol)
    #prsity = np.concatenate((dummy_array_prsity, p, dummy_array_prsity), axis=2) 
    prsity = p

    al = 0.05 # longitudental dispersivity
    trpt = 0.3 # ratio of horizontal transverse dispersivity to longitudenal dispersivity
    trpv = 0.3 # ratio of vertical transverse dispersivity to longitudenal dispersivity

# =============================================================================
#     BOUNDARY AND INTIAL CONDITIONS
# =============================================================================
    # backpressure, give this in kPa for conversion
    bp_kpa = 70
    # Injection rate in defined units
    injection_rate = 4 # [cm^3/min]
    # Core radius
    core_radius = 2.54 # [cm]
    # Calculation of core area
    core_area = 3.1415*core_radius**2
    # Calculation of mask area
    mask_area = np.sum(core_mask)*grid_size[0]*grid_size[1]
    # total specific discharge or injection flux (rate/area)
    # q = injection_rate/core_area # [mL/min]
    q = injection_rate*(mask_area/core_area)/np.sum(core_mask)

    # number of stress periods (MF input), calculated from period length input
    nper = len(perlen_mf)
    # Initial concentration (MT input)
    c0 = 1.
    # Stress period 2 concentration
    c1 = 0.0
    # period length in selected units (provided as input, but could be defined here)
    # perlen_mf = [1., 50]

    # MODFLOW head boundary conditions, <0 = specified head, 0 = no flow, >0 variable head
    # ibound = np.ones((nlay, nrow, ncol), dtype=np.int)
    ibound = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)

    # inlet conditions (currently set with well so inlet is zero)
    # ibound[:, :, 0] = -1
    # outlet conditions
    # ibound[5:15, 5:15, -1] = -1
    ibound[:, :, -1] = ibound[:, :, -1]*-1

    # MODFLOW constant initial head conditions
    strt = np.zeros((nlay, nrow, ncol), dtype=np.float)
    # convert backpressure to head units
    if lenuni == 3: # centimeters
        hout = bp_kpa*1000/(1000*9.81)*100
    else: # elseif meters
        if lenuni == 2:
            hout = bp_kpa*1000/(1000*9.81)
    # assign outlet pressure as head converted from 'bp_kpa' input variable
    # strt[:, :, -1] = core_mask*hout
    strt[:, :, -1] = hout

    # Stress period well data for MODFLOW. Each well is defined through defintition
    # of layer (int), row (int), column (int), flux (float). The first number corresponds to the stress period
    # Example for 1 stress period: spd_mf = {0:[[0, 0, 1, q],[0, 5, 1, q]]}
    well_info = np.zeros((int(np.sum(core_mask)), 4), dtype=np.float)
    # Nested loop to define every inlet face grid cell as a well
    index_n = 0
    for layer in range(0, nlay):
        for row in range(0, nrow):
            # index_n = layer*nrow + row
            # index_n +=1
            # print(index_n)
            if core_mask[layer, row] > 0:
                well_info[index_n] = [layer, row, 0, q]
                index_n +=1

    # Now insert well information into stress period data
    # *** TO DO: Generalize this for multiple stress periods
    # This has the form: spd_mf = {0:[[0, 0, 0, q],[0, 5, 1, q]], 1:[[0, 1, 1, q]]}
    spd_mf={0:well_info}

    # MT3D stress period data, note that the indices between 'spd_mt' must exist in 'spd_mf'
    # This is used as input for the source and sink mixing package
    # Itype is an integer indicating the type of point source, 2=well, 3=drain, -1=constant concentration
    itype = 2
    cwell_info = np.zeros((int(np.sum(core_mask)), 5), dtype=np.float)
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
    # Now apply stress period info
    spd_mt = {0:cwell_info, 1:cwell_info2}

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
    timprs = np.linspace(0, np.sum(perlen_mf), nprs, endpoint=False)
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
    btn = flopy.mt3d.Mt3dBtn(mt, ncomp=1, icbund=icbund, prsity=prsity, sconc=0,
                             tunit=mt_tunit, lunit=mt_lunit, nprs=nprs, timprs=timprs)

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

    # set inactive cell pressures to zero, by default inactive cells have a pressure of -999
    heads[heads < -990] = 0

    # convert heads to pascals
    if lenuni == 3: # centimeters
        pressures = heads/100*(1000*9.81)
    else: # elseif meters
        if lenuni == 2:
            pressures = heads*(1000*9.81)


    # crop off extra concentration slices
    conc = conc[:,:,:,1:-1]

    # calculate pressure drop
    center = round(nrow/2)
    dp = np.mean(pressures[center-3:center+2, center-3:center+2, 0]) - np.mean(pressures[center-3:center+2, center-3:center+2, -1])
    # print('Pressure drop: '+ str(dp/1000) + ' kPa')

    # crop off extra pressure slices
    pressures = pressures[:,:,1:-1]

    # Print final run time
    end_time = time.time() # end timer
    # print('Model run time: ', end - start) # show run time
    print(f"Model run time: {end_time - start:0.4f} seconds")

    # calculate mean permeability from pressure drop
    # water viscosity
    mu_water = 0.00089 # Pa.s
    L = (hk_size[2]*delc)
    km2_mean = (q*np.sum(core_mask)/mask_area)*L*mu_water/dp /(60*100**2)

    return mf, mt, conc, timearray, km2_mean


def quantile_calc(btc_1d, timearray, quantile, t_increment):
    # find length of time array
    ntime = timearray.shape[0]
    # calculate zero moment
    M0 = np.trapz(btc_1d, timearray)
    # reset incremental quantile numerator tracker
    M0im = 0

    for i in range(1, ntime):
        # numerically integrate from the beginning to the end of the dataset
        M0i = np.trapz(btc_1d[:i], timearray[:i])
        # check to see if the quantile has been surpassed, if so then linearly
        # interpolate between the current measurement and the previous measurement
        if M0i/M0 > quantile:
            # recalculate the integral of the previous measurement (i-1)
            M0im = np.trapz(btc_1d[:i-1], timearray[:i-1])
            # linear interpolation
            m = (btc_1d[i-1] - btc_1d[i-2])/(timearray[i-1] - timearray[i-2])
            b = btc_1d[i-2] - m*timearray[i-2]
            # now search through the space between the two measurements to find
            # when the area under the curve is equal to the desired quantile
            for xt in np.arange(timearray[i-2], timearray[i-1] +t_increment, t_increment):
                # calculate the linearly interpolated area under the curve
                M0int = M0im + np.trapz([btc_1d[i-2], m*xt+b], [timearray[i-2], xt])

                if M0int/M0 > quantile:
                    tau = xt
                    break # the inside FOR loop

            break # the outside FOR loom
        else:
            tau = 0

    return tau


# Function to calculate the quantile arrival time map
def flopy_arrival_map_function(conc, timearray, grid_size, quantile, t_increment):
    # start timer
    tic = time.perf_counter()

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
    tau_in = quantile_calc(oned[:,0], timearray, quantile, t_increment/10)

    # arrival time calculation in outlet slice
    tau_out = quantile_calc(oned[:,-1], timearray, quantile, t_increment/10)

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
                        tau_vox = quantile_calc(cell_btc, timearray, quantile, t_increment)
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

    if tau_vox == 0:
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

    # stop timer
    toc = time.perf_counter()
    print(f"Arrival time function runtime: {toc - tic:0.4f} seconds")
    return at_array, at_array_norm, at_diff_norm
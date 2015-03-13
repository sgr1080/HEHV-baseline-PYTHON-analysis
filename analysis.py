import scipy
import matplotlib.pyplot as plt
from scipy import integrate
import numpy
import glob
from IPython.display import HTML
from scipy import stats
    
def get_data(filename,numcells):
    # manipulate ASCII files for easier processing
    for k in range(1,numcells):  
	infile = open(glob.glob(filename % (k))[0]).read()
	outfile = open(glob.glob(filename % (k))[0], 'w')
	replacements = {'C':'1', 'D':'-1', 'R':'0', 'O':'0','S':'0', 'P':'0'}
	for j in replacements.keys():
	    infile = infile.replace(j, replacements[j])
	outfile.write(infile)
	outfile.close()
	
    # data processing
    trial = {}

    for i in range(1,numcells+1):
	data = numpy.loadtxt(glob.glob(filename % (i))[0], skiprows = 2, usecols=(1,3,5,7,8,9))
	trial['trial%d' % i] = {}
	fin_cyc = numpy.amax(data[:,0]).astype(int)
	for x in range(1,fin_cyc):   
	    cyc_num = x
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}
	    trial['trial%d' % i]['cycnum%d' % x] = {}

	    # Extract raw data for the cycle 
	    cyc_cat = data[data[:,0]==cyc_num]
	    cyc_cat_crg = cyc_cat[cyc_cat[:,5]==1]
	    cyc_cat_drg = cyc_cat[cyc_cat[:,5]==-1]

	    # Extract time data in seconds.
	    time_cyc = (cyc_cat[:,1]-numpy.amin(cyc_cat[:,1]))*60
	    time_cyc_crg = (cyc_cat_crg[:,1]-numpy.amin(cyc_cat_crg[:,1]))*60
	    time_cyc_drg = (cyc_cat_drg[:,1]-numpy.amin(cyc_cat_drg[:,1]))*60

	    # Extract current data in A
	    crnt_cyc = cyc_cat[:,3]*cyc_cat[:,5]
	    crnt_cyc_crg = cyc_cat_crg[:,3]
	    crnt_cyc_drg = cyc_cat_drg[:,3]

	    # Calculate the cummulative capacity in units C. 
	    chrg_cyc = scipy.integrate.cumtrapz(crnt_cyc,x=time_cyc,initial=0)
	    chrg_cyc_crg = scipy.integrate.cumtrapz(crnt_cyc_crg,x=time_cyc_crg,initial=0)
	    chrg_cyc_drg = scipy.integrate.cumtrapz(crnt_cyc_drg,x=time_cyc_drg,initial=0)
	    tchrg_cyc_crg = numpy.trapz(crnt_cyc_crg,x=time_cyc_crg)
	    tchrg_cyc_drg = numpy.trapz(crnt_cyc_drg,x=time_cyc_drg)

	    # Extract voltage in V
	    vltg_cyc = cyc_cat[:,4]
	    vltg_cyc_crg = cyc_cat_crg[:,4]
	    avg_vltg_crg = numpy.trapz(vltg_cyc_crg,x=chrg_cyc_crg)/tchrg_cyc_crg
	    vltg_cyc_drg = cyc_cat_drg[:,4]
	    avg_vltg_drg = numpy.trapz(vltg_cyc_drg,x=chrg_cyc_drg)/tchrg_cyc_drg

	    # Calculate the capacitance in units F
	    numer = numpy.gradient(chrg_cyc)
	    denom = numpy.gradient(cyc_cat[:,4])
	    cpct_cyc = (numer/denom)*cyc_cat[:,5]
	    dvdq_cyc = (denom/numer)*cyc_cat[:,5]

	    numer_crg = numpy.gradient(chrg_cyc_crg)
	    denom_crg = numpy.gradient(cyc_cat_crg[:,4])
	    cpct_crg = (numer_crg/denom_crg)*cyc_cat_crg[:,5]

	    numer_drg = numpy.gradient(chrg_cyc_drg)
	    denom_drg = numpy.gradient(cyc_cat_drg[:,4])
	    cpct_drg = (numer_drg/denom_drg)*cyc_cat_drg[:,5]

	    # Calculate the charge/discharge energy in J
	    enrg_crg = numpy.trapz(vltg_cyc_crg*crnt_cyc_crg,x=time_cyc_crg)
	    enrg_drg = numpy.trapz(vltg_cyc_drg*crnt_cyc_drg,x=time_cyc_drg)

	    trial['trial%d' % i]['cycnum%d' % x]['cyc_num'] = cyc_num
	    trial['trial%d' % i]['cycnum%d' % x]['tt_crg_cap'] = tchrg_cyc_crg
	    trial['trial%d' % i]['cycnum%d' % x]['tt_drg_cap'] = tchrg_cyc_drg
	    trial['trial%d' % i]['cycnum%d' % x]['c_efficiency'] = tchrg_cyc_drg/tchrg_cyc_crg
	    trial['trial%d' % i]['cycnum%d' % x]['avg_vltg_crg'] = avg_vltg_crg
	    trial['trial%d' % i]['cycnum%d' % x]['avg_vltg_drg'] = avg_vltg_drg
	    trial['trial%d' % i]['cycnum%d' % x]['v_efficiency'] = avg_vltg_drg/avg_vltg_crg
	    trial['trial%d' % i]['cycnum%d' % x]['avg_enrg_crg'] = enrg_crg
	    trial['trial%d' % i]['cycnum%d' % x]['avg_enrg_drg'] = enrg_drg
	    trial['trial%d' % i]['cycnum%d' % x]['e_efficiency'] = enrg_drg/enrg_crg		
    return trial
  
def plot_raw_basemetrics(data,cyc_range,trial_range,colour):  
    num_plots = 3
    fig_width_pt = num_plots*350.0 
    inches_per_pt = 1.0/72.27              
    golden_mean = (numpy.sqrt(5)-1.0)/2.0         
    fig_width = fig_width_pt*inches_per_pt  
    fig_height = fig_width*golden_mean      
    fig_size =  [fig_width,fig_height]
    params = {'backend': 'GTKAgg',
	      'axes.labelsize': 15,
	      'text.fontsize': 15,
	      'legend.fontsize': 10,
	      'xtick.labelsize': 15,
	      'ytick.labelsize': 15,
	      'text.usetex': True,
	      'ps.useafm' : False,
	      'figure.figsize': fig_size}
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rcParams.update(params)
    fig = plt.figure()
    plt.subplots_adjust(left=None, bottom=None,right=None, top=None, 
			wspace=0.40, hspace=0.53)

    clr = colour
    alp = 1.0
    lwd = 0.0
    mks = 8
    lbpd = 10
    
    eg_scale = 0.000277777778
    cp_scale = 0.277777778
    
    host = fig.add_subplot(3,3,1)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['tt_crg_cap'])*cp_scale,
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'capacity (mA h)',labelpad=lbpd)
    host.text(0.1,1.05,'charge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(1.0,3.2)

    host = fig.add_subplot(3,3,2)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['tt_drg_cap'])*cp_scale,
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'capacity (mA h)',labelpad=lbpd)
    host.text(0.1,1.05,'discharge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(1.0,3.2)

    host = fig.add_subplot(3,3,3)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['c_efficiency']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'coulombic efficiency',labelpad=lbpd)
    host.locator_params(nbins=5)
    host.set_ylim(0.950,1.00)

    host = fig.add_subplot(3,3,4)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['avg_vltg_crg']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'average voltage (V)',labelpad=lbpd)
    host.text(0.1,1.05,'charge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(3.2,4.2)

    host = fig.add_subplot(3,3,5)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['avg_vltg_drg']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'average voltage (V)',labelpad=lbpd)
    host.text(0.1,1.05,'discharge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(3.2,4.2)
    
    host = fig.add_subplot(3,3,6)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['v_efficiency']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'voltage efficiency',labelpad=lbpd)
    host.locator_params(nbins=5)
    host.set_ylim(0.85,1.00)
    
    
    host = fig.add_subplot(3,3,7)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['avg_enrg_crg'])*eg_scale,
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'energy (W h)',labelpad=lbpd)
    host.text(0.1,1.05,'charge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(0.002,0.015)

    host = fig.add_subplot(3,3,8)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['avg_enrg_drg'])*eg_scale,
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'energy (W h)',labelpad=lbpd)
    host.text(0.1,1.05,'discharge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(0.002,0.015)

    host = fig.add_subplot(3,3,9)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['e_efficiency']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'energy efficiency',labelpad=lbpd)
    host.locator_params(nbins=5)
    host.set_ylim(0.85,1.00)
 

def plot_mscaled_basemetrics(data,cyc_range,trial_range,colour,avg_mass):  
    num_plots = 3
    fig_width_pt = num_plots*350.0 
    inches_per_pt = 1.0/72.27              
    golden_mean = (numpy.sqrt(5)-1.0)/2.0         
    fig_width = fig_width_pt*inches_per_pt  
    fig_height = fig_width*golden_mean      
    fig_size =  [fig_width,fig_height]
    params = {'backend': 'GTKAgg',
	      'axes.labelsize': 15,
	      'text.fontsize': 15,
	      'legend.fontsize': 10,
	      'xtick.labelsize': 15,
	      'ytick.labelsize': 15,
	      'text.usetex': True,
	      'ps.useafm' : False,
	      'figure.figsize': fig_size}
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rcParams.update(params)
    fig = plt.figure()
    plt.subplots_adjust(left=None, bottom=None,right=None, top=None, 
			wspace=0.40, hspace=0.53)

    clr = colour
    alp = 1.0
    lwd = 0.0
    mks = 8
    lbpd = 10
    
    eg_mass = avg_mass*1E-6
    cp_mass = avg_mass*0.001
    
    eg_scale = 0.000277777778/(eg_mass)
    cp_scale = 0.277777778/(cp_mass)
    
    host = fig.add_subplot(3,3,1)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['tt_crg_cap'])*cp_scale,
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'capacity (mA h g$^{-1}$)',labelpad=lbpd)
    host.text(0.1,1.05,'charge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(1.0/cp_mass,3.2/cp_mass)

    host = fig.add_subplot(3,3,2)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['tt_drg_cap'])*cp_scale,
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'capacity (mA h g$^{-1}$)',labelpad=lbpd)
    host.text(0.1,1.05,'discharge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(1.0/cp_mass,3.2/cp_mass)

    host = fig.add_subplot(3,3,3)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['c_efficiency']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'coulombic efficiency',labelpad=lbpd)
    host.locator_params(nbins=5)
    host.set_ylim(0.950,1.00)

    host = fig.add_subplot(3,3,4)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['avg_vltg_crg']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'average voltage (V)',labelpad=lbpd)
    host.text(0.1,1.05,'charge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(3.2,4.2)

    host = fig.add_subplot(3,3,5)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['avg_vltg_drg']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'average voltage (V)',labelpad=lbpd)
    host.text(0.1,1.05,'discharge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(3.2,4.2)
    
    host = fig.add_subplot(3,3,6)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['v_efficiency']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'voltage efficiency',labelpad=lbpd)
    host.locator_params(nbins=5)
    host.set_ylim(0.85,1.00)
    
    
    host = fig.add_subplot(3,3,7)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['avg_enrg_crg'])*eg_scale,
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'energy (W h)',labelpad=lbpd)
    host.text(0.1,1.05,'charge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(0.002/eg_mass,0.015/eg_mass)

    host = fig.add_subplot(3,3,8)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['avg_enrg_drg'])*eg_scale,
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'energy (W h)',labelpad=lbpd)
    host.text(0.1,1.05,'discharge',transform=host.transAxes,fontsize=15)
    host.locator_params(nbins=5)
    host.set_ylim(0.002/eg_mass,0.015/eg_mass)

    host = fig.add_subplot(3,3,9)
    for j in trial_range:
	for i in cyc_range:
	    host.plot(float(data['trial%d' % j]['cycnum%d' % i]['cyc_num']),
		      float(data['trial%d' % j]['cycnum%d' % i]['e_efficiency']),
		      lw=lwd, mew=0, ms=mks, marker='o', color=clr, alpha=alp)
    host.set_xlabel(r'cycle number')
    host.set_ylabel(r'energy efficiency',labelpad=lbpd)
    host.locator_params(nbins=5)
    host.set_ylim(0.85,1.00)

def plot_singlefig_setup():
    num_plots = 3
    fig_width_pt = num_plots*350.0 
    inches_per_pt = 1.0/72.27              
    golden_mean = (numpy.sqrt(5)-1.0)/2.0         
    fig_width = fig_width_pt*inches_per_pt  
    fig_height = fig_width*golden_mean      
    fig_size =  [fig_width,fig_height]
    params = {'backend': 'GTKAgg',
	      'axes.labelsize': 15,
	      'text.fontsize': 15,
	      'legend.fontsize': 10,
	      'xtick.labelsize': 15,
	      'ytick.labelsize': 15,
	      'text.usetex': True,
	      'ps.useafm' : False,
	      'figure.figsize': fig_size}
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rcParams.update(params)

    clr = '#e9a3c9'
    alp = 1.0
    lwd = 0.0
    mks = 8
    
    fig = plt.figure()
    host = fig.add_subplot(3,3,1)

    return fig,host    
    
    





  
# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created:      Nov. 15th, 2016                                               #
# Description:  We analyze the output of the WavMixer.                        #
#               We compute the number of photons generated in different       #
#               experimental configurations:                                  #
#                - On-axis parabolic mirrors (HNA)                            #
#                - On-axis parabolic mirrors with hole (HNA-h)                #
#                - Off-axis parabolic mirrors (OFF)                           #
#                - Transmission parabolic mirrors (TRA)                       #
#                - Transmission parabolic mirrors with hole (TRA-h)           #
# Dependencies: - NumPy                                                       #
#               - SciPy                                                       #
#               - H5Py                                                        #
#               - matplotlib                                                  #
# --------------------------------------------------------------------------- #

# --------------------------- Modules Importation --------------------------- #
import numpy as np
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.signal as signal
import scipy.integrate as integration
import scipy.interpolate as interp
import argparse
import h5py
import time
import math
import configparser
from mpl_toolkits.axes_grid1 import make_axes_locatable
import vphys

# ------------------------------ Configuration ------------------------------ #

pgf_with_pdflatex = {
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
      r"\usepackage{amsmath}",
      r"\usepackage{siunitx}",
      #r"\usepackage{mathspec}",
      r"\usepackage[charter]{mathdesign}",
      r"\usepackage{fontspec}",
      #r"\setmathfont{Fira Sans}",
      r"\setmainfont{Oswald}",
      ]
}
matplotlib.rcParams.update(pgf_with_pdflatex)

# -- Fonts
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['font.family'] = 'serif'

# -- Plots
#matplotlib.rcParams['axes.labelsize'] = 'large'
#matplotlib.rcParams['xtick.labelsize'] = 'large'
#matplotlib.rcParams['ytick.labelsize'] = 'large'
#matplotlib.rcParams['legend.numpoints'] = 5
#matplotlib.rcParams['figure.figsize'] = '4,2'
matplotlib.rcParams['axes.grid'] = True

# -------------------------------- Functions  ------------------------------- #
def _infunc(x,func,gfun,hfun,more_args):
    a = gfun(x)
    b = hfun(x)
    myargs = (x,) + more_args
    return integration.quad(func,a,b,args=myargs)[0]

def custom_dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8, maxp1=100, limit=100):
    return integration.quad(_infunc, a, b, (func, gfun, hfun, args),epsabs=epsabs, epsrel=epsrel, maxp1=maxp1, limit=limit)

def fmt(x, pos):
    a, b = '{:1.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \times\, 10^{{{}}}$'.format(a, b)

# -------------------------------- Constants -------------------------------- #
UNIT_MASS      = 9.109382914e-31
UNIT_LENGTH    = 3.86159e-13
UNIT_TIME      = 1.2880885e-21
SPEED_OF_LIGHT = 299792458
EPSILON_0      = 8.85418782e-12
MU_0           = 4*np.pi*1.0e-7
ALPHA          = 1.0/137.035999074
UNIT_E_FIELD   = 1.3e18*np.sqrt(4*np.pi*ALPHA)
UNIT_B_FIELD   = UNIT_E_FIELD/SPEED_OF_LIGHT

# -------------------- Analysis of the Number of Photons -------------------- #
# -- We analyze the number of photons generated in a given geometry.       -- #
# --------------------------------------------------------------------------- #

# -- We parse the arguments.
parser = argparse.ArgumentParser()
parser.add_argument("min",        type=int,          help="Minimum index of simulation to analyze.")
parser.add_argument("max",        type=int,          help="Maximum index of simulation to analyze.")
parser.add_argument("dim",        type=int,          help='Dimension of the focal region.')
parser.add_argument("--geometry", dest='geom',       help="Geometry under consideration.")
parser.add_argument("--prefix",   dest='prefix',     help="Folder prefix.")
parser.add_argument("--config",   dest='configFile', help="INI file containing the parameters of the simualtion.")
args = parser.parse_args()

# We analyze the simulation in between min and max.
simu_dir = args.prefix+"_{0:05}.BQ".format(1)+"/../"

# -- Global analysis.
n_photons_file = open(simu_dir+args.geom+"_data.txt", 'w')
max_angle_file = open(simu_dir+args.geom+"_max_angle.txt", 'w')

# We determine if we analyze the shadow.
analyze_shadow_bool = (args.geom=="hna-h-artifical" or args.geom=="tra-h" or args.geom=="hna-h" or args.geom == "off-axis-hole")

if (analyze_shadow_bool):
  n_photons_shadow_file = open(simu_dir+args.geom+"_shadow_data.txt", 'w+')

for i in range(args.min,args.max+1):

  # -- We open the files.
  simu_prefix               = args.prefix+"_{0:05d}.BQ/{0:05d}.BQ/".format(i)
  try:
    n_photons_first_file    = h5py.File(simu_prefix+"number_of_photons_first_harmonic.hdf5", 'r')
    spatial_dist_first_file = h5py.File(simu_prefix+"spatial_dist_first_harmonic.hdf5", 'r')

    n_photons_third_file    = h5py.File(simu_prefix+"number_of_photons_third_harmonic.hdf5", 'r')
    spatial_dist_third_file = h5py.File(simu_prefix+"spatial_dist_third_harmonic.hdf5", 'r')

    config = configparser.ConfigParser(inline_comment_prefixes=";")
    config.read(simu_prefix+"/"+args.configFile)

  except:
    continue

  focal_length     = float(config['Parabola']['focal_length'])
  rmax             = float(config['Parabola']['r_max'])

  # -- We plot the total spectrum of photons for both harmonics.
  n_photons_first   = n_photons_first_file['/spectrum/Number of photons'][:]
  wavelengths_first = n_photons_first_file['/spectrum/wavelength (m)'][:]
  freqs_first       = n_photons_first_file['/spectrum/frequency (Hz)'][:]

  n_photons_third   = n_photons_third_file['/spectrum/Number of photons'][:]
  wavelengths_third = n_photons_third_file['/spectrum/wavelength (m)'][:]
  freqs_third       = n_photons_third_file['/spectrum/frequency (Hz)'][:]

  phi_first         = spatial_dist_first_file['/coordinates/phi'][:]
  phi_first_deg     = np.degrees(phi_first)

  if args.dim == 3:
    theta_first       = spatial_dist_first_file['/coordinates/theta'][:]
    theta_first_deg   = np.degrees(theta_first)

  # -- Support older versions of the WaveMixer.
  try:
    n_density_first   = spatial_dist_first_file['/field/Component0'][:]

  except:
    n_density_first   = spatial_dist_first_file['/field/ScalarField'][:]

  phi_third         = spatial_dist_third_file['/coordinates/phi'][:]
  phi_third_deg     = np.degrees(phi_third)
  if args.dim == 3:
    theta_third       = spatial_dist_third_file['/coordinates/theta'][:]
    theta_third_deg   = np.degrees(theta_third)

  # -- Support older versions of the WaveMixer.
  try:
    n_density_third   = spatial_dist_third_file['/field/Component0'][:]

  except:
    n_density_third   = spatial_dist_third_file['/field/ScalarField'][:]

  # -- Determine the phi at which the emission is maximum.
  max_idx_f = np.argmax(n_density_first)

  if args.dim == 3:
    max_phi = phi_first_deg[np.unravel_index(max_idx_f, n_density_first.shape)[0]]
  else:
    max_phi = phi_first_deg[max_idx_f]

  # Create the figures.
  plot_options = {"rasterized": True, "shading": "interp", "cmap":"magma"}
  n_photons_fig = plt.figure(figsize=(6,4))

  n_photons_f_spec_ax = n_photons_fig.add_subplot(221)
  plt.plot(wavelengths_first/1.0e-9,n_photons_first)
  plt.xlabel("Wavelength (nm)")
  plt.ylabel("Number of photons")
  plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

  n_photons_t_spec_ax = n_photons_fig.add_subplot(222)
  plt.plot(wavelengths_third/1.0e-9,n_photons_third)
  plt.xlabel("Wavelength (nm)")
  plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

  photon_density_f_ax = n_photons_fig.add_subplot(223)
  if args.dim == 2:
    plt.plot(phi_first_deg,n_density_first)
    plt.xlabel('$\\phi$ (degrees)')
    plt.ylabel('Photon density')

  if args.dim == 3:
    im=photon_density_f_ax.pcolormesh(theta_first_deg,phi_first_deg,n_density_first,**plot_options)
    photon_density_f_ax.axis([0.0,360.0,0.0,180.0])
    photon_density_f_ax.set_aspect('equal')
    photon_density_f_ax.set_xlabel('$\\theta$ (degrees)')
    photon_density_f_ax.set_ylabel('$\\phi$ (degrees)')
    photon_density_f_ax.set_xticks(np.arange(0,365,45))
    photon_density_f_ax.set_yticks(np.arange(0,185,30))
    cbar=plt.colorbar(im,shrink=0.62,ax=photon_density_f_ax)
    cbar.formatter.set_powerlimits((0,0))
    cbar.update_ticks()

  photon_density_t_ax = n_photons_fig.add_subplot(224)
  if args.dim == 2:
    plt.plot(phi_third_deg,n_density_third)
    plt.xlabel('$\\phi$ (degrees)')
    plt.ylabel('Photon density')

  if args.dim ==3:
    plt.pcolormesh(theta_third_deg,phi_third_deg,n_density_third,**plot_options)
    plt.axis([0.0,360.0,0.0,180.0])
    plt.gca().set_aspect('equal')
    plt.xlabel('$\\theta$ (degrees)')
    plt.xticks(np.arange(0,365,45))
    plt.yticks(np.arange(0,185,30))
    cbar=plt.colorbar(shrink=0.62)
    cbar.formatter.set_powerlimits((0,0))
    cbar.update_ticks()
    #plt.ylabel('$\\phi$ (degrees)')

  plt.tight_layout()
  plt.savefig(simu_prefix+"n_photons.pdf", dpi=500)
  plt.close()

  # -- THESIS READY PLOTS
  if args.dim == 3:
    figDensityFirst = plt.figure(num="figFirst",figsize=(4,3))
    ax = figDensityFirst.add_subplot(111)
    im=ax.pcolormesh(theta_first_deg,phi_first_deg,n_density_first,**plot_options)
    ax.axis([0.0,360.0,0.0,180.0])
    ax.set_aspect('equal')
    ax.set_xlabel('$\\theta$ (degrees)')
    ax.set_ylabel('$\\phi$ (degrees)', rotation='horizontal', ha='left')
    ax.yaxis.set_label_coords(-0.1, 1.1, transform=ax.transAxes)
    ax.set_xticks(np.arange(0,365,45))
    ax.set_yticks(np.arange(0,185,30))

    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.1)
    cbar    = plt.colorbar(im, cax=cax)
    cbar.formatter.set_powerlimits((0,0))
    cbar.update_ticks()

    plt.savefig(simu_prefix+"photon_density_f.pdf", bbox_inches='tight', dpi=500)

    figDensityThird = plt.figure(num="figThird",figsize=(4,3))
    ax = figDensityThird.add_subplot(111)
    im=ax.pcolormesh(theta_third_deg,phi_third_deg,n_density_third,**plot_options)
    ax.axis([0.0,360.0,0.0,180.0])
    ax.set_aspect('equal')
    ax.set_xlabel('$\\theta$ (degrees)')
    ax.set_ylabel('$\\phi$ (degrees)', rotation='horizontal', ha='left')
    ax.yaxis.set_label_coords(-0.1, 1.1, transform=ax.transAxes)
    ax.set_xticks(np.arange(0,365,45))
    ax.set_yticks(np.arange(0,185,30))

    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.1)
    cbar    = plt.colorbar(im, cax=cax)
    cbar.formatter.set_powerlimits((0,0))
    cbar.update_ticks()

    plt.savefig(simu_prefix+"photon_density_t.pdf", bbox_inches='tight', dpi=500)

    # -- Contours of density plots.
    contour_plot_options = {"linestyles": '--', "colors": 'k', 'linewidths': 0.5}
    figDensityFirst = plt.figure(num="figFirst")
    ax = figDensityFirst.get_axes()[0]
    ax.contour(theta_first_deg,phi_first_deg,n_density_first,**contour_plot_options)
    plt.savefig(simu_prefix+"photon_density_f_mod.pdf", bbox_inches='tight', dpi=500)

    figDensityThird = plt.figure(num="figThird")
    ax = figDensityThird.get_axes()[0]
    ax.contour(theta_third_deg,phi_third_deg,n_density_third,**contour_plot_options)
    plt.savefig(simu_prefix+"photon_density_t_mod.pdf", bbox_inches='tight', dpi=500)

  print("------------- i = {} -----------------------".format(i))

  # -- Write the number of photons.
  print("The total number of photons is \n(1st harmonic): {} (3rd harmonic): {}".format(sum(n_photons_first),sum(n_photons_third)))
  n_photons_file.write("{}\t{}\t{}".format(2*focal_length/rmax,sum(n_photons_first),sum(n_photons_third)))
  n_photons_file.write("\n")

  # -- Write the angles.
  max_angle_file.write("{}\t{}".format(2*focal_length/rmax,max_phi))
  max_angle_file.write("\n")

  # -- We now plot the detectable number of photons (in the shadow).
  if (analyze_shadow_bool):

    # For the HNA parabola, we compute the number of photons that
    # are emitted in the shadow of a hole burred in the deep region
    # of the parabola. Contrary to the transmission parabola, this results
    # in a forward facing shadow.

    # Real hole.
    # Actual hole in the parabola when computing the number of photons.
    if args.geom == "hna-h":

      r_hole    = float(config['Parabola']['r_min'])
      z_hole    = r_hole**2/(4.0*focal_length)-focal_length
      th_shadow = np.arctan2(r_hole,focal_length)

      # -- Arrays for manual integration.
      n_density_first_integrand = np.zeros_like(n_density_first)
      n_density_third_integrand = np.zeros_like(n_density_third)

      if args.dim == 3:
        # -- We prepare the integrands of the photon densities.
        for idx_i in range(len(theta_first)):
          for idx_j in range(len(phi_first)):
            n_density_first_integrand[idx_j][idx_i] = n_density_first[idx_j][idx_i]*np.sin(phi_first[idx_j])

        for idx_i in range(len(theta_third)):
          for idx_j in range(len(phi_third)):
            n_density_third_integrand[idx_j][idx_i] = n_density_third[idx_j][idx_i]*np.sin(phi_third[idx_j])

         # -- We interpolate the integrands.
        n_density_first_interp = interp.interp2d(theta_first,phi_first,n_density_first_integrand, kind='cubic')
        n_density_third_interp = interp.interp2d(theta_third,phi_third,n_density_third_integrand, kind='cubic')

        # -- We integrate in the shadow.
        n_photon_first_total  = custom_dblquad(n_density_first_interp,0.0, np.pi,     lambda x: 0.0, lambda x: 2.0*np.pi)
        n_photon_first_shadow = custom_dblquad(n_density_first_interp,0.0, th_shadow, lambda x: 0.0, lambda x: 2.0*np.pi)

        n_photon_third_total  = custom_dblquad(n_density_third_interp,0.0, np.pi,     lambda x: 0.0, lambda x: 2.0*np.pi)
        n_photon_third_shadow = custom_dblquad(n_density_third_interp,0.0, th_shadow, lambda x: 0.0, lambda x: 2.0*np.pi)

        # - We store the values in the file.
        n_photons_shadow_file.write("{}\t{}\t{}\t{}".format(2*focal_length/rmax,r_hole,n_photon_first_shadow[0],n_photon_third_shadow[0]))
        n_photons_shadow_file.write("\n")

    # Artifical hole.
    # No hole in the simulation, so no loss of energy. We can get an approximate
    # number of photons by scaling by the approximate energy loss a posteriori.
    if args.geom =="hna-h-artifical":
      r_hole = np.linspace(5.0e-3,15.0e-3,10)

      # -- Arrays for manual integration.
      n_density_first_integrand = np.zeros_like(n_density_first)
      n_density_third_integrand = np.zeros_like(n_density_third)

      if args.dim == 3:
        # -- We prepare the integrands of the photon densities.
        for idx_i in range(len(theta_first)):
          for idx_j in range(len(phi_first)):
            n_density_first_integrand[idx_j][idx_i] = n_density_first[idx_j][idx_i]*np.sin(phi_first[idx_j])

        for idx_i in range(len(theta_third)):
          for idx_j in range(len(phi_third)):
            n_density_third_integrand[idx_j][idx_i] = n_density_third[idx_j][idx_i]*np.sin(phi_third[idx_j])

         # -- We interpolate the integrands.
        n_density_first_interp = interp.interp2d(theta_first,phi_first,n_density_first_integrand, kind='cubic')
        n_density_third_interp = interp.interp2d(theta_third,phi_third,n_density_third_integrand, kind='cubic')

        # -- We open a file for the current value of the focal length.
        n_photons_hna_shadow_file = open(simu_prefix+"/"+args.geom+"_shadow.txt", 'w')

        for j in range(r_hole.size):
          z_hole    = r_hole[j]**2/(4.0*focal_length)-focal_length
          th_shadow = np.arctan2(r_hole[j],focal_length)

          # -- We integrate in the shadow.
          n_photon_first_total  = custom_dblquad(n_density_first_interp,0.0, np.pi,     lambda x: 0.0, lambda x: 2.0*np.pi)
          n_photon_first_shadow = custom_dblquad(n_density_first_interp,0.0, th_shadow, lambda x: 0.0, lambda x: 2.0*np.pi)

          n_photon_third_total  = custom_dblquad(n_density_third_interp,0.0, np.pi,     lambda x: 0.0, lambda x: 2.0*np.pi)
          n_photon_third_shadow = custom_dblquad(n_density_third_interp,0.0, th_shadow, lambda x: 0.0, lambda x: 2.0*np.pi)

          # - We store the values in the file.
          n_photons_hna_shadow_file.write("{}\t{}\t{}\t{}".format(2*focal_length/rmax,r_hole[j],n_photon_first_shadow[0],n_photon_third_shadow[0]))
          n_photons_hna_shadow_file.write("\n")

          # -- We print the values.
          #print("---------- r_hole = {} -----------".format(r_hole[j]))
          #print("Number of photons (total) :\n {} and {}".format(n_photon_first_total[0],n_photon_third_total[0]))
          #print("Number of photons (shadow):\n {} and {}".format(n_photon_first_shadow[0],n_photon_third_shadow[0]))

        n_photons_hna_shadow_file.close()

    # For the transmission parabola, we compute the number of photons
    # that are emitted in the shadow of the incident beam, plus an engineering
    # factor of 2 degrees.
    if args.geom=="tra-h":
      z_rmax         = np.abs(0.25*rmax**2/focal_length - focal_length)
      angle_shadow   = np.pi-np.arctan2(rmax,z_rmax) + np.radians(2.0)
      angle_hole_deg = 180-np.degrees(angle_shadow)

      # -- Arrays for manual integration.
      n_density_first_integrand = np.zeros_like(n_density_first)
      n_density_third_integrand = np.zeros_like(n_density_third)

      if args.dim == 2:
        # -- We prepare the integrands of the photon densities.
        for idx_i in range(len(phi_first)):
          n_density_first_integrand[idx_i] = n_density_first[idx_i]*np.sin(phi_first[idx_i])

        for idx_i in range(len(phi_third)):
          n_density_third_integrand[idx_i] = n_density_third[idx_i]*np.sin(phi_third[idx_i])

        # -- We interpolate the integrands.
        n_density_first_interp = interp.interp1d(phi_first,n_density_first_integrand, kind='cubic')
        n_density_third_interp = interp.interp1d(phi_third,n_density_third_integrand, kind='cubic')

        # -- We integrate in the shadow.
        n_photon_first_total  = integration.quad(n_density_first_interp,0.0,         np.pi)
        n_photon_first_shadow = integration.quad(n_density_first_interp,angle_shadow,np.pi)

        n_photon_third_total  = integration.quad(n_density_third_interp,0.0,         np.pi)
        n_photon_third_shadow = integration.quad(n_density_third_interp,angle_shadow,np.pi)

      if args.dim == 3:

        # -- We prepare the integrands of the photon densities.
        for idx_i in range(len(theta_first)):
          for idx_j in range(len(phi_first)):
            n_density_first_integrand[idx_j][idx_i] = n_density_first[idx_j][idx_i]*np.sin(phi_first[idx_j])

        for idx_i in range(len(theta_third)):
          for idx_j in range(len(phi_third)):
            n_density_third_integrand[idx_j][idx_i] = n_density_third[idx_j][idx_i]*np.sin(phi_third[idx_j])

        # -- We interpolate the integrands.
        n_density_first_interp = interp.interp2d(theta_first,phi_first,n_density_first_integrand, kind='cubic')
        n_density_third_interp = interp.interp2d(theta_third,phi_third,n_density_third_integrand, kind='cubic')

        # -- We integrate in the shadow.
        n_photon_first_total  = custom_dblquad(n_density_first_interp,0.0,          np.pi, lambda x: 0.0, lambda x: 2.0*np.pi)
        n_photon_first_shadow = custom_dblquad(n_density_first_interp,angle_shadow, np.pi, lambda x: 0.0, lambda x: 2.0*np.pi)

        n_photon_third_total  = custom_dblquad(n_density_third_interp,0.0,          np.pi, lambda x: 0.0, lambda x: 2.0*np.pi)
        n_photon_third_shadow = custom_dblquad(n_density_third_interp,angle_shadow, np.pi, lambda x: 0.0, lambda x: 2.0*np.pi)

      # -- We print and save the values.
      print("Number of photons (total) :\n {} and {}".format(n_photon_first_total[0],n_photon_third_total[0]))
      print("Number of photons (shadow):\n {} and {}".format(n_photon_first_shadow[0],n_photon_third_shadow[0]))

      n_photons_shadow_file.write("{}\t{}\t{}".format(2*focal_length/rmax,n_photon_first_shadow[0],n_photon_third_shadow[0]))
      n_photons_shadow_file.write("\n")

    # For an off-axis hole, we compute the position of the hole in cylindrical
    # coordinates, then compute the number of photons over that region.
    # THAT DOESN'T WORK LUL, I INTEGRATE IN THE HOLE, NOT IN THE SHADOW OF THE WHOLE.
    if args.geom == "off-axis-hole":
      # -- Arrays for manual integration.
      n_density_first_integrand = np.zeros_like(n_density_first)
      n_density_third_integrand = np.zeros_like(n_density_third)

      if args.dim == 3:

        # -- We compute the position of the hole.
        mask_x_pos  = float(config['Model']['mask_x_pos'])
        mask_y_pos  = float(config['Model']['mask_y_pos'])
        mask_radius = float(config['Model']['mask_radius'])

        mask_r_pos  = np.sqrt(mask_x_pos**2+mask_y_pos**2);
        mask_t_pos  = np.arctan2(mask_x_pos,mask_y_pos)

        def z_abs(r):
          z_abs       = np.abs(r**2/(4*focal_length)-focal_length)
          return z_abs

        mask_r_min  = mask_r_pos-mask_radius
        mask_r_max  = mask_r_pos+mask_radius
        phi_min     = np.arctan((mask_r_min)/z_abs(mask_r_min))
        phi_max     = np.arctan((mask_r_max)/z_abs(mask_r_max))

        # -- Fix this.
        if (2*focal_length < rmax):
          phi_min = np.pi - phi_min
          phiMax  = np.pi - phi_max

        def theta_min(phi):
          r       = np.abs(((1.0+np.sqrt(1.0+np.tan(phi)**2))/np.tan(phi)))*(2.0*focal_length)
          #print(r, mask_r_pos-mask_radius, mask_r_pos+mask_radius)
          sq_root = np.sqrt((mask_radius**2-(r-mask_r_pos)**2)/mask_r_pos**2)
          theta_min = mask_t_pos - np.arcsin(sq_root)
          return theta_min
          return mask_t_pos - np.arctan(mask_radius/mask_r_pos) + np.pi

        def theta_max(phi):
          r       = np.abs(((1.0+np.sqrt(1.0+np.tan(phi)**2))/np.tan(phi)))*(2.0*focal_length)
          sq_root = np.sqrt((mask_radius**2-(r-mask_r_pos)**2)/mask_r_pos**2)
          return mask_t_pos + np.arcsin(sq_root)
          return mask_t_pos + np.arctan(mask_radius/mask_r_pos) + np.pi

        # -- We prepare the integrands of the photon densities.
        for idx_i in range(len(theta_first)):
          for idx_j in range(len(phi_first)):
            n_density_first_integrand[idx_j][idx_i] = n_density_first[idx_j][idx_i]*np.sin(phi_first[idx_j])

        for idx_i in range(len(theta_third)):
          for idx_j in range(len(phi_third)):
            n_density_third_integrand[idx_j][idx_i] = n_density_third[idx_j][idx_i]*np.sin(phi_third[idx_j])

        # -- We interpolate the integrands.
        n_density_first_interp = interp.interp2d(theta_first,phi_first,n_density_first_integrand, kind='cubic')
        n_density_third_interp = interp.interp2d(theta_third,phi_third,n_density_third_integrand, kind='cubic')

        # -- We integrate in the shadow.
        n_photon_first_total  = custom_dblquad(n_density_first_interp,0.0,     np.pi,   lambda x: 0.0, lambda x: 2.0*np.pi)
        n_photon_first_shadow = custom_dblquad(n_density_first_interp,phi_min, phi_max, theta_min,     theta_max)

        n_photon_third_total  = custom_dblquad(n_density_third_interp,0.0,     np.pi,   lambda x: 0.0, lambda x: 2.0*np.pi)
        n_photon_third_shadow = custom_dblquad(n_density_third_interp,phi_min, phi_max, theta_min, theta_max)
        randomwhatever = z_abs(0.0)

      # -- We print and save the values.
      print("Number of photons (total) :\n {} and {}".format(n_photon_first_total[0],n_photon_third_total[0]))
      print("Number of photons (shadow):\n {} and {}".format(n_photon_first_shadow[0],n_photon_third_shadow[0]))


    print("-------------------------------------------------------------------")

  plt.close("figFirst")
  plt.close("figThird")

  # -- We close the files.
  n_photons_first_file.close()
  spatial_dist_first_file.close()
  n_photons_third_file.close()
  spatial_dist_third_file.close()

# -- We close the data file.
max_angle_file.close()
if (analyze_shadow_bool):
  n_photons_shadow_file.close()
n_photons_file.close()

# -- Draw some additional figures.
max_phi_data = np.loadtxt(simu_dir+args.geom+"_max_angle.txt")
print(max_phi_data[:,0])
figMaxPhi = plt.figure(figsize=(4,3))
axMaxPhi  = figMaxPhi.add_subplot(111)
axMaxPhi.plot(max_phi_data[:,0], max_phi_data[:,1])
plt.savefig(simu_dir+"max_angle.pdf", bbox_inches='tight', dpi=500)

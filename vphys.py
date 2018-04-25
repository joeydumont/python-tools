# ------------------------------- Information ------------------------------- #
# Author:       Joey Dumont                    <joey.dumont@gmail.com>        #
# Created       Mar. 1st, 2018                                                #
# Description:  Misc configuration for NumPy, SciPy and matplotlib.           #
# Dependencies: - NumPy                                                       #
#               - Scipy                                                       #
#               - Matplotlib                                                  #
# --------------------------------------------------------------------------- #

# ----------------------------- Misc. Functions ----------------------------- #
def mkdir_p(mypath):
  """
  Creates a directory. equivalent to using mkdir -p on the command line
  """

  from errno import EEXIST
  from os import makedirs,path

  try:
      makedirs(mypath)
  except OSError as exc: # Python >2.5
      if exc.errno == EEXIST and path.isdir(mypath):
          pass
      else: raise


# --------------------------- matplotlib Functions -------------------------- #
def adjust_spines(ax, spines, points_outward=10):
  """
  Helps in re-creating the spartan style of Jean-luc Doumont's graphs.

  Removes the spines that are not specified in spines, and colours the specified
  ones in gray, and pushes them outside the graph area.
  """
  for loc, spine in ax.spines.items():
      if loc in spines:
          spine.set_position(('outward', points_outward))  # outward by 10 points
          #spine.set_smart_bounds(True)
          spine.set_color('gray')
      else:
          spine.set_color('none')  # don't draw spine

  # turn off ticks where there is no spine
  if 'left' in spines:
      ax.yaxis.set_ticks_position('left')
  else:
      # no yaxis ticks
      ax.yaxis.set_ticks([])

  if 'bottom' in spines:
      ax.xaxis.set_ticks_position('bottom')
  else:
      # no xaxis ticks
      ax.xaxis.set_ticks([])


def default_pgf_configuration():
  """
  Defines a default configuration for the pgf engine of matplotlib, with
  LaTeX support.
  """
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

  return pgf_with_pdflatex

def BarPlotWithLogAxes(ax_handle,x,y,width, xdelta=0.0, **plot_kwargs):
    """
    This plots a bar graph with a log-scaled x axis by manually filling rectangles.
    We use the
    """
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(len(x)):
        artist, = ax_handle.fill([10**(np.log10(x[i])-width-xdelta), 10**(np.log10(x[i])-width-xdelta), 10**(np.log10(x[i])+width-xdelta),10**(np.log10(x[i])+width-xdelta)], [0, y[i], y[i], 0],**plot_kwargs)

    return artist

# ------------------------------ MPI Functions ------------------------------ #
def GenerateIndicesForDifferentProcs(nprocs, loopsize):
  """
  Generates a list that contains the elements of the loops that each
  rank will process. In the case that the number of processors does not divide
  the loop size, We divide the rest of the work amongst the first (loopsize % nprocs)
  processors.
  """
  rank = MPI.COMM_WORLD.Get_rank()
  if (nprocs <= loopsize):
    sizes        = divmod(loopsize,nprocs)
    indices_rank = np.empty((sizes[0]), dtype=int)

    for i in range(sizes[0]):
      indices_rank[i] = rank*sizes[0]+i

    if MPI.COMM_WORLD.Get_rank() < sizes[1]:
      indices_rank = np.append(indices_rank, nprocs+MPI.COMM_WORLD.Get_rank())

  elif (nprocs > loopsize):
    indices_rank = None

    if rank < loopsize:
      indices_rank = np.array([rank])

  return indices_rank

# ------------------------ Cluster-Related Functions ------------------------ #

def ListSimulationDirectories(bin_dir):
  """
  We count the number of directory that end in \d{5}.BQ. This gives us the
  number of simulation that we ran, and also their names.
  """
  import os
  import re
  dirList = [f for f in os.listdir(bin_dir) if re.search(r'(.*\d{5}.BQ)', f)]

  sortedList = sorted(dirList, key=str.lower)

  for i in range(len(sortedList)):
    sortedList[i] += "/{:05g}.BQ/".format(i+1)

  return sortedList


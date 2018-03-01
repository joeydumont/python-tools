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
def adjust_spines(ax, spines):
  """
  Helps in re-creating the spartan style of Jean-luc Doumont's graphs.

  Removes the spines that are not specified in spines, and colours the specified
  ones in gray, and pushes them outside the graph area.
  """
  for loc, spine in ax.spines.items():
      if loc in spines:
          spine.set_position(('outward', 10))  # outward by 10 points
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
        r"\usepackage{mathspec}",
        r"\usepackage[charter]{mathdesign}",
        r"\usepackage{fontspec}",
        r"\setmathfont{Fira Sans}",
        r"\setmainfont{Oswald}",
        ]
  }

  return pgf_with_pdflatex

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

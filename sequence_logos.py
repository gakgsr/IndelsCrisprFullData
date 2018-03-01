'''
Reference: https://stackoverflow.com/questions/42615527/sequence-logos-in-matplotlib-aligning-xticks
referenced on 03/01/2018 at 11:09 AM
'''
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings("ignore")

fp = FontProperties(family="Arial", weight="bold") 
globscale = 1.35
LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
            "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
            "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
            "C" : TextPath((-0.366, 0), "C", size=1, prop=fp) }
COLOR_SCHEME = {'G': 'orange', 
                'A': 'red', 
                'C': 'blue', 
                'T': 'darkgreen'}

def letterAt(letter, x, y, yscale = 1, ax = None):
  text = LETTERS[letter]
  t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
      mpl.transforms.Affine2D().translate(x,y) + ax.transData
  p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
  if ax != None:
    ax.add_artist(p)
  return p


# logo_heights is (n, 4) matrix of logo heights, reshaped as (n*4, 1)
def plot_seq_logo(logo_heights, name):
  logo_heights = np.reshape(logo_heights, (-1, 4))
  fig, ax = plt.subplots(figsize = (10, 3))
  x = 1
  maxi = 0
  mini = 0
  for i in range(logo_heights.shape[0]):
    list_of_heights = [('A', logo_heights[i, 0]), ('C', logo_heights[i, 1]), ('G', logo_heights[i, 2]), ('T', logo_heights[i, 3])]
    list_of_heights = sorted(list_of_heights, key = lambda x: x[1])
    y = np.sum(logo_heights[i][logo_heights[i] < 0])
    mini = min(y, mini)
    for base, score in list_of_heights:
      letterAt(base, x, y, abs(score), ax)
      y += abs(score)
    x += 1
    maxi = max(maxi, y)

  plt.xticks(range(1, x))
  plt.xlim((0, x)) 
  plt.ylim((mini, maxi)) 
  plt.tight_layout()      
  plt.savefig('sequence_logo_' + name + '.pdf')
  plt.clf()
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

fp = FontProperties(family="Arial", weight="bold") 
globscale = 1.35
LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
            "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
            "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
            "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
            "N" : TextPath((-0.35, 0), "N", size=1, prop=fp),
            ".": TextPath((-0.1, 0), ".", size=1, prop=fp),
            "-": TextPath((-0.2, 0), "-", size=1, prop=fp)}

COLOR_SCHEME = {'G': 'orange', 
                'A': 'red', 
                'C': 'blue', 
                'T': 'darkgreen',
                'N': 'gray',
                '.': 'black',
                '-': 'black'}

def letter_at(letter, x, y, xscale=0.3, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(xscale*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData

    #bbox_props = dict(facecolor=COLOR_SCHEME[letter], alpha=0.8)
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter], transform=t)
    #p = PathPatch(text, lw=0, fc='k', clip_box=bbox_props, transform=t)
    #p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter])
    if ax != None:
        ax.add_artist(p)
    return p

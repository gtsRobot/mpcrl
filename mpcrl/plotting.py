from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib import font_manager

def set_paper_style():
    names={f.name for f in font_manager.fontManager.ttflist}
    if 'SimSun' in names: plt.rcParams['font.sans-serif']=['SimSun']
    elif 'Songti SC' in names: plt.rcParams['font.sans-serif']=['Songti SC']
    else: plt.rcParams['font.sans-serif']=['DejaVu Sans']
    plt.rcParams['font.size']=10.5; plt.rcParams['axes.unicode_minus']=False

def outside_legend(ax): ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0),borderaxespad=0.0,frameon=False)

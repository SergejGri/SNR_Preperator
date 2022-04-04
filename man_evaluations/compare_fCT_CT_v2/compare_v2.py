import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cycler import cycler
import numpy as np


def compare_v2():
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 12,  # LaTeX default is 10pt font.
        "font.size": 12,
        "legend.fontsize": 12,  # Make the legend/label fonts
        "xtick.labelsize": 12,  # a little smaller
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.labelsize": 12,
        "pgf.preamble": "\n".join([r"\usepackage{libertine}",
                                   r"\usepackage[libertine]{newtxmath}",
                                   r"\usepackage{siunitx}"
                                   r"\usepackage[utf8]{inputenc}",
                                   r"\usepackage[T1]{fontenc}"])
    }
    mpl.use("pgf")
    mpl.rcParams.update(pgf_with_latex)
    plt.rcParams["figure.figsize"] = (6.29, 7.9)
    plt.rc('lines', linewidth=2)
    custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color',
                                                              ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97',
                                                               '#474747', '#9e9e9e'])
    plt.rc('axes', prop_cycle=custom_cycler)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, gridspec_kw={'height_ratios': [1, 2, 3, 3]})
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ROI = 20
    ONLY_FCT = False

    sv_name = rf'compare_fct_ct_ROI-{ROI}mum_newversion'
    sv_path = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\compare_fCT_CT_v2'

    data_fct = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\final_evaluations\3D_SNR_eval_10032022_biggerview_Cylinder_v1_ss20\SNR-Karte\fct_data.txt",
        skip_header=1)

    data_ct30 = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\final_evaluations\3D_SNR_eval_10032022_biggerview_Cylinder_v1_ss20\SNR-Karte\data_merge_data_small.txt",
        skip_header=1)
    data_ct120 = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\3D_SNR_eval_24012022_view_only_Cylinder_v1_ss58_without_bad_px\SNR-Karte\ct120_data.txt",
        skip_header=1)

    theta = np.arange(0, 360, 7.2)
    snr30 = data_ct30[:, 1]
    #theta30 = data_ct30[:, 3]

    snr120 = data_ct120[:, 1]
    #theta120 = data_ct120[:, 3]

    ax1.plot(theta, snr30)
    ax1.plot(theta, snr120)

    ax3.plot(theta, snr30)
    ax3.plot(theta, snr120)

    plt.tight_layout()
    plt.savefig(os.path.join(sv_path, sv_name + '.pdf'), bbox_inches='tight', dpi=600)







compare_v2()
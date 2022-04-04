import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from cycler import cycler
import numpy as np


def compare_fCT_CT():
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 12,  # LaTeX default is 10pt font.
        "font.size": 12,
        "legend.fontsize": 11.5,  # Make the legend/label fonts
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
    plt.rcParams["figure.figsize"] = (6.28, 7.9) # 6.29, 7.9
    plt.rc('lines', linewidth=2)
    custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
    plt.rc('axes', prop_cycle=custom_cycler)


    fig, (ax1, ax2, ax3) = plt.subplots(3, gridspec_kw={'height_ratios': [2, 5, 3]})

    ROI = 150
    ONLY_FCT = False

    sv_name = rf'compare_plt_fct_ct_ROI-{ROI}mum_20220318'
    sv_path = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\compare_fCT_CT'

    data_fct = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\fct_data.txt",
                    skip_header=1)

    data_ctm = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\avg_snr.txt",
                    skip_header=1)
    data_cto = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\ct480_data.txt",
                    skip_header=1)

    m               = data_fct[:, 5]
    theta           = data_fct[:, 3]
    fct_T           = data_fct[:, 0]
    fct_snr         = data_fct[:, 1]

    ct_Tm           = data_ctm[:, 0]
    ct_snrm         = data_ctm[:, 1]
    ct_snrmt        = data_ctm[:, 2]

    ct_To         = data_cto[:, 0]
    ct_snro         = data_cto[:, 1]
    ct_snrot        = data_cto[:, 2]

    m_compare = ct_snrmt / ct_snrot
    ct_expect = fct_snr*m



    index = np.argwhere(ct_To < 0)

    theta = np.delete(theta, index)
    m = np.delete(m, index)
    fct_T = np.delete(fct_T, index)
    fct_snr = np.delete(fct_snr, index)

    ct_Tm = np.delete(ct_Tm, index)
    ct_snrm = np.delete(ct_snrm, index)
    ct_snrmt = np.delete(ct_snrmt, index)

    ct_To = np.delete(ct_To, index)
    ct_snro = np.delete(ct_snro, index)
    ct_snrot = np.delete(ct_snrot, index)



    cfct = '#EE7972'
    cm = '#0C5DA5'
    co = '#FF9500'
    carea = '#0C5DA5'
    aarea = 0.1
    ms = 4
    txtpad = 0.05
    afct = 1



    ax1.step(theta, m, linestyle='-', c='k')
    #ax1.step(theta, m_compare, c='blue')
    ax1.set_ylabel(r'$m$')
    ax1.set_ylim([2.7, 6.5])
    ax1.set_yticks(np.arange(min(m), max(m) + 1, 1.0))


    ax2.plot(theta, fct_snr, c='#9e9e9e', zorder=5, marker='o', markersize=ms, alpha=afct, label=r'$\text{SNR}_{\text{fCT}}$')
    ax2.plot(theta, ct_expect, marker='o', markersize=ms, zorder=10, alpha=1, label=r'$\text{SNR}_{\text{theo}}$')
    ax2.plot(theta, ct_snrmt, c=cm, zorder=10, marker='o', markersize=ms, label=r'$\text{SNR}_{m}$')
    ax2.plot(theta, ct_snrot, c=co, zorder=10, marker='o', markersize=ms, label=r'$\text{SNR}_{0}$')
    ax2.set_ylim([20, 420])
    ax2.legend(loc='best', ncol=1).set_zorder(11)
    ax2.set_yticks(np.arange(20, 420, 50))
    #ax2.set_xlabel(r'Durchstrahlungswinkel $\theta$ $[^\circ]$')
    ax2.set_ylabel(r'SNR')
    ax2.grid(alpha=0.3, linewidth=1)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax2.axhline(y=350, linestyle='--', linewidth=1, c='k')


    ax3.plot(theta, ct_snrot, c=co, zorder=10, marker='o', markersize=ms)
    ax3.set_xlabel(r'Durchstrahlungswinkel $\theta$ $[^\circ]$')
    ax3.set_ylabel(r'$\text{SNR}_{0}$')
    ax3.grid(alpha=0.3, linewidth=1, c='#9e9e9e')

    ax2.annotate(r'$\text{SNR}_{\text{B}}$', zorder=12, xy=(0, 350), ha='center', va='center')

    a1 = ax3.fill_between(theta, 30, 60, where=np.logical_and(theta[1] <= theta,  theta <= theta[3]),
                    facecolor=carea, alpha=aarea, zorder=5, label=r'Überlappung der Ecken')

    a2 = ax3.fill_between(theta, 30, 60, where=np.logical_and(theta[7] <= theta, theta <= theta[9]),
                     facecolor=carea, alpha=aarea, zorder=5)

    a3 = ax3.fill_between(theta, 30, 60, where=np.logical_and(theta[25] <= theta, theta <= theta[27]),
                     facecolor=carea, alpha=aarea, zorder=5)

    a4 = ax3.fill_between(theta, 30, 60, where=np.logical_and(theta[31] <= theta, theta <= theta[33]),
                     facecolor=carea, alpha=aarea, zorder=5)

    handles, labels = plt.gca().get_legend_handles_labels()
    #handles.extend([a1])
    ax3.legend(handles=handles, loc='lower left')
    ax3.set_ylim([35, 57])
    #ax1.annotate(r'\textbf{A}', xy=(0, 5.6), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5),
    #             ha='center', va='center')

    #ax2.annotate(r'\textbf{B}', xy=(0, 300), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5),
    #             ha='center', va='center')


    plt.tight_layout()
    plt.savefig(os.path.join(sv_path, sv_name + '.pdf'), bbox_inches='tight', dpi=600)




def compare_fCT_CT_window():
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 12,  # LaTeX default is 10pt font.
        "font.size": 12,
        "legend.fontsize": 11.5,  # Make the legend/label fonts
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
    plt.rcParams["figure.figsize"] = (2.5, 4) # 6.29, 7.9
    plt.rc('lines', linewidth=2)
    custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
    plt.rc('axes', prop_cycle=custom_cycler)


    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [2, 5]})

    ROI = 150
    ONLY_FCT = False

    sv_name = rf'compare_window_ROI-{ROI}mum'
    sv_path = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\compare_fCT_CT'

    data_fct = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\fct_data.txt",
                    skip_header=1)

    data_ctm = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\avg_snr.txt",
                    skip_header=1)
    data_cto = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\ct480_data.txt",
                    skip_header=1)

    m               = data_fct[:, 5]
    theta           = data_fct[:, 3]
    fct_T           = data_fct[:, 0]
    fct_snr         = data_fct[:, 1]

    ct_Tm           = data_ctm[:, 0]
    ct_snrm         = data_ctm[:, 1]
    ct_snrmt        = data_ctm[:, 2]

    ct_To         = data_cto[:, 0]
    ct_snro         = data_cto[:, 1]
    ct_snrot        = data_cto[:, 2]


    index = np.argwhere(ct_To < 0)

    theta = np.delete(theta, index)
    m = np.delete(m, index)
    fct_T = np.delete(fct_T, index)
    fct_snr = np.delete(fct_snr, index)

    ct_Tm = np.delete(ct_Tm, index)
    ct_snrm = np.delete(ct_snrm, index)
    ct_snrmt = np.delete(ct_snrmt, index)

    ct_To = np.delete(ct_To, index)
    ct_snro = np.delete(ct_snro, index)
    ct_snrot = np.delete(ct_snrot, index)

    cfct = '#EE7972'
    cm = '#0C5DA5'
    co = '#FF9500'
    ms = 4
    txtpad = 0.05
    afct = 1

    ax1.step(theta, m, linestyle='-', c='#0C5DA5')
    ax1.set_ylabel(r'$m$')
    ax1.set_ylim([2.7, 6.5])
    ax1.set_xlim([0, 60])
    ax1.set_yticks(np.arange(min(m), max(m) + 1, 1.0))

    ax2.plot(theta, fct_snr, c='#9e9e9e', zorder=0, marker='o', markersize=ms, alpha=afct, label=r'$\text{SNR}_{\text{fCT}}$ (\SI{90}{\kilo\volt})')
    if not ONLY_FCT:
        ax2.plot(theta, ct_snrmt, c=cm, zorder=10, marker='o', markersize=ms, label=r'$\text{SNR}_{m}$ (\SI{50}{\kilo\volt})')
        ax2.plot(theta, ct_snrot, c=co, zorder=10, marker='o', markersize=ms, label=r'$\text{SNR}_{0}$ (\SI{50}{\kilo\volt})')
    ax2.set_ylim([20, 330])
    ax2.set_xlim([0, 60])
    ax2.set_yticks(np.arange(20, 330, 50))
    ax2.set_xlabel(r'Durchstrahlungswinkel $\theta$ $[^\circ]$')
    ax2.set_ylabel(r'SNR')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(sv_path, sv_name + '.pdf'), bbox_inches='tight', dpi=600)


def compare_SNRm_SNRo():
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 12,  # LaTeX default is 10pt font.
        "font.size": 12,
        "legend.fontsize": 11.5,  # Make the legend/label fonts
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
    plt.rcParams["figure.figsize"] = (6.28, 6)  # 6.29, 7.9
    plt.rc('lines', linewidth=2)
    custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color',
                                                              ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97',
                                                               '#474747', '#9e9e9e'])
    plt.rc('axes', prop_cycle=custom_cycler)

    fig, (ax1, ax2, ax3) = plt.subplots(3, gridspec_kw={'height_ratios': [2, 3, 3]})

    ROI = 150
    ONLY_FCT = False

    sv_name = rf'compare_SNRm-SNRo-{ROI}mum_20220318'
    sv_path = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\compare_fCT_CT'

    data_fct = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\fct_data.txt",
        skip_header=1)

    data_ctm = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\avg_snr.txt",
        skip_header=1)
    data_cto = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\ct480_data.txt",
        skip_header=1)

    m = data_fct[:, 5]
    theta = data_fct[:, 3]
    fct_T = data_fct[:, 0]
    fct_snr = data_fct[:, 1]

    ct_Tm = data_ctm[:, 0]
    ct_snrm = data_ctm[:, 1]
    ct_snrmt = data_ctm[:, 2]

    ct_To = data_cto[:, 0]
    ct_snro = data_cto[:, 1]
    ct_snrot = data_cto[:, 2]

    index = np.argwhere(ct_To < 0)

    theta = np.delete(theta, index)
    m = np.delete(m, index)
    fct_T = np.delete(fct_T, index)
    fct_snr = np.delete(fct_snr, index)

    ct_Tm = np.delete(ct_Tm, index)
    ct_snrm = np.delete(ct_snrm, index)
    ct_snrmt = np.delete(ct_snrmt, index)

    ct_To = np.delete(ct_To, index)
    ct_snro = np.delete(ct_snro, index)
    ct_snrot = np.delete(ct_snrot, index)

    cfct = '#EE7972'
    cm = '#0C5DA5'
    co = '#FF9500'
    ms = 4
    txtpad = 0.05
    afct = 1

    #mean_fct = np.mean(fct_snr)
    #median_fct = np.median(fct_snr)
    #print(mean_fct)
    #print(median_fct)
    ax1.step(theta, m, linestyle='-', c='#0C5DA5')
    ax1.set_ylabel(r'$m$')
    ax1.set_ylim([2.7, 6.5])
    ax1.set_yticks(np.arange(min(m), max(m) + 1, 1.0))


    ax2.plot(theta, ct_snrot, c=co, zorder=10, marker='o', markersize=ms, label=r'$SNR_{0}$ (\SI{50}{\kilo\volt})')
    ax2.axvline(x=theta[1])
    ax2.axvline(x=theta[2])

    ax2.axvline(x=theta[5])
    ax2.axvline(x=theta[7])

    ax3.plot(theta, ct_snrmt, c=cm, zorder=10, marker='o', markersize=ms,label=r'$SNR_{m}$ (\SI{50}{\kilo\volt})')

    ax3.set_ylim([20, 330])
    ax3.legend(loc='upper right', ncol=1)
    ax3.set_yticks(np.arange(20, 330, 50))
    ax3.set_xlabel(r'Durchstrahlungswinkel $\theta$ $[^\circ]$')
    ax3.set_ylabel(r'$SNR$')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.grid(alpha=0.3, linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(sv_path, sv_name + '.pdf'), bbox_inches='tight', dpi=600)


def compare_T_only():
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 12,  # LaTeX default is 10pt font.
        "font.size": 12,
        "legend.fontsize": 11.5,  # Make the legend/label fonts
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
    plt.rcParams["figure.figsize"] = (6.28, 4) # 6.29, 7.9
    plt.rc('lines', linewidth=2)
    custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
    plt.rc('axes', prop_cycle=custom_cycler)


    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [5, 5]})

    ROI = 150
    ONLY_FCT = False

    sv_name = rf'compare_plt_T_only_ROI-{ROI}mum_20220318'
    sv_path = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\compare_fCT_CT'

    data_fct = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\fct_data.txt",
                    skip_header=1)

    data_ctm = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\avg_snr.txt",
                    skip_header=1)
    data_cto = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-150_SNRB-350\SNR-Karte\ct480_data.txt",
                    skip_header=1)

    m               = data_fct[:, 5]
    theta           = data_fct[:, 3]
    fct_T           = data_fct[:, 0]
    fct_snr         = data_fct[:, 1]

    ct_Tm           = data_ctm[:, 0]
    ct_snrm         = data_ctm[:, 1]
    ct_snrmt        = data_ctm[:, 2]

    ct_To         = data_cto[:, 0]
    ct_snro         = data_cto[:, 1]
    ct_snrot        = data_cto[:, 2]


    index = np.argwhere(ct_To < 0)

    theta = np.delete(theta, index)
    m = np.delete(m, index)
    fct_T = np.delete(fct_T, index)
    fct_snr = np.delete(fct_snr, index)

    ct_Tm = np.delete(ct_Tm, index)
    ct_snrm = np.delete(ct_snrm, index)
    ct_snrmt = np.delete(ct_snrmt, index)

    ct_To = np.delete(ct_To, index)
    ct_snro = np.delete(ct_snro, index)
    ct_snrot = np.delete(ct_snrot, index)



    cfct = '#EE7972'
    cm = '#0C5DA5'
    co = '#FF9500'
    ms = 5
    txtpad = 0.05
    afct = 1


    ax2.plot()
    ax2.set_xlim([-17.5, 370.5])
    ax2.set_ylabel(r'Obj. Orientierung')
    ax2.set_xlabel(r'Durchstrahlungswinkel $\theta$ $[^\circ]$')

    ax1.plot(theta, fct_T, c='#9e9e9e', zorder=0, marker='o', markersize=ms, alpha=afct, label=r'$T_{\text{fCT}}$ (\SI{90}{\kilo\volt})')
    if not ONLY_FCT:
        ax1.plot(theta, ct_Tm, c=cm, zorder=10, marker='o', markersize=ms, label=r'$T_{SNR_{m}}$ (\SI{50}{\kilo\volt})')
        ax1.plot(theta, ct_To, c=co, zorder=5, marker='o', markersize=ms)
    ax1.set_ylim([0.14, 0.5])
    ax1.legend()
    ax1.set_ylabel(r'Transmission [w.E.]')

    plt.tight_layout()
    plt.savefig(os.path.join(sv_path, sv_name + '.pdf'), bbox_inches='tight', dpi=600)


compare_fCT_CT()
compare_T_only()
compare_fCT_CT_window()
compare_SNRm_SNRo()
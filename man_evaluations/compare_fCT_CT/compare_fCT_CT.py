import os
import matplotlib as mpl
import matplotlib.pyplot as plt
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
        "legend.fontsize": 10,  # Make the legend/label fonts
        "xtick.labelsize": 12,  # a little smaller
        "ytick.labelsize": 12,
        "pgf.preamble": "\n".join([r"\usepackage{libertine}",
                                   r"\usepackage[libertine]{newtxmath}",
                                   r"\usepackage{siunitx}"
                                   r"\usepackage[utf8]{inputenc}",
                                   r"\usepackage[T1]{fontenc}"])
    }
    mpl.use("pgf")
    mpl.rcParams.update(pgf_with_latex)
    plt.rcParams["figure.figsize"] = (6.3, 8.5)
    plt.rc('lines', linewidth=2)
    #plt.rc('axes', prop_cycle=(cycler('color', ['#332288', '#117733', '#44AA99', '#882255', '#999933', '#88CCEE', '#DDCC77', '#CC6677', '#882255', '#AA4499', '#BBBBBB'])))
    #plt.rc('axes',prop_cycle=(cycler('color', [#102694,   '#469D59',  '#EE7972', '#E2B43C', '#CC7444', '#002C2B',  '#3372F5', '#FDEADB', '#BBBBBB'])))
    #                                           dunkelblau  hellgrün      rot        gelb      orange    dunkelgrün   BLAU        beige       grau
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, gridspec_kw={'height_ratios': [1, 2, 3, 3]})



    ROI = 58
    ONLY_FCT = False

    sv_name = rf'compare_plt_fct_ct_ROI-{ROI}mum'
    sv_path = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\compare_fCT_CT'

    data_fct = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\3D_SNR_eval_24012022_view_only_Cylinder_v1_ss58_without_bad_px\SNR-Karte\fct_data.txt",
                    skip_header=1)

    data_ct30 = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\3D_SNR_eval_24012022_view_only_Cylinder_v1_ss58_without_bad_px\SNR-Karte\ct30_data.txt",
                    skip_header=1)
    data_ct120 = np.genfromtxt(
                    r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\3D_SNR_eval_24012022_view_only_Cylinder_v1_ss58_without_bad_px\SNR-Karte\ct120_data.txt",
                    skip_header=1)

    ct_T30      = data_ct30[:, 0]
    ct_snr30    = data_ct30[:, 1]
    ct_d30      = data_ct30[:, 2]
    ct_theta30  = data_ct30[:, 3]
    #ct_texp30  = data_ct30[:, 4]
    ct_avg30    = data_ct30[:, 5]

    ct_T120     = data_ct120[:, 0]
    ct_snr120   = data_ct120[:, 1]
    ct_d120     = data_ct120[:, 2]
    ct_theta120 = data_ct120[:, 3]
    #ct_texp120 = data_ct120[:, 4]
    ct_avg120   = data_ct120[:, 5]

    fct_T       = data_fct[:, 0]
    fct_snr     = data_fct[:, 1]
    fct_d       = data_fct[:, 2]
    fct_theta   = data_fct[:, 3]
    #fct_texp   = data_fct[:, 4]
    fct_avg     = data_fct[:, 5]

    cfct = '#EE7972'
    cct30 = '#3574F7'
    cct120 = '#469D59'
    ms = 20
    txtpad = 0.05



    #ax11 = ax1.twinx()
    ax1.scatter(fct_theta, fct_avg, s=ms, c='#555555', label='gem. Proj.')
    ax1.plot(fct_theta, fct_avg, c='#555555')
    ax1.set_ylabel(r'Gemittelte Proj.')
    ax1.set_yticks(np.arange(min(ct_avg30), max(ct_avg30) + 1, 1.0))

    fct = ax2.scatter(fct_theta, fct_d, s=ms, c=cfct, label=r'fCT')
    ax2.plot(fct_theta, fct_d, c=cfct)
    if not ONLY_FCT:
        ct30 = ax2.scatter(ct_theta30, ct_d30, s=ms, c=cct30, label=r'$\text{CT}_{30}$')
        ax2.plot(ct_theta30, ct_d30, c=cct30)
        ct120 = ax2.scatter(ct_theta120, ct_d120, s=ms, c=cct120, label=r'$\text{CT}_{120}$')
        ax2.plot(ct_theta120, ct_d120, c=cct120)
    #ax2.set_yticks([6, 9, 12, 15, 18])
    ax2.set_ylim([8, 19])
    ax2.set_ylabel(r'$d_{\text{theo}}$ [mm]')

    ax3.scatter(fct_theta, fct_snr, s=ms, c=cfct, label=r'$SNR(\theta)_{\text{fCT}}$ (\SI{90}{\kilo\eV})')
    ax3.plot(fct_theta, fct_snr, c=cfct)
    if not ONLY_FCT:
        ax3.scatter(ct_theta30, ct_snr30, s=ms, c=cct30, label=r'$SNR(\theta)_{\text{CT}}^{30}$ (\SI{102}{\kilo\eV})')
        ax3.plot(ct_theta30, ct_snr30, c=cct30)
        ax3.scatter(ct_theta120, ct_snr120, s=ms, c=cct120, label=r'$SNR(\theta)_{\text{CT}}^{120}$ (\SI{102}{\kilo\eV})')
        ax3.plot(ct_theta120, ct_snr120, c=cct120)
    #ax3.legend(loc=2, handletextpad=txtpad)
    ax3.set_ylim([0.0, 0.85])
    ax3.set_ylabel(r'SNR')

    ax4.scatter(fct_theta, fct_T, s=ms, c=cfct, label=r'$T_{\mathrm{fCT}}$ (\SI{90}{\kilo\eV})')
    ax4.plot(fct_theta, fct_T, c=cfct)
    if not ONLY_FCT:
        ax4.scatter(ct_theta30, ct_T30, s=ms, c=cct30, label=r'$T_{\text{CT}}^{30}$ (\SI{102}{\kilo\eV})')
        ax4.plot(ct_theta30, ct_T30, c=cct30)
        ax4.scatter(ct_theta120, ct_T120, s=ms, c=cct120, label=r'$T_{\text{CT}}^{120}$ (\SI{102}{\kilo\eV})')
        ax4.plot(ct_theta120, ct_T120, c=cct120)
    #ax4.legend(loc=2, handletextpad=txtpad)
    ax4.set_xlabel(r'Winkel $\theta [\circ]$')
    ax4.set_ylabel(r'Transmission [w.E.]')

    ax4.legend(handles=[fct, ct30, ct120],
               labels=[r'$\text{fCT}(\SI{90}{\kilo\eV})$', r'$\text{CT}_{30}(\SI{102}{\kilo\eV})$',
                       r'$\text{CT}_{120}(\SI{102}{\kilo\eV})$'],
               loc='upper center',
               bbox_to_anchor=(0.5, -0.3),
               handletextpad=txtpad,
               fancybox=False,
               shadow=False,
               ncol=3)

    fig.suptitle(f'Messung vs. Theorie @ {ROI}$\mu m$')
    plt.tight_layout()
    plt.savefig(os.path.join(sv_path, sv_name + '.pdf'), bbox_inches='tight', dpi=600)

compare_fCT_CT()

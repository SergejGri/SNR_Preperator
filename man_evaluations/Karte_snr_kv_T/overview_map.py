import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from itertools import cycle
import matplotlib.gridspec as gridspec


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
    "ytick.labelsize": 12,
    "pgf.preamble": "\n".join([r"\usepackage{libertine}",
                               r"\usepackage[libertine]{newtxmath}",
                               r"\usepackage{siunitx}"
                               r"\usepackage[utf8]{inputenc}",
                               r"\usepackage[T1]{fontenc}"])
}
mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)


custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color', ['#3372F5', '#469D59', '#EE7972', '#E2B43C', '#CC7444'])
plt.rc('axes', prop_cycle=custom_cycler)
#plt.rc('axes', prop_cycle=(cycler('color', ['#3372F5', '#469D59', '#EE7972', '#E2B43C', '#CC7444']) +
#                               cycler('linestyle', ['-', '--'])))


def kv_T_kv_SNR():
    plt.rcParams["figure.figsize"] = (6.3, 8)
    base_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge\Karte\Kurven'
    path_result = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\Karte_snr_kv_T'
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28, 32]

    files = {}
    for file in os.listdir(base_path):
        mm = int(file.split('mm')[0])
        sub_dict = {f'{file}': mm}
        files.update(sub_dict)
    fsorted = {k: v for k, v in sorted(files.items(), key=lambda x: x[1])}

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    for f in fsorted:
        working_file = os.path.join(base_path, f)
        d = int(f.split('mm')[0])
        if d in raw_mm:
            data = np.genfromtxt(fname=working_file, skip_header=0, delimiter=';')
            kV = data[:, 0]
            T = data[:, 1]
            SNR = data[:, 2]
            l = f'{d} mm'
            ax1.plot(kV, T, marker='o')
            ax2.plot(kV, SNR, label=l, marker='o')

    ax1.set_prop_cycle(custom_cycler)
    ax1.set_ylabel(r'Transmission [w.E.]')
    ax2.set_xlabel(r'Spannung $U$ [\SI{1e3}{\kilo\volt}]')
    ax2.set_ylim([0.005, 140])
    ax2.set_ylabel(r'SNR [$s^{-1}$]')
    ax2.set_yscale('log')
    ax2.legend(loc='lower center', fancybox=True, shadow=False, ncol=5, handletextpad=0.1, labelspacing=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'kV-T.pdf'), bbox_inches="tight", dpi=600)





def T_SNR_karte():
    plt.rcParams["figure.figsize"] = (8.74, 6)
    plt.rc('lines', linewidth=2)

    base_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge\Karte\Kurven'
    path_result = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\Karte_snr_kv_T'
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28, 32]

    files = {}
    for file in os.listdir(base_path):
        mm = int(file.split('mm')[0])
        sub_dict = {f'{file}': mm}
        files.update(sub_dict)
    fsorted = {k: v for k, v in sorted(files.items(), key=lambda x: x[1])}

    '''
    for f in fsorted:
        d = int(f.split('mm')[0])
        if d not in raw_mm:
            working_file = os.path.join(base_path, f)
            data = np.genfromtxt(fname=working_file, skip_header=0, delimiter=';')
            kV = data[:, 0]
            T = data[:, 1]
            SNR = data[:, 2]

            l = f'{d} mm'

            c = '#BBBBBB'
            ax1.plot(kV, T, c=c, linewidth=1, zorder=0)
            ax2.plot(kV, SNR, c=c, linewidth=1, zorder=0)
        else:
            pass
    '''
    plt.figure()
    fig, ax3 = plt.subplots()

    for f in fsorted:
        working_file = os.path.join(base_path, f)
        d = int(f.split('mm')[0])
        if d in raw_mm:
            data = np.genfromtxt(fname=working_file, skip_header=0, delimiter=';')
            kV = data[:, 0]
            T = data[:, 1]
            SNR = data[:, 2]

            if d == 2:
                ax3.annotate('\SI{50}{\kilo\eV}', xy=(T[0], SNR[0]), xytext=(-30, 20),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='#BBBBBB'))
                ax3.annotate('\SI{180}{\kilo\eV}', xy=(T[-1], SNR[-1]), xytext=(20, -30),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='#BBBBBB'))
            l = f'{d} mm'

            ax3.plot(T, SNR, label=l, marker='o')
        else:
            pass




    ax3.set_xlabel(r'Transmission [w.E.]')
    ax3.set_ylabel(r'SNR [$s^{-1}$]')
    ax3.legend(loc='best', fancybox=True, shadow=False, ncol=1)
    ax3.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'T_SNR_Karte.pdf'), bbox_inches="tight", dpi=600)






kv_T_kv_SNR()
T_SNR_karte()
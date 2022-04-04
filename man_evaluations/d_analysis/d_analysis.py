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
    "legend.fontsize": 11,  # Make the legend/label fonts
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

ms = 5

custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
plt.rc('axes', prop_cycle=custom_cycler)



def d_analysis():
    plt.rcParams["figure.figsize"] = (6.3, 7.1)
    base_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge\Karte\Kurven'
    path_result = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\Karte_snr_kv_T'
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28]
    kvs = [50, 60, 80, 90, 100, 120, 130, 150, 160, 180]

    files = {}
    for file in os.listdir(base_path):
        mm = int(file.split('mm')[0])
        sub_dict = {f'{file}': mm}
        files.update(sub_dict)
    fsorted = {k: v for k, v in sorted(files.items(), key=lambda x: x[1])}


    Ts = []
    SNRs = []
    resultT = []
    resultSNR = []
    for d in raw_mm:
        working_file = os.path.join(base_path, f'{d}mm_raw-data.txt')
        data = np.genfromtxt(fname=working_file, skip_header=0, delimiter=';')
        for i in range(len(kvs)):
            T = data[i, 1]
            SNR = data[i, 2]
            Ts.append(T)
            SNRs.append(SNR)
        resultT.append(Ts)
        resultSNR.append(SNRs)
        Ts = []
        SNRs = []


    for j in range(len(resultSNR)):
        for k in range(len(resultSNR[0])-1):
            deltaT = resultT[j][k + 1] - resultT[j][k]
            deltaSNR = resultSNR[j][k+1] - resultSNR[j][k]



    fig, ax = plt.subplots()
    for f, ff in zip(fsorted, fsorted):
        working_file = os.path.join(base_path, f)
        d = int(f.split('mm')[0])
        if d in raw_mm:
            data = np.genfromtxt(fname=working_file, skip_header=0, delimiter=';')
            kV = data[:, 0]
            T = data[:, 1]
            SNR = data[:, 2]



            l = f'{d} mm'
            ax.plot(kV, T, marker='o', markersize=ms)
            ax.plot(kV, SNR, label=l, marker='o', markersize=ms)

    ax.annotate(r'Aluminium (Z=13)', xy=(65, 0.8), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                 ha='center', va='bottom')


    ax.set_ylabel(r'Transmission [w. E.]')
    ax.set_xlabel(r'Spannung $U$ [\SI{e3}{\kilo\volt}]')
    ax.set_ylim([0.005, 200])
    ax.set_ylabel(r'SNR [$s^{-1}$]')
    ax.set_yscale('log')


    ax.legend(loc='lower center', fancybox=True, shadow=False, ncol=5, handletextpad=0.2, labelspacing=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'kV-T-new.pdf'), bbox_inches="tight", dpi=600)


d_analysis()
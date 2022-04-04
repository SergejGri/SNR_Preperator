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



def kv_T_kv_SNR():
    plt.rcParams["figure.figsize"] = (6.3, 7.1)
    base_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge\Karte\Kurven'
    path_result = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\Karte_snr_kv_T'
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28]

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
            _max_idx = np.argmax(SNR)

            l = f'{d} mm'
            ax1.plot(kV, T, marker='o', markersize=ms)
            ax2.plot(kV, SNR, label=l, marker='o', markersize=ms)

    ax1.annotate(r'Aluminium (Z=13)', xy=(65, 0.8), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                 ha='center', va='bottom')


    ax1.set_ylabel(r'Transmission [w. E.]')
    ax2.set_xlabel(r'Spannung $U$ [\SI{e3}{\kilo\volt}]')
    ax2.set_ylim([0.005, 200])
    ax2.set_ylabel(r'SNR [$s^{-1}$]')
    ax2.set_yscale('log')

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax2.legend(loc='lower center', fancybox=True, shadow=False, ncol=5, handletextpad=0.2, labelspacing=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'kV-T-new.pdf'), bbox_inches="tight", dpi=600)


def kv_SNR():
    plt.rcParams["figure.figsize"] = (6.3, 7.1)
    base_path = r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\20220223_stepWedge\Karte\Kurven'
    path_result = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\Karte_snr_kv_T'
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28]

    files = {}
    for file in os.listdir(base_path):
        mm = int(file.split('mm')[0])
        sub_dict = {f'{file}': mm}
        files.update(sub_dict)
    fsorted = {k: v for k, v in sorted(files.items(), key=lambda x: x[1])}


    fig, ax1 = plt.subplots()
    for f in fsorted:
        working_file = os.path.join(base_path, f)
        d = int(f.split('mm')[0])
        if d in raw_mm:
            data = np.genfromtxt(fname=working_file, skip_header=0, delimiter=';')
            kV = data[:, 0]
            T = data[:, 1]
            SNR = data[:, 2]
            _max_idx = np.argmax(SNR)

            l = f'{d} mm'
            ax1.plot(SNR, kV, marker='o', markersize=ms, label=l)


    ax1.set_ylabel(r'U')
    ax1.set_xlabel(r'SNR')
    #ax2.set_xlabel(r'Spannung $U$ [\SI{e3}{\kilo\volt}]')
    #ax2.set_ylim([0.005, 200])
    #ax2.set_ylabel(r'SNR [$s^{-1}$]')
    ax1.set_xscale('log')

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax1.legend(loc='lower center', fancybox=True, shadow=False, ncol=5, handletextpad=0.2, labelspacing=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'U(SNR).pdf'), bbox_inches="tight", dpi=600)




def T_SNR_karte():
    plt.rcParams["figure.figsize"] = (8.77, 6.3)
    plt.rc('lines', linewidth=2)

    base_path = r'C:\Users\Sergej Grischagin\Desktop\final_evaluations\20220223_stepWedge\Karte\Kurven'
    path_result = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\Karte_snr_kv_T'
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28, 32]

    files = {}
    for file in os.listdir(base_path):
        mm = int(file.split('mm')[0])
        sub_dict = {f'{file}': mm}
        files.update(sub_dict)
    fsorted = {k: v for k, v in sorted(files.items(), key=lambda x: x[1])}


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
                ax3.annotate(r'\SI{50}{\kilo\volt}', xy=(T[0], SNR[0]), xytext=(-30, 20),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='#BBBBBB'))
                ax3.annotate(r'\SI{180}{\kilo\volt}', xy=(T[-1], SNR[-1]), xytext=(20, -30),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='#BBBBBB'))

            l = f'{d} mm'

            ax3.plot(T, SNR, label=l, marker='o', markersize=ms)
        else:
            pass
    ax3.annotate(r'Aluminium (Z=13)', xy=(0.07, 90), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                 ha='center', va='bottom')


    ax3.set_xlabel(r'Transmission [w. E.]')
    ax3.set_ylabel(r'SNR [$s^{-1}$]')
    ax3.legend(loc='lower right', fancybox=True, shadow=False, ncol=1)
    ax3.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'T_SNR_Karte.pdf'), bbox_inches="tight", dpi=600)



def d_kv():
    plt.rcParams["figure.figsize"] = (8.74, 6)
    plt.rc('lines', linewidth=2)

    base_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge\Karte\Kurven'
    path_result = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\Karte_snr_kv_T'
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28]

    files = {}
    for file in os.listdir(base_path):
        mm = int(file.split('mm')[0])
        sub_dict = {f'{file}': mm}
        files.update(sub_dict)
    fsorted = {k: v for k, v in sorted(files.items(), key=lambda x: x[1])}


    plt.figure()
    fig, ax1 = plt.subplots()

    for f in fsorted:
        working_file = os.path.join(base_path, f)
        d = int(f.split('mm')[0])
        if d in raw_mm:
            data = np.genfromtxt(fname=working_file, skip_header=0, delimiter=';')
            kV = data[:, 0]
            T = data[:, 1]
            SNR = data[:, 2]

            if d == 2:
                ax1.annotate(r'\SI{50}{\kilo\volt}', xy=(T[0], SNR[0]), xytext=(-30, 20),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='#BBBBBB'))
                ax1.annotate(r'\SI{180}{\kilo\volt}', xy=(T[-1], SNR[-1]), xytext=(20, -30),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='#BBBBBB'))

            l = f'{d} mm'

            ax1.plot(raw_mm, SNR, label=l, marker='o')
        else:
            pass
    ax1.annotate(r'Aluminium (Z=13)', xy=(0.07, 90), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                 ha='center', va='bottom')


    ax1.set_xlabel(r'd [w. E.]')
    ax1.set_ylabel(r'Spannung [$s^{-1}$]')
    ax1.legend(loc='lower right', fancybox=True, shadow=False, ncol=1)
    ax1.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'd_kv.pdf'), bbox_inches="tight", dpi=600)



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
                ax3.annotate(r'\SI{50}{\kilo\volt}', xy=(T[0], SNR[0]), xytext=(-30, 20),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='#BBBBBB'))
                ax3.annotate(r'\SI{180}{\kilo\volt}', xy=(T[-1], SNR[-1]), xytext=(20, -30),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='#BBBBBB'))

            l = f'{d} mm'
            ax3.plot(T, SNR, label=l, marker='o', markersize=ms)
            if d == 4:
                print(f'd=4: SNR_max={SNR[3]}')
            if d == 8:
                print(f'd=8: SNR_max={SNR[3]}')
            if d == 24:
                print(f'd=24: SNR_max={SNR[3]}')
            if d == 28:
                print(f'd=28: SNR_max={SNR[3]}')
        else:
            pass
    ax3.annotate(r'Aluminium (Z=13)', xy=(0.07, 90), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                 ha='center', va='bottom')


    ax3.set_xlabel(r'Transmission [w. E.]')
    ax3.set_ylabel(r'SNR [$s^{-1}$]')
    ax3.legend(loc='lower right', fancybox=True, shadow=False, ncol=1)
    ax3.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'T_SNR_Karte.pdf'), bbox_inches="tight", dpi=600)



def log():
    plt.rcParams["figure.figsize"] = (6.3, 8)
    base_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge\Karte\Kurven'
    path_result = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\Karte_snr_kv_T'
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28]

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
            _max_idx = np.argmax(SNR)

            l = f'{d} mm'
            ax1.semilogy(kV, -np.log(T)/d, marker='o', markersize=ms)
            ax2.plot(kV, SNR, label=l, marker='o', markersize=ms)

    ax1.annotate(r'Aluminium (Z=13)', xy=(65, 0.8), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                 ha='center', va='bottom')


    ax1.set_ylabel(r'Transmission [w. E.]')
    ax2.set_xlabel(r'Spannung $U$ [\SI{e3}{\kilo\volt}]')
    ax2.set_ylim([0.005, 200])
    ax2.set_ylabel(r'SNR [$s^{-1}$]')
    ax2.set_yscale('log')

    ax2.legend(loc='lower center', fancybox=True, shadow=False, ncol=5, handletextpad=0.1, labelspacing=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_result, f'kV-T-ges.pdf'), bbox_inches="tight", dpi=600)



kv_SNR()
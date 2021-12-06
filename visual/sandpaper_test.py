from cycler import cycler
import matplotlib.gridspec as gridspec
import matplotlib as plt
import numpy as np
import os

def sandpaper_test():
    plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    plt.rc('text', usetex=True)
    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=(cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])))
    plt.rcParams['axes.xmargin'] = 0

    gs = gridspec.GridSpec(8, 8)
    fig = plt.figure(figsize=(12,5))
    ax1 = plt.subplot(gs[:7, :4])
    ax2 = plt.subplot(gs[:7, 4:])

    lbl_size = 12                   # label font size
    ttl_size = 15                   # title font size
    axs_size = 13                   # axis font size
    lgnd_size = 11                  # legend font size
    _w = 1.5                        # plot border width


    file_1 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\2xP600-2xP240-2xP80\2021-9-28_SNR\100kV\SNR_100kV_2_mm_expTime_365.txt"
    file_2 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\4xP5000-2xP600-2xP240-2xP80\2021-9-28_SNR\100kV\SNR_100kV_2_mm_expTime_365.txt"
    file_3 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\4xP80\2021-9-28_SNR\100kV\SNR_100kV_2_mm_expTime_365.txt"
    file_4 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\without-sandpaper\2021-9-28_SNR\100kV\SNR_100kV_2_mm_expTime_365.txt"

    file_1_1 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\2xP600-2xP240-2xP80\2021-9-28_SNR\100kV\SNR_100kV_4_mm_expTime_365.txt"
    file_2_2 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\4xP5000-2xP600-2xP240-2xP80\2021-9-28_SNR\100kV\SNR_100kV_4_mm_expTime_365.txt"
    file_3_3 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\4xP80\2021-9-28_SNR\100kV\SNR_100kV_4_mm_expTime_365.txt"
    file_4_4 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\without-sandpaper\2021-9-28_SNR\100kV\SNR_100kV_4_mm_expTime_365.txt"

    path_save = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach"
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    _d = 4
    param_1 = r'2xP600-2xP240-2xP80'
    param_2 = r'4xP5000-2xP600-2xP240-2xP80'
    param_3 = r'4xP80'
    param_4 = r'without sandpaper'


    file_1 = np.genfromtxt(file_1, dtype=float, skip_header=3)
    file_2 = np.genfromtxt(file_2, dtype=float, skip_header=3)
    file_3 = np.genfromtxt(file_3, dtype=float, skip_header=3)
    file_4 = np.genfromtxt(file_4, dtype=float, skip_header=3)

    file_1_1 = np.genfromtxt(file_1_1, dtype=float, skip_header=3)
    file_2_2 = np.genfromtxt(file_2_2, dtype=float, skip_header=3)
    file_3_3 = np.genfromtxt(file_3_3, dtype=float, skip_header=3)
    file_4_4 = np.genfromtxt(file_4_4, dtype=float, skip_header=3)

    X_1 = file_1[:, 0]

    ax1_2 = ax1.twiny()
    ax1Ticks = ax1.get_xticks()
    ax1_2Ticks = ax1Ticks[1:]

    ax2_2 = ax2.twiny()
    ax2Ticks = ax2.get_xticks()
    ax2_2Ticks = ax2Ticks[1:]


    def tick_function(x):
        Xx = 1 / (2 * x*10**(-1))
        return ["%.2f" % z for z in Xx]

    ax1_2.set_xticks(ax1_2Ticks)
    ax1_2.set_xbound(ax1.get_xbound())
    ax1_2.set_xticklabels(tick_function(ax1_2Ticks))

    ax2_2.set_xticks(ax2_2Ticks)
    ax2_2.set_xbound(ax2.get_xbound())
    ax2_2.set_xticklabels(tick_function(ax2_2Ticks))

    Y_1 = file_1[:, 1]
    Y_2 = file_2[:, 1]
    Y_3 = file_3[:, 1]
    Y_4 = file_4[:, 1]

    Y_1_1 = file_1_1[:, 1]
    Y_2_2 = file_2_2[:, 1]
    Y_3_3 = file_3_3[:, 1]
    Y_4_4 = file_4_4[:, 1]


    ax1.plot(X_1, Y_1)
    ax1.plot(X_1, Y_2)
    ax1.plot(X_1, Y_3)
    ax1.plot(X_1, Y_4, linestyle='--')

    ax1.set_ylim([10**(-4), 10**(0)])

    ax2.plot(X_1, Y_1_1, label=f'{param_1}')
    ax2.plot(X_1, Y_2_2, label=f'{param_2}')
    ax2.plot(X_1, Y_3_3, label=f'{param_3}')
    ax2.plot(X_1, Y_4_4, label=f'{param_4}', linestyle='--')

    ax2.set_ylim([10 ** (-4), 10 ** (0)])


    ax3 = plt.axes([0.26, 0.56, .2, .35])
    ax3.plot(X_1, Y_1)
    ax3.plot(X_1, Y_2)
    ax3.plot(X_1, Y_3)
    ax3.plot(X_1, Y_4, linestyle='--')
    plt.setp(ax3)

    ax3.set_xlim([0.0, 0.015])
    ax3.set_ylim([50**(-2), 10**(0)])


    ax4 = plt.axes([0.771, 0.56, .2, .35])
    ax4.plot(X_1, Y_1_1)
    ax4.plot(X_1, Y_2_2)
    ax4.plot(X_1, Y_3_3)
    ax4.plot(X_1, Y_4_4, linestyle='--')
    plt.setp(ax4)

    ax4.set_xlim([0.0, 0.015])
    ax4.set_ylim([50 ** (-2), 10 ** (0)])


    ax1.set_title(f'2 mm Al @ 100 kV', fontsize=ttl_size, weight='bold')
    ax1.tick_params(which='major', direction='in', width=_w, length=6, labelsize=lbl_size)
    ax1.tick_params(which='minor', direction='in', width=_w, length=3)
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.set_ylabel('SNR$\cdot s^{-1}$', fontsize=axs_size)
    ax1.set_xlabel('spatial size $[\mu m]$', fontsize=axs_size)
    ax1.set_yscale('log')


    ax2.set_title(f'4 mm Al @ 100 kV', fontsize=ttl_size, weight='bold')
    ax2.tick_params(which='major', direction='in', width=_w, length=6, labelsize=lbl_size)
    ax2.tick_params(which='minor', direction='in', width=_w, length=3)
    ax2.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax2.set_ylabel('SNR$\cdot s^{-1}$', fontsize=axs_size)
    ax2.set_xlabel('spatial frequency', fontsize=axs_size)
    ax2.legend(fontsize=lgnd_size, loc='center right', bbox_to_anchor=(0.97, 0.35))
    ax2.set_yscale('log')


    ax3.tick_params(which='major', direction='in', width=_w, length=6, labelsize=lbl_size)
    ax3.tick_params(which='minor', direction='in', width=_w, length=3)
    ax3.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax3.set_yscale('log')

    ax4.tick_params(which='major', direction='in', width=_w, length=6, labelsize=lbl_size)
    ax4.tick_params(which='minor', direction='in', width=_w, length=3)
    ax4.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax4.set_yscale('log')


    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(_w)
        ax1_2.spines[axis].set_linewidth(_w)
        ax2.spines[axis].set_linewidth(_w)
        ax2_2.spines[axis].set_linewidth(_w)
        ax3.spines[axis].set_linewidth(_w)
        ax4.spines[axis].set_linewidth(_w)

    fig.tight_layout(rect=(0, 0, 1, 1), pad=0.3)
    fig.subplots_adjust(wspace=2.5, hspace=1.5)
    plt.show()
    fig.savefig(os.path.join(path_save, f'Compare_2mm-4mm_100kV.pdf'), dpi=600)


def coin_wedge_compare():
    file_1 = r""
    file_2 = r""
    file_1 = np.genfromtxt(file_1, dtype=float, skip_header=3)
    file_2 = np.genfromtxt(file_2, dtype=float, skip_header=3)
    X = file_1[:, 0]
    Y_1 = file_1[:, 1]
    Y_2 = file_2[:, 1]

    fig = plt.figure(figsize=(12, 5))




import os
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.gridspec as gridspec



class Plotter:
    def __init__(self):
        self.ft_sz_size = 10
        self.ft_sz_title = 10


    def create_plot_v1(self, path_result: str, act_object, ds: list, Y_style: str = 'log'):
        fig = plt.figure()
        ax = fig.add_subplot()

        for _c in act_object.curves:
            if _c.d in ds:
                color=TerminalColor.flat_gray
                data_size = 40
                ax.text(_c.T[0] - 0.05, _c.SNR[0], f'{int(_c.d)}mm')
                _alpha = 0.7
                linew = 3
            else:
                color = TerminalColor.flat_gray
                data_size = 15
                _alpha = 0.2
                linew = 1

            plt.scatter(_c.T, _c.SNR, label=f'{_c.d}mm', marker='o', alpha=_alpha, c=color, s=data_size)
            a, b, c = np.polyfit(_c.T, _c.SNR, deg=2)
            x = np.linspace(_c.T[0], _c.T[-1], 141)
            y = self.func_poly(x, a, b, c)
            if _c.d == act_object.d_opt:
                plt.plot(x, y, c=TerminalColor.flat_red, alpha=0.9, linewidth=3)
            plt.plot(x, y, c=color, alpha=_alpha, linewidth=linew)

        ax.text(0, 0, '$kV_opt = {}$'.format(act_object.kV_opt))
        plt.scatter(act_object.X_opt, act_object.Y_opt, marker='x', c='black', s=70)
        plt.title(f'$SRN(T)$ with $U_{0} = {act_object.U0}$kV       FIT: $f(x) = a x^{2} + bx + c$')
        plt.xlabel('Transmission a.u.')
        plt.ylabel('SNR/s')
        plt.xlim(act_object.curves[-1].T[0] - 0.05, act_object.curves[0].T[-1] + 0.02)
        plt.plot(act_object.x_U0_c, act_object.y_U0_c, c=TerminalColor.flat_green, linestyle='--', linewidth=2)
        plt.axvline(x=act_object.T_min, c=TerminalColor.flat_green, linestyle='--', alpha=0.5, linewidth=2)
        plt.scatter(act_object.intercept_x, act_object.intercept_y, c=TerminalColor.flat_red, marker='x', s=50)
        plt.yscale(Y_style)
        plt.show()
        fig.savefig(os.path.join(path_result, f'MAP_kVopt{act_object.kV_opt}.pdf'), dpi=600)



    def create_plot(self, path_result: str, object, Y_style: str = 'log'):
        fig = plt.figure()
        roi_l = object['ROIs']['lb']
        roi_r = object['ROIs']['rb']

        for d in object['d_curves']:
            d_wu = self.rm_underscore(d)
            _c = object['curves'][d]
            plt.plot(_c[:, 1], _c[:, 2], linestyle='-', label=d_wu)
            plt.scatter(_c[:, 1], _c[:, 2], marker='o',)
        plt.legend()
        plt.title(f'SNR MAP @ {roi_l}-{roi_r} $\mu m$')
        plt.yscale(Y_style)
        plt.xlabel('Transmission a.u.')
        plt.ylabel('SNR/s')
        sv_path = os.path.join(path_result, 'plots')
        if not os.path.isdir(sv_path):
            os.makedirs(sv_path)
        fig.savefig(os.path.join(path_result, 'plots', f'MAP_ROI-{roi_l}-{roi_r}.pdf'), dpi=600)


    def create_v_plot(self, path_result: str, object, Y_style: str = 'log', full: bool = False):
        fig = plt.figure()
        ax = fig.add_subplot()
        #ax2 = plt.twiny()
        roi_l = object['ROIs']['lb']
        roi_r = object['ROIs']['rb']

        sorted_curves = dict(sorted(object['d_curves'].items()))
        for d in sorted_curves:
            _c_fit = sorted_curves[d]['fit']
            _c_data = sorted_curves[d]['data']
            _c_max_idx = sorted_curves[d]['max_idx']

            is_int = self.check_ds_nature(d)


            if is_int and d in object['ds']:
                ax.scatter(_c_fit[:, 1][_c_max_idx], _c_fit[:, 2][_c_max_idx], marker='x', alpha=1.0, s=15, c='grey')
                ax.scatter(_c_fit[:, 1], _c_fit[:, 2], marker='o', alpha=0.7)
                #ax.plot(_c_fit[:, 1], _c_fit[:, 2], linestyle='-', linewidth='2', alpha=1.0, label=f'{d} mm')
                ax.scatter(_c_data[:, 1], _c_data[:, 2], marker='o', alpha=1.0)

            else:
                ax.scatter(_c_fit[:, 1][_c_max_idx], _c_fit[:, 2][_c_max_idx], marker='x', alpha=0.8, s=10, c='grey')
                ax.plot(_c_data[:, 1], _c_data[:, 2], linestyle='-', linewidth=1, alpha=0.15, c='grey')


        if full and object['intercept_found']:
            tt = ax.axvline(x=object['T_min'], c='green', linestyle='--', alpha=0.5, linewidth=1)


        _c_U0 = object['U0_curve']['fit']
        _c_opt = object['opt_curve']['fit']
        ax.plot(_c_U0[:, 0], _c_U0[:, 1], linewidth=1.5, label='$U_{0}$ curve')
        ax.plot(_c_opt[:, 0], _c_opt[:, 1],linewidth=1.5, label='$U_{opt}$ curve', c='red')

        ax.legend(loc="upper left")
        ax.set_title(f'SNR MAP @ {roi_l}-{roi_r} $\mu m$')
        ax.set_yscale(Y_style)
        ax.set_xlabel('Transmission a.u.')
        ax.set_ylabel('SNR/s')
        sv_path = os.path.join(path_result, 'plots')
        if not os.path.isdir(sv_path):
            os.makedirs(sv_path)
        plt.show()
        fig.savefig(os.path.join(path_result, 'plots', f'MAP_ROI-{roi_l}-{roi_r}.pdf'), dpi=600)


    def fit_me_for_plot(self, _c):
        a, b, c = np.polyfit(_c[:, 1], _c[:, 2], deg=2)
        x = np.linspace(_c[:, 1][0], _c[:, 1][-1], 141)
        y = self.func_poly(x, a, b, c)
        return x, y


    def create_evolution_plot(self,path_snr:str, path_T:str, path_result: str, object: object, d: int, spatial_list: list = None, Y_style: str = 'log'):
        if spatial_list is not None:
            sizes = spatial_list
        else:
            sizes = [250, 150, 100, 50, 10]

        base_MAP = SNRMapGenerator(path_snr=path_snr, path_T=path_T, path_fin=path_result, d=[d])

        MAP_dict = {}
        for spt_s in sizes:
            MAP_dict[f'{spt_s}'] = base_MAP.create_MAP(spatial_range=spt_s)['curves'][f'{d}_mm']

        fig = plt.figure()
        for _s in sizes:
            ROI = MAP_dict[f'{_s}']
            x = MAP_dict[f'{_s}']['T']
            y = MAP_dict[f'{_s}']['SNR']
            plt.plot(x, y, linestyle='-', label=f'{_s} $\mu m$')
            plt.scatter(x, y, marker='o', cmap='RdPu')
        plt.legend()
        plt.yscale(Y_style)
        plt.xlabel('Transmission a.u.')
        plt.ylabel('SNR/s')
        plt.title(f'SNR/s (T) for different spatial sizes (@{d} mm Al)')
        plt.show()
        fig.savefig(os.path.join(path_result, f'evo_MAP_.pdf'), dpi=600)
        print('test')



    def compare_plot(self):
        '''
        Method for comparing of data from different measurements
        '''

        plt.rc('lines', linewidth=2)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm'])))

        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(15,6))

        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[1:, 0])
        ax2 = plt.subplot(gs[:, 1:])

        path1 = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\2021-08-30_Evaluation\Eval_Result'
        path2 = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\2021-09-18_Evaluation\Eval_Result'

        db1 = DB(path1)
        db2 = DB(path2)

        ds = [1, 4, 5, 8, 9]
        colors = ['r', 'g', 'b', 'y', 'm']
        for _d, color in zip(ds, colors):
            old_alpha = 0.2
            new_alpha = 1.0

            V_1, T_1, SNR_1 = db1.read_data(d=_d, mode='raw')
            V_2, T_2, SNR_2 = db2.read_data(d=_d, mode='raw')

            ax0.scatter(T_1, SNR_1, marker='x', s=40, alpha=old_alpha)
            ax1.scatter(T_2, SNR_2, marker='x', s=40, label=f'{_d}mm new')
            ax2.scatter(T_1, SNR_1, marker='x', s=40, c=color, alpha=old_alpha)
            ax2.scatter(T_2, SNR_2, marker='x', s=40, c=color, alpha=new_alpha)


            a_1, b_1, c_1 = np.polyfit(T_1, SNR_1, deg=2)
            a_2, b_2, c_2 = np.polyfit(T_2, SNR_2, deg=2)

            x_1 = np.linspace(T_1[0], T_1[-1], 141)
            y_1 = self.func_poly(x_1, a_1, b_1, c_1)
            x_2 = np.linspace(T_2[0], T_2[-1], 141)
            y_2 = self.func_poly(x_2, a_2, b_2, c_2)

            ax0.plot(x_1, y_1, alpha=0.5)
            ax1.plot(x_2, y_2)
            ax2.plot(x_1, y_1, c=color, alpha=old_alpha)
            ax2.plot(x_2, y_2, c=color, label=f'{_d}mm')

        #ax0.grid()
        ax0.set_title('Measurement @ 30.08.2021')
        ax0.tick_params(which='major', direction='in', width=1.2, length=6)
        ax0.tick_params(which='minor', direction='in', width=1.2, length=2.5)
        ax0.set_yscale('log')

        ax1.grid()
        ax1.set_title('Measurement @ 17.09.2021')
        ax1.tick_params(which='major', direction='in', width=1.2, length=6)
        ax1.tick_params(which='minor', direction='in', width=1.2, length=2.5)
        ax1.set_yscale('log')

        #ax2.grid()
        ax2.set_title('30.08.2021 + 17.09.2021')
        ax2.tick_params(which='major', direction='in', width=1.2, length=6)
        ax2.tick_params(which='minor', direction='in', width=1.2, length=2.5)
        ax2.legend()
        ax2.set_yscale('log')


        for axis in ['top', 'bottom', 'left', 'right']:
            ax0.spines[axis].set_linewidth(1.2)
            ax1.spines[axis].set_linewidth(1.2)
            ax2.spines[axis].set_linewidth(1.2)

        plt.legend()
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.3)
        plt.show()
        fig.savefig(os.path.join(path2, f'MAP_compare.pdf'), dpi=600)

    def rm_underscore(self, d):
        d = d.replace('_', ' ')
        d = float(d)
        return d

    @staticmethod
    def check_ds_nature(var):
        if var % 1.0 == 0.0:
            return True
        else:
            return False

    @staticmethod
    def func_poly(x, a, b, c):
        return a * x ** 2 + b * x + c


def sandpaper_test():
    #plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
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

    file_1_1 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-30-Sergej_SNR-Stufenkeil_200proj_6W_Sndppr-test\2xP600-2xP240-2xP80\2021-9-30_SNR\100kV\SNR_100kV_4_mm_expTime_375.txt"
    #file_2_2 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\4xP5000-2xP600-2xP240-2xP80\2021-9-28_SNR\100kV\SNR_100kV_4_mm_expTime_365.txt"
    file_3_3 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-30-Sergej_SNR-Stufenkeil_200proj_6W_Sndppr-test\4xP80\2021-9-30_SNR\100kV\SNR_100kV_4_mm_expTime_375.txt"
    file_4_4 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach\without-sandpaper\2021-9-28_SNR\100kV\SNR_100kV_4_mm_expTime_365.txt"
    file_5_5 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-30-Sergej_SNR-Stufenkeil_200proj_6W_Sndppr-test\4xP240\2021-9-30_SNR\100kV\SNR_100kV_4_mm_expTime_375.txt"

    path_save = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-27-Sergej_SNR-Stufenkeil_130proj_6W_Sndppr-test\new_approach"
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    _d = 4
    param_1 = r'2xP600-2xP240-2xP80'
    param_2 = r'4xP5000-2xP600-2xP240-2xP80'
    param_3 = r'4xP80'
    param_4 = r'without sandpaper'
    param_5 = r'4xP240'


    file_1 = np.genfromtxt(file_1, dtype=float, skip_header=3)
    file_2 = np.genfromtxt(file_2, dtype=float, skip_header=3)
    file_3 = np.genfromtxt(file_3, dtype=float, skip_header=3)
    file_4 = np.genfromtxt(file_4, dtype=float, skip_header=3)

    file_1_1 = np.genfromtxt(file_1_1, dtype=float, skip_header=3)
    #file_2_2 = np.genfromtxt(file_2_2, dtype=float, skip_header=3)
    file_3_3 = np.genfromtxt(file_3_3, dtype=float, skip_header=3)
    file_4_4 = np.genfromtxt(file_4_4, dtype=float, skip_header=3)
    file_5_5 = np.genfromtxt(file_5_5, dtype=float, skip_header=3)

    u = file_1[:, 0]
    X_1 = file_1[:, 0]
    #X_1 = 1 / (2 * u)

    #ax1_2 = ax1.twiny()
    #ax1Ticks = ax1.get_xticks()
    #ax1Ticks = ax1Ticks[1:]

    #ax2_2 = ax2.twiny()
    #ax2Ticks = ax2.get_xticks()
    #ax2_2Ticks = ax2Ticks[1:]


    def tick_function(x):
        Xx = 1 / (2 * x*10**(-1))
        return ["%.2f" % z for z in Xx]

    #ax1.set_xticks(ax1Ticks)
    #ax1.set_xbound(ax1.get_xbound())
    #ax1.set_xticklabels(tick_function(ax1Ticks))

    #ax2_2.set_xticks(ax2_2Ticks)
    #ax2_2.set_xbound(ax2.get_xbound())
    #ax2_2.set_xticklabels(tick_function(ax2_2Ticks))

    Y_1 = file_1[:, 1]
    Y_2 = file_2[:, 1]
    Y_3 = file_3[:, 1]
    Y_4 = file_4[:, 1]

    Y_1_1 = file_1_1[:, 1]
    #Y_2_2 = file_2_2[:, 1]
    Y_3_3 = file_3_3[:, 1]
    Y_4_4 = file_4_4[:, 1]
    Y_5_5 = file_5_5[:, 1]

    ax1.plot(X_1, Y_1)
    ax1.plot(X_1, Y_2)
    ax1.plot(X_1, Y_3)
    ax1.plot(X_1, Y_4, linestyle='--')
    ax1.set_ylim([10**(-4), 10**(0)])

    ax2.plot(X_1, Y_1_1, label=f'{param_1}')
    #ax2.plot(X_1, Y_2_2, label=f'{param_2}')
    ax2.plot(X_1, Y_3_3, label=f'{param_3}')
    ax2.plot(X_1, Y_4_4, label=f'{param_4}', linestyle='--')
    ax2.plot(X_1, Y_5_5, label=f'{param_5}')
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
    #ax4.plot(X_1, Y_2_2)
    ax4.plot(X_1, Y_3_3)
    ax4.plot(X_1, Y_4_4, linestyle='--')
    ax4.plot(X_1, Y_5_5)
    plt.setp(ax4)
    ax4.set_xlim([0.0, 0.02])
    ax4.set_ylim([10 ** (-3), 10 ** (0)])


    ax1.set_title(f'2 mm Al @ 100 kV', fontsize=ttl_size, weight='bold')
    ax1.tick_params(which='major', direction='in', width=_w, length=6, labelsize=lbl_size)
    ax1.tick_params(which='minor', direction='in', width=_w, length=3)
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.set_ylabel('SNR$\cdot s^{-1}$', fontsize=axs_size)
    ax1.set_xlabel('spatial frequency', fontsize=axs_size)
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
        #ax1_2.spines[axis].set_linewidth(_w)
        ax2.spines[axis].set_linewidth(_w)
        #ax2_2.spines[axis].set_linewidth(_w)
        ax3.spines[axis].set_linewidth(_w)
        ax4.spines[axis].set_linewidth(_w)

    fig.tight_layout(rect=(0, 0, 1, 1), pad=0.3)
    fig.subplots_adjust(wspace=2.5, hspace=1.5)
    plt.show()
    fig.savefig(os.path.join(path_save, f'Compare_2mm-4mm_100kV.pdf'), dpi=600)


def focal_point():
    path_save = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\Brennfleck"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    file_1 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-28_Brennfleck\1W\2021-9-30_SNR\100kV\SNR_100kV_4_mm_expTime_2250.txt"
    file_2 = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-09-28_Brennfleck\6W\2021-9-30_SNR\100kV\SNR_100kV_4_mm_expTime_375.txt"

    file_1 = np.genfromtxt(file_1, dtype=float, skip_header=3)
    file_2 = np.genfromtxt(file_2, dtype=float, skip_header=3)


    X = file_1[:, 0]
    Y_1 = file_1[:, 1]
    Y_2 = file_2[:, 1]

    fig = plt.figure()
    plt.semilogy(X, Y_1, label='1W $t_{exp} = 2250$ms')
    plt.semilogy(X, Y_2, label='6W $t_{exp} = 375$ms')
    plt.xlim(0, 0.1)
    plt.title(f'Compare of 1W and 6W SNR/s')
    plt.xlabel('spatial frequency')
    plt.ylabel('SNR $\cdot s^{-1}$')
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(path_save, f'compare_1W_6W.pdf'), dpi=600)



def get_color_ls(index):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color = colors[index%len(colors)]
    ls = ['-', '--', ':'][(index//len(colors)) % 3]
    return color, ls


class TerminalColor:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    flat_gray = '#d3d3d3'
    flat_red = '#D56489'
    flat_yellow = '#ECE6A6'
    flat_blue = '#009D9D'
    flat_green = '#41BA90'

import os
import numpy as np
import helpers as hlp
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.gridspec as gridspec


class Plotter:
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

    custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color',
                                                              ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97',
                                                               '#474747', '#9e9e9e'])
    plt.rc('axes', prop_cycle=custom_cycler)


    def map_plot(self, path_result: str, object, Y_style: str = 'log', detailed: bool = False):
        plt.rcParams["figure.figsize"] = (8.77, 5.3)
        plt.rc('lines', linewidth=2)

        fig = plt.figure()
        ax = fig.add_subplot()
        roi_l = object['ROIs']['lb']
        roi_r = object['ROIs']['rb']

        for d in object['d_curves']:
            _c_fit = object['d_curves'][d]['full']

            if hlp.is_int(d) and d in object['ds']:
                _c_data = object['d_curves'][d]['raw_data']
                _a = 1.0
                ax.plot(_c_fit[:, 1], _c_fit[:, 3], zorder=10, alpha=_a, label=f'{d} mm')
                ax.scatter(_c_data[:, 1], _c_data[:, 2], zorder=15, marker='o', alpha=_a)

            if d == 4:
                ax.annotate(r'\SI{50}{\kilo\volt}', xy=(_c_data[:, 1][0], _c_data[:, 2][0]), xytext=(-30, 20),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='k'))
                ax.annotate(r'\SI{160}{\kilo\volt}', xy=(_c_data[:, 1][-1], _c_data[:, 2][-1]), xytext=(20, -30),
                             textcoords='offset points', ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2', fc='None', edgecolor='#BBBBBB', alpha=1),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                             color='k'))

            else:
                pass
                #_c = '#BBBBBB'
                #_a = 0.5
                #ax.plot(_c_fit[:, 1], _c_fit[:, 3], linestyle='-', zorder=5, linewidth=0.8, alpha=_a, c=_c)

        #ax.axvline(x=object['T_min'], color='k', linestyle='--', linewidth=1)
        _c_U0 = object['U0_curve']['raw_data']
        _c_Ubest = object['Ubest_curve']['raw_data']
        #ax.plot(_c_U0[:, 0], _c_U0[:, 1], linewidth=1.5, label=r'$U_{0}$')
        ax.plot(_c_Ubest[:, 0], _c_Ubest[:, 1], linewidth=1.5, linestyle='-', c='k', label=r'$U_{\text{opt}}$')
        ax.legend(loc="lower right")
        #ax.set_title(f'AL (Z=13) @ {roi_l}-{roi_r} $\mu m$')
        ax.annotate(r'Aluminium (Z=13)', xy=(0.0, 1800), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                     ha='left', va='center')

        ax.annotate(rf'SNR gemittl. $\in$ [{roi_l},{roi_r}]' + r' \SI{}{\micro\meter}', xy=(0.0, 1000),
                    bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                    ha='left', va='center')

        ax.annotate(r'$U \in [50, 60, 80, 90, 100, 120, 130, 150, 160]$ \SI{}{\kilo\volt}', xy=(0.0, 555),
                    bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                    ha='left', va='center')


        ax.set_yscale(Y_style)
        ax.set_xlabel('Transmission [w.E.]')
        ax.set_ylabel(r'$\text{SNR}$ [$s^{-1}$]')
        sv_path = os.path.join(path_result, 'plots')
        plt.tight_layout()
        if not os.path.isdir(sv_path):
            os.makedirs(sv_path)
        fig.savefig(os.path.join(path_result, f'SNR-Karte-ROI-{roi_l}-{roi_r}.pdf'), bbox_inches="tight", dpi=600)



    def T_kv_plot(self, path_result: str, object, Y_style: str = 'log', detailed: bool = False):
        roi_l = object['ROIs']['lb']
        roi_r = object['ROIs']['rb']

        fig = plt.figure()
        ax = fig.add_subplot()

        for d in object['d_curves']:

            _c_fit = object['d_curves'][d]['full']
            _c_data = object['d_curves'][d]['raw_data']

            if hlp.is_int(d) and d in object['ds']:
                _a = 1.0
                ax.plot(_c_fit[:, 0], _c_fit[:, 1], linestyle='-', linewidth='2', alpha=_a, label=f'{d} mm')
                ax.scatter(_c_data[:, 0], _c_data[:, 1], marker='o', alpha=_a)
            else:
                _c = 'grey'
                _a = 0.2
                ax.plot(_c_fit[:, 0], _c_fit[:, 1], linestyle='-', linewidth='1', alpha=_a, c=_c)

        ax.legend(loc='best')
        ax.set_yscale(Y_style)
        ax.set_xlabel(r'Voltage $[ \SI{1e3}{ \volt}]$')
        ax.set_ylabel('Transmission [w.E.]')
        sv_path = os.path.join(path_result, 'plots')
        if not os.path.isdir(sv_path):
            os.makedirs(sv_path)
        plt.tight_layout()
        fig.savefig(os.path.join(path_result, 'plots', f'T_kV-{roi_l}-{roi_r}.pdf'), dpi=600)


    def snr_kv_plot(self, path_result: str, object, Y_style: str = 'log', detailed: bool = False):
        roi_l = object['ROIs']['lb']
        roi_r = object['ROIs']['rb']

        fig = plt.figure()
        ax = fig.add_subplot()

        for d in object['d_curves']:

            _c_fit = object['d_curves'][d]['full']
            _c_data = object['d_curves'][d]['raw_data']

            if hlp.is_int(d) and d in object['ds']:
                _a = 1.0
                ax.plot(_c_fit[:, 0], _c_fit[:, 3], linestyle='-', linewidth='2', alpha=_a, label=f'{d} mm')
                ax.scatter(_c_data[:, 0], _c_data[:, 2], marker='o', alpha=_a)
            else:
                _c = 'grey'
                _a = 0.2
                ax.plot(_c_fit[:, 0], _c_fit[:, 3], linestyle='-', linewidth='1', alpha=_a, c=_c)

        ax.legend(loc='best')
        ax.set_yscale(Y_style)
        ax.set_xlabel(r'Voltage $[ \SI{1e3}{ \volt}]$')
        ax.set_ylabel('SNR [$s^{-1}$]')
        sv_path = os.path.join(path_result, 'plots')
        if not os.path.isdir(sv_path):
            os.makedirs(sv_path)
        plt.tight_layout()
        fig.savefig(os.path.join(path_result, 'plots', f'snr_kV-{roi_l}-{roi_r}.pdf'), dpi=600)


    def compare_fCT_CT(self, object):
        fig, (ax1, ax2, ax3) = plt.subplots(3)

        ROIl, ROIr = object.map['ROIs']['lb'], object.map['ROIs']['rb']

        self.object = object
        CT_avg = object.CT_data['avg_num']
        CT_theta = object.CT_data['theta']
        CT_T = object.CT_data['T']
        CT_snr = object.CT_data['snr']
        CT_d = object.CT_data['d']
        CT_texp = object.CT_data['texp']

        fCT_avg = object.fCT_data['avg_num']
        fCT_theta = object.fCT_data['theta']
        fCT_T = object.fCT_data['T']
        fCT_snr = object.fCT_data['snr']
        fCT_d = object.fCT_data['d']
        fCT_texp = object.fCT_data['texp']

        ax11 = ax1.twinx()
        ax1.scatter(CT_theta, CT_avg, label='CT_avg')
        ax11.scatter(fCT_theta, fCT_avg, label='fCT_avg')
        ax1.legend()
        ax11.legend()
        ax1.set_xlabel(r'Winkel $\theta [\circ]$')
        ax1.set_ylabel('Gemittelte Projektionen')

        ax2.scatter(CT_theta, CT_snr, label=r'$SNR(\theta)$')
        ax2.scatter(fCT_theta, fCT_snr, label=r'$fSNR(\theta)$')
        ax2.legend()
        ax2.set_xlabel(r'Winkel $\theta [\circ]$')
        ax2.set_ylabel(r'SNR $\cdot s^{-1}$')

        ax3.scatter(CT_theta, CT_d, label=f'CT d')
        ax3.scatter(fCT_theta, fCT_d, label=f'fCT d')
        ax3.legend()
        ax3.set_xlabel(r'Winkel $\theta [\circ]$')
        ax3.set_ylabel(r'Berechnete Objektdicke')

        plt.tight_layout()
        plt.savefig(os.path.join(object.p_fin, 'Overview_plot_ROI{ROIl}-{ROIr}.pdf'), dpi=600)


    def fit_me_for_plot(self, _c):
        a, b, c = np.polyfit(_c[:, 1], _c[:, 2], deg=2)
        x = np.linspace(_c[:, 1][0], _c[:, 1][-1], 141)
        y = self.func_poly(x, a, b, c)
        return x, y


    def rm_underscore(self, d):
        d = d.replace('_', ' ')
        d = float(d)
        return d

    @staticmethod
    def func_poly(x, a, b, c):
        return a * x ** 2 + b * x + c



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

import numpy as np
#from matplotlib import figure
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from cycler import cycler


def plot():
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
    plt.rcParams["figure.figsize"] = (6.3, 4)

    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=(cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])))

    density_AL = 2.7    #g/cm^3
    data_tot = np.genfromtxt(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\mu_abs_al\nist_abs_data.txt', skip_header=3)
    x_tot = data_tot[:, 0]
    y_tot = data_tot[:, 1] * density_AL

    data_photo_abs = np.genfromtxt(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\mu_abs_al\ph_abs.txt', skip_header=3)
    x_ph_abs = data_photo_abs[:, 0]
    y_ph_abs = data_photo_abs[:, 1] * density_AL


    data_pair = np.genfromtxt(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\mu_abs_al\pair_prod.txt', skip_header=3)
    x_pair = data_pair[:, 0]
    y_pair_1 = data_pair[:, 1] * density_AL
    y_pair_2 = data_pair[:, 2] * density_AL
    y_pair = y_pair_1 + y_pair_2


    data_scatter = np.genfromtxt(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\mu_abs_al\scatter.txt', skip_header=3)
    x_scatter = data_scatter[:, 0]
    y_scatter_1 = data_scatter[:, 1] * density_AL
    y_scatter_2 = data_scatter[:, 2] * density_AL


    sum = y_scatter_2 + y_pair + y_ph_abs

    plt.plot(x_scatter, y_scatter_2, c='#00B945', label=r'$\mu_{\text{Compton}}$')
    plt.plot(x_pair, y_pair, c='#FF2C00', label=r'$\mu_{\text{Paar}}')
    plt.plot(x_ph_abs, y_ph_abs, c='#0C5DA5', label=r'$\mu_{\text{Photo}}')
    plt.plot(x_tot, sum, linewidth=3, c='#474747', label=r'$\mu_{\text{Gesamt}}$')


    #plt.text(x=0.08, y=300, s='Aluminium (Z=13)')
    plt.annotate(r'Aluminium (Z=13)', xy=(0.2, 250), bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB"),
                 ha='center', va='bottom')

    plt.gca().add_patch(Rectangle((x_tot[0], 0.00008), 0.018, 1100, color='#BBBBBB', alpha=0.8))

    plt.ylabel(r'Absorptionskoeffizient $\mu$ ($\text{cm}^{-1}$)')
    plt.xlabel(r'Photonenenergie $h\nu$ ($\SI{e6}{\eV}$)')

    plt.ylim(0.0001)
    plt.xlim(x_tot[0], 10)
    plt.margins(x=0)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\mu_abs_al\mu.pdf', bbox_inches='tight', dpi=600)


plot()
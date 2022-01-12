import numpy as np
#from matplotlib import figure
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def plot():
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 15,  # LaTeX default is 10pt font.
        "font.size": 15,
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
    plt.rcParams["figure.figsize"] = (8.3, 4)


    density_AL = 2.7    #g/cm^3
    data_tot = np.genfromtxt(r'/man_evaluations/mu_abs_al/nist_abs_data.txt', skip_header=3)
    x_tot = data_tot[:, 0]
    y_tot = data_tot[:, 1] * density_AL

    data_photo_abs = np.genfromtxt(r'/man_evaluations/mu_abs_al/ph_abs.txt', skip_header=3)
    x_ph_abs = data_photo_abs[:, 0]
    y_ph_abs = data_photo_abs[:, 1] * density_AL


    data_pair = np.genfromtxt(r'/man_evaluations/mu_abs_al/pair_prod.txt', skip_header=3)
    x_pair = data_pair[:, 0]
    y_pair_1 = data_pair[:, 1] * density_AL
    y_pair_2 = data_pair[:, 2] * density_AL
    y_pair = y_pair_1 + y_pair_2


    data_scatter = np.genfromtxt(r'/man_evaluations/mu_abs_al/scatter.txt', skip_header=3)
    x_scatter = data_scatter[:, 0]
    y_scatter_1 = data_scatter[:, 1] * density_AL
    y_scatter_2 = data_scatter[:, 2] * density_AL


    sum = y_scatter_2 + y_pair + y_ph_abs
    plt.plot(x_scatter, y_scatter_2, linewidth=1.5, c='#9F84BD', label=r'$\mu_{\mathrm{Compton}}$')
    plt.plot(x_pair, y_pair, linewidth=1.5, c='#FA7E61', label=r'$\mu_{\mathrm{Paar}}')
    plt.plot(x_ph_abs, y_ph_abs, linewidth=1.5, c='#721121', label=r'$\mu_{\mathrm{Photo}}')
    plt.plot(x_tot, sum, linewidth=3, label=r'$\mu_{\mathrm{tot}}$', c='#090446')

    plt.text(x=0.08, y=300, s='Aluminium (Z=13)')

    plt.gca().add_patch(Rectangle((x_tot[0], 0.00008), 0.018, 1100, color='#F4F4F6'))

    plt.ylabel(r'\textbf{Absorptionskoeffizient} $\mu$ ($\mathrm{cm}^{-1}$)')
    plt.xlabel(r'\textbf{Photonenenergie} $h\nu$ ($\SI{e6}{\eV}$)')

    plt.ylim(0.0001)
    plt.xlim(x_tot[0], 10)
    plt.margins(x=0)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mu.pdf', bbox_inches='tight', dpi=600)


plot()
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cycler import cycler
import numpy as np


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

custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
plt.rc('axes', prop_cycle=custom_cycler)



def CT_mean_compare_full():
    #plt.rcParams["figure.figsize"] = (6.29, 2.7)  # 6.29, 2.5
    plt.rcParams["figure.figsize"] = (5.6, 2.9)

    plt.rc('lines', linewidth=2)

    data_path = r"XXX"
    for dirname in os.listdir(data_path):
         if os.path.isdir(os.path.join(data_path, dirname)):
            for file in os.listdir(os.path.join(data_path, dirname)):
                if file.endswith('.csv'):
                    orientation = dirname.split('-slice')[0]
                    slice = dirname.split('-')[1]
                    sv_name = rf'{orientation}-{slice}'
                    sv_path = r'XXX'


                    data = np.genfromtxt(os.path.join(data_path, dirname, file), skip_header=1, delimiter=',')
                    fig, ax1 = plt.subplots()
                    dist = data[:, 0]
                    mean1 = data[:, 1]
                    mean2 = data[:, 2]
                    mean3 = data[:, 3]
                    mean_var = data[:, 4]

                    alpha = 0.5

                    plt.plot(dist, mean1, label='$m = 1$', alpha=alpha, linestyle='--')
                    plt.plot(dist, mean2, label='$m = 2$')
                    plt.plot(dist, mean3, label='$m = 3$', alpha=alpha, linestyle='--')
                    plt.plot(dist, mean_var, label=r'$m = m(\theta)\Bigr|_{m_{\text{min}}=1}^{m_{\text{max}}=3}$')
                    plt.ylabel('Grauwert [w.E.]')
                    plt.xlabel('Position [px]')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(sv_path, sv_name + '.pdf'), bbox_inches='tight', dpi=600)




def CT_mean_compare():
    plt.rcParams["figure.figsize"] = (5.6, 2.9)  # 6.29, 7.9
    plt.rc('lines', linewidth=2)


    data_path1 = r"XXX"
    data_path2 = r"XXX"
    data_path3 = r"XXX"

    sv_name = r'KurzeSeite'
    sv_path = r'XXX'

    data1 = np.genfromtxt(data_path1, skip_header=1, delimiter=',')
    data2 = np.genfromtxt(data_path2, skip_header=1, delimiter=',')
    data3 = np.genfromtxt(data_path3, skip_header=1, delimiter=',')


    alpha = 0.5

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    data1_dist = data1[:, 0]
    data1_mean1 = data1[:, 1]
    data1_mean2 = data1[:, 2]
    data1_mean3 = data1[:, 3]
    data1_mean_var = data1[:, 4]

    data2_dist = data2[:, 0]
    data2_mean1 = data2[:, 1]
    data2_mean2 = data2[:, 2]
    data2_mean3 = data2[:, 3]
    data2_mean_var = data2[:, 4]

    data3_dist = data3[:, 0]
    data3_mean1 = data3[:, 1]
    data3_mean2 = data3[:, 2]
    data3_mean3 = data3[:, 3]
    data3_mean_var = data3[:, 4]


    ax1.plot(data1_dist, data1_mean1, alpha=alpha, linestyle='--')
    ax1.plot(data1_dist, data1_mean2)
    ax1.plot(data1_dist, data1_mean3, alpha=alpha, linestyle='--')
    ax1.plot(data1_dist, data1_mean_var)
    ax1.set_ylabel(r'Grauwert [w. E.]')

    ax2.plot(data2_dist, data2_mean1, alpha=alpha, linestyle='--')
    ax2.plot(data2_dist, data2_mean2)
    ax2.plot(data2_dist, data2_mean3, alpha=alpha, linestyle='--')
    ax2.plot(data2_dist, data2_mean_var)
    ax2.set_ylabel(r'Grauwert [w. E.]')

    ax3.plot(data3_dist, data3_mean1, label='mean 1', alpha=alpha, linestyle='--')
    ax3.plot(data3_dist, data3_mean2, label='mean 2',)
    ax3.plot(data3_dist, data3_mean3, label='mean 3', alpha=alpha, linestyle='--')
    ax3.plot(data3_dist, data3_mean_var, label='mean var',)
    ax3.set_xlabel(r'Position [px]')
    ax3.set_ylabel(r'Grauwert [w. E.]')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(sv_path, sv_name + '.pdf'), bbox_inches='tight', dpi=600)



#CT_mean_compare()
CT_mean_compare_full()



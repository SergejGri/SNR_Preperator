import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os



def lineplot():
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
    plt.rcParams["figure.figsize"] = (3.3, 1.2)
    plt.rc('lines', linewidth=1)
    #plt.rc('axes', prop_cycle=(cycler('color', ['#332288', '#117733', '#44AA99', '#882255', '#999933', '#88CCEE', '#DDCC77', '#CC6677', '#882255', '#AA4499', '#BBBBBB'])))
    #plt.rc('axes',prop_cycle=(cycler('color', [#102694,   '#469D59',  '#EE7972', '#E2B43C', '#CC7444', '#002C2B',  '#3372F5', '#FDEADB', '#BBBBBB'])))
    #                                           dunkelblau  hellgrün      rot        gelb      orange    dunkelgrün   BLAU        beige       grau
    fig = plt.subplots()



    sv_path = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\lineplot'

    data = np.genfromtxt(r"C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\lineplot\Values_detektor_linie.csv", skip_header=1, delimiter=',')

    x = data[:, 0]
    y = data[:, 1]

    plt.plot(x, y, c='k')
    plt.xlim(0, 232)
    plt.ylim(3.3, 3.325)
    plt.tight_layout()
    #plt.xlabel('Pixelposition')
    #plt.ylabel('Intensität [w. E.]')
    plt.savefig(os.path.join(sv_path, 'lineplot_detektor_linie.pdf'), bbox_inches='tight', dpi=900)



lineplot()
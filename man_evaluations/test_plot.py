import matplotlib.pyplot as plt
import numpy as np


def plot():
    ct30 = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\3D_SNR_eval_22012022_view_only_Cylinder_v2_ss58_without_bad_px\snr_eval_30imgs\snr\SNR-5.0W-102kV-@14.4angle.txt", skip_header=3)
    ct120 = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\3D_SNR_eval_22012022_view_only_Cylinder_v2_ss58_without_bad_px\snr_eval_120imgs\snr\SNR-5.0W-102kV-@14.4angle.txt", skip_header=3)



    plt.plot(1/(2*ct30[:, 0]), ct30[:, 1], label='ct30')
    plt.plot(1/(2*ct120[:, 0]), ct120[:, 1], label='ct120')
    plt.semilogy()
    plt.legend()
    plt.show()


def plot_std():

    std = np.genfromtxt(
        r"C:\Users\Sergej Grischagin\Desktop\Results.csv", skip_header=1, delimiter=',')

    x = np.arange(0, 50)*7.2
    y1 = std[0:50, 2].T
    y2 = std[50:100, 2].T
    plt.plot(x, y1, label='std_0-49')
    plt.plot(x, y2, label='std_50-100')
    plt.legend()
    plt.show()


plot_std()
import os
import matplotlib.pyplot as plt
import numpy as np
import Activator as act



class Plotter:
    def create_plot(self, path_result: str, act_object, ds: list, Y_style: str = 'lin'):
        fig = plt.figure()
        ax = fig.add_subplot()

        for _c in act_object.curves:
            if _c.d in ds:
                color=TerminalColor.flat_grey
                data_size = 40
                ax.text(_c.T[0] - 0.05, _c.SNR[0], f'{int(_c.d)}mm')
                _alpha = 0.7
                linew = 3
            else:
                color = TerminalColor.flat_grey
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

    @staticmethod
    def func_poly(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def whole_num(num):
        if num - int(num) == 0:
            return True
        else:
            return False


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

    flat_grey = '#d3d3d3'
    flat_red = '#D56489'
    flat_yellow = '#ECE6A6'
    flat_blue = '#009D9D'
    flat_green = '#41BA90'

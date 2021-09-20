from cycler import cycler
import matplotlib.gridspec as gridspec
from SNR_Calculation.CurveDB import *


class Plotter:
    def create_plot(self, path_result: str, act_object, ds: list, Y_style: str = 'lin'):
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



    def compare_plot(self):
        '''
        Method for comparing of measurements
        '''

        plt.rc('lines', linewidth=2)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm'])))

        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(15,6))

        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[1:, 0])
        ax2 = plt.subplot(gs[:, 1:])


        path1 = r'C:\Users\Rechenfuchs\PycharmProjects\SNR_Preperator_new_approach'
        path2 = r'C:\Users\Rechenfuchs\PycharmProjects\SNR_Preperator'

        db1 = DB(path1)
        db2 = DB(path2)

        ds = [1, 4, 5, 8, 9]

        for _d in ds:
            V_1, T_1, SNR_1 = db1.read_data(d=_d, mode='raw')
            V_2, T_2, SNR_2 = db2.read_data(d=_d, mode='raw')

            ax0.scatter(T_1, SNR_1, marker='x', s=50, label=f'{_d}mm old')
            ax1.scatter(T_2, SNR_2, marker='x', s=50, label=f'{_d}mm new')
            ax2.scatter(T_1, SNR_1, marker='x', s=50)
            ax2.scatter(T_2, SNR_2, marker='x', s=50)


            a_1, b_1, c_1 = np.polyfit(T_1, SNR_1, deg=2)
            a_2, b_2, c_2 = np.polyfit(T_2, SNR_2, deg=2)

            x_1 = np.linspace(T_1[0], T_1[-1], 141)
            y_1 = self.func_poly(x_1, a_1, b_1, c_1)
            x_2 = np.linspace(T_2[0], T_2[-1], 141)
            y_2 = self.func_poly(x_2, a_2, b_2, c_2)

            ax0.plot(x_1, y_1)
            ax1.plot(x_2, y_2)
            ax2.plot(x_1, y_1, x_2, y_2)

        ax0.grid()
        ax0.set_title('Measurement @ 30.08.2021')
        ax0.set_yscale('log')

        ax1.grid()
        ax1.set_title('Measurement @ 17.09.2021')
        ax1.set_yscale('log')

        ax2.grid()
        ax2.set_title('30.08.2021 + 17.09.2021')
        ax2.set_yscale('log')

        plt.legend()
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.4)
        plt.show()
        fig.savefig(os.path.join(path2, f'MAP_compare.pdf'), dpi=600)



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

    flat_gray = '#d3d3d3'
    flat_red = '#D56489'
    flat_yellow = '#ECE6A6'
    flat_blue = '#009D9D'
    flat_green = '#41BA90'

import numpy as np
import matplotlib.pyplot as plt

def sin(x, A, f):
    return A*np.sin(x*f)


def ft():
    fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                                   ['lower left', 'right']],
                                  figsize=(5.5, 3.5), constrained_layout=True)
    x = np.linspace(0, 2*np.pi, 1000)
    y_30 = sin(x, A=1, f=30)
    y_3 = sin(x, A=5, f=3)

    axd['upper left'].plot(x, y_30, label='30 Hz')
    axd['lower left'].plot(x, y_3, label='3 Hz')
    axd['right'].plot(x, y_3+y_30, label='sum')


    plt.legend()
    plt.show()

ft()
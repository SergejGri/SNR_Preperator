import SNRMapGenerator as snrg


def get_d():
    thick_0 = [4, 8, 12, 16, 20, 24, 28, 32]
    thick_1 = [5, 9, 13, 17, 21, 25, 29, 33]
    thick_2 = [6, 10, 14, 18, 22, 26, 30, 34]
    thicknesses = [thick_0, thick_1, thick_2]
    return thicknesses


def main():
    base_path_fin = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\MAP'
    base_path_snr = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v2'
    base_path_T = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\Transmission_v2'

    # TODO: implement more robust detection of voltages/thicknesses independent on style of passed strings
    # '160kV == '160_kV' == '160' == '160kv'... // '6mm' == '6_mm' ...
    kV_filter = ['40_kV']
    d_filter = ['6', '16']

    thicknesses = get_d()
    for i in range(len(thicknesses)):
        for j in range(len(thicknesses[0])):
            _d = thicknesses[i][j]
            if _d not in d_filter:
                generator = snrg.SNRMapGenerator(path_snr=base_path_snr,
                                            path_T=base_path_T,
                                            path_fin=base_path_fin,
                                            d=_d,
                                            kV_filter=kV_filter)
                generator()


if __name__ == '__main__':
    main()
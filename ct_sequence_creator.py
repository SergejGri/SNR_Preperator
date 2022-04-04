import os




def main():
    path_seq = ''
    texp = []

    projections = 1500
    full_angle = 360

    single_step = full_angle / projections

    with open("\\132.187.193.8\junk\sgrischagin\ct_sequence_py.txt", "w") as f:
        f.write(os.path.join("wait", "\t", "all", "\n"))
        f.write(os.path.join("rot_object", "\t", "0", "\n"))
        f.write(os.path.join("wait", "\t", "all", "\n"))

        angle = 0
        for i in range(projections):
            f.write(os.path.join("trigger_detector", "\t", f"{texp}", "\n"))
            f.write(os.path.join("rot_object", "\t", f"{angle}", "\n"))
            f.write(os.path.join("wait", "\t", "all", "\n"))
            angle = angle + single_step



if __name__ == '__main__':
    main()

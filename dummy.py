import os

def create_files():
    pathh = r'C:\Users\Sergej Grischagin\Desktop\testFolder_for_xCT'
    n = 0
    for file in os.listdir(pathh):
        if not os.path.isdir(os.path.join(pathh, file)):
            f = open(f'CT_{n}.txt', 'w')
            f.write("Your text goes here")
            f.close()
        n += 1


def main():
    pass



if __name__ == '__main__':
    main()
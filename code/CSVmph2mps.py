import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VelocityConverter():
    def __init__(self):
        self.c_mps2kmh = 3.6
        self.c_mps2mph = 2.2369362920544

        self.data = []
        self.is_mph = False
        self.file_exist = False

    def load_csv(self, path_to_csv):
        try:
            self.data = pd.read_csv(path_to_csv, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
            print("Loaded file")
            self.file_exist = True
        except:
            print("File not existent")
            return -1

    def save_changes_to_csv(self, path_to_csv):
        if self.file_exist:
            self.data.to_csv(path_to_csv, index=False, header=False, sep=',')
            print("File saved")
        else:
            print('File not existent')
        return


    def get_data(self):
        return self.data

    def change_col_mph2mps(self):
        if self.file_exist:
            self.data.speed /= self.c_mps2mph
        else:
            print('File not existent')
        return

    def show_histo(self):
        if self.file_exist:
            num_bins = 50

            avg_samples_per_bin = self.data.speed.size / num_bins
            hist, bins = np.histogram(self.data.speed, num_bins)

            width = 0.7 * (bins[1] - bins[0])
            cent = (bins[:-1] + bins[1:]) / 2
            plt.bar(cent, hist, align='center', width=width)
            plt.plot((np.min(self.data.speed), np.max(self.data.speed)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
            plt.show()
        else:
            print('File not existent')
        return


if __name__ == '__main__':
    # Test
    path = "./rec_data/dataset_1/driving_log.csv"
    conv = VelocityConverter()

    edit = False

    conv.load_csv(path)
    conv.show_histo()

    if edit:
        conv.change_col_mph2mps()
        conv.save_changes_to_csv(path)
        conv.show_histo()

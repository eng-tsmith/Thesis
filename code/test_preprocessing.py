from preprocessing import batch_generator2, batch_generator_old, plot_image
import logging
logging.basicConfig(level=logging.INFO)
from train import load_data


class Args():
    def __init__(self, data_dir, test_size):
        self.data_dir = data_dir
        self.test_size = test_size

##############################
# Test Pipeline
###############################
data_dir = './rec_data'
test_size = 0.2
batch_size = 32

# 1. Load Data
arg = Args(data_dir, test_size)
X_train, X_valid, y_train, y_valid = load_data(arg, print_enabled=True)
p = batch_generator2(data_dir, X_train, y_train, batch_size, False)

for i in p:
    print(i[0].shape)
    print(i[1].shape)

    image = i[0][0]
    train = i[1][0]

    print(train)
    plot_image(image)
    print(i[1][5])
    plot_image(i[0][5])


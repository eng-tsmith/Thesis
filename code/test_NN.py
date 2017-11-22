import logging
logging.basicConfig(level=logging.INFO)
from preprocessing import batch_generator2, batch_generator_old, plot_image
import numpy as np
from keras.models import load_model
from train import load_data



class Args():
    def __init__(self, data_dir, test_size):
        self.data_dir = data_dir
        self.test_size = test_size


###############################
# Test NN
###############################
data_dir = './rec_data'
test_size = 0.2
batch_size = 1

# 1. Load Data
arg = Args(data_dir, test_size)
X_train, X_valid, y_train, y_valid = load_data(arg)
p = batch_generator2(data_dir, X_train, y_train, batch_size, True)

# 2. Load Model
path_to_model = 'C:/Users/ga29mos/Dev/Thesis/code/logs/New/model-070.h5'
logging.info("Loading model at: " + path_to_model)
model = load_model(path_to_model)
model.summary()

# 3. Predict
for i in p:
    image = i[0][0]
    plot_image(image)
    image = np.array([image])
    train = i[1]

    # predict the steering angle for the image
    steering_angle = float(model.predict(image, batch_size=1)[0][0])

    logging.info(steering_angle)
    logging.info(train)
    logging.info(30*'-')


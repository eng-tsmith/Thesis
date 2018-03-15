import base64
import os
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
import preprocessing
import logging
import sys
import cv2
import argparse


class PIDController:
    def __init__(self, kp, ki, kd):
        """
        Init function for new PID control.
        :param kp: p-Anteil
        :param ki: i-Anteil
        :param kd: d-Anteil
        """
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

        self.set_point = 0.0
        self.error = 0.0

        self.integrator = 0.0
        self.derivator = 0.0

        self.integrator_min = 0.0
        self.integrator_max = 500.0

    def set_desired(self, desired):
        self.set_point = desired
        # self.integrator = 0.0
        # self.derivator = 0.0

    def update(self, current_value):
        self.error = self.set_point - current_value

        p_value = self.Kp * self.error
        d_value = self.Kd * (self.error - self.derivator)

        self.derivator = self.error
        self.integrator += self.error

        # windup limiter
        if self.integrator > self.integrator_max:
            self.integrator = self.integrator_max
        elif self.integrator < self.integrator_min:
            self.integrator = self.integrator_min

        i_value = self.Ki * self.integrator

        pid = p_value + i_value + d_value

        return pid

    def get_error(self):
        return self.error

    def get_integrator(self):
        return self.integrator


# Init logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

# initialize server & Flask app
sio = socketio.Server()
app = Flask(__name__)

# Some models use different output dim
use_double_output = None
# Some models use different input dim
model_name = None

# Init Keras
model = None
prev_image_array = None

# Init GUI
EXTRA_GUI = True

# Init PID
pid_controller = PIDController(5., 0., 0.)
pid_controller.set_desired(0.0)


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    """
    This function is called after receiving telemtry event
    :param sid:
    :param data: data packet received from UNITY
    """
    global pid_controller
    global use_double_output
    global model_name

    if data:
        # Get current data of car
        # steering_angle = float(data["steering_angle"])
        # throttle = float(data["throttle"])
        speed = float(data["speed"])
        img_str = data["image"]
        image_src = Image.open(BytesIO(base64.b64decode(img_str)))

        try:
            # Preprocessing
            image = np.asarray(image_src)
            image = preprocessing.preprocess(image, model_name)

            # predict the steering angle and velocity for the image
            if use_double_output:
                output_2 = model.predict(image[None, :, :, :], batch_size=1)
                desired_steering_angle = output_2[0][0][0]
                desired_speed = output_2[1][0][0]
            else:
                desired_steering_angle, desired_speed = model.predict(image[None, :, :, :], batch_size=1)[0]

            # check if desired steering angle between -1 and 1
            desired_steering_angle = np.clip(desired_steering_angle, -1.0, 1.0)

            # Denormalize speed
            desired_speed = preprocessing.denormalize_speed(desired_speed)

            # Control speed
            pid_controller.set_desired(desired_speed)
            throttle = pid_controller.update(speed)

            # Prevent bremsen zittern
            if -1 < pid_controller.get_error() < 0:
                throttle = 0

            logging.info("sa: {:.4f}  \tacc: {:.4f}  \tv_err: {:.4f}  \tv_current: {:.4f}   \tv_int: {:.4f}".format(
                desired_steering_angle, throttle, pid_controller.get_error(), speed, pid_controller.get_integrator()))

            send_control(desired_steering_angle, throttle)

            # Extra gui
            if EXTRA_GUI:
                # im = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
                im = preprocessing.process_img_for_visualization(image, angle=desired_steering_angle)
                cv2.imshow('Center camera', cv2.cvtColor(np.asarray(image_src), cv2.COLOR_RGB2BGR))
                cv2.imshow('CNN input', im/255)
                cv2.waitKey(1)
        except Exception as e:
            logging.info(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    """
    This funtion is called after successful connection with socketIO
    :param sid: ID of client
    :param environ:
    """
    logging.info("connect " + sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    """
    Used to send driving controls to client
    :param steering_angle:
    :param throttle:
    """
    sio.emit("steer",
             data={'steering_angle': steering_angle.__str__(),
                   'throttle': throttle.__str__()},
             skip_sid=True)


def run(path_to_model):
    global sio
    global app
    global model
    global EXTRA_GUI
    global model_name

    if path_to_model == 'main':
        logging.info("Can't run file as main. Exiting")
        sys.exit()

    logging.info("Loading model at: " + path_to_model)
    model = load_model(path_to_model)
    model.summary()

    if model.input_shape[1] == 66 and model.input_shape[2] == 200:
        model_name = 'nvidia'
    elif model.input_shape[1] == 64 and model.input_shape[2] == 64:
        model_name = 'electron'

    logging.info("Creating image folder at {}".format('./data/'))
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    else:
        # shutil.rmtree('./data/')
        # os.makedirs('./data/')
        logging.info("RECORDING THIS RUN ...")

    # Initialize OpenCV image windows
    if EXTRA_GUI:
        cv2.namedWindow('Center camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('CNN input', cv2.WINDOW_NORMAL)

    # wrap Flask application with engineIO's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


if __name__ == '__main__':
    global use_double_output
    use_double_output = True
    # Parser
    parser = argparse.ArgumentParser(description='Behavioral Cloning Drive Program')
    parser.add_argument('-p', help='path to model.h5 file', dest='model_path', type=str, default=None)
    args = parser.parse_args()

    if args.model_path:
        logging.info('Loading model.')
        run(args.model_path)
    else:
        logging.info('No path specified. Loading default.')
        # run('C:/Users/timmy/Documents/Dev/Thesis/code/logs/montr_val_3/model-005.h5')
        # run('C:/ProgramData/Thesis/code/logs/1517044193.003092/model-053.h5')
        # run('C:/Users/timmy/Documents/Dev/Thesis/code/logs/TEST_elec/model-100.h5')
        # run('C:/Users/timmy/Documents/Dev/Thesis/code/logs/FINAL_nvidia_adam_split_try/model-098.h5')
        run('C:/Users/timmy/Documents/Dev/Thesis/code/logs/FINAL_nvidia_adam_split_tobi/model-003.h5')
        # run('C:/Users/timmy/Documents/Dev/Thesis/code/logs/FINAL_electron_nadam_val_split/model-099.h5')

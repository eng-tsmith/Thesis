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

# initialize server & Flask app
sio = socketio.Server()
app = Flask(__name__)

use_NVIDIA = True

model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED

EXTRA_GUI = True

# # PI controller
# Kp = 1/30.0 * 2.5  # P gain
# Ki = 0.01  # I gain
# # Integration buffer for the PI speed controller
# speed_integrated = 0


class PIDController:
    def __init__(self, kp, ki, kd):
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
        # self.integrator = 0.0  # TODO check if necessary
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


pid_controller = PIDController(5., 0., 0.)
pid_controller.set_desired(0.0)


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    global pid_controller
    global use_NVIDIA

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
            image = preprocessing.preprocess(image)

            # predict the steering angle and velocity for the image
            if use_NVIDIA:
                output_2 = model.predict(image[None, :, :, :], batch_size=1)
                desired_steering_angle = output_2[0][0][0]
                desired_speed = output_2[1][0][0]
            else:
                desired_steering_angle, desired_speed = model.predict(image[None, :, :, :], batch_size=1)[0]

            # check if desired steering angle between -1 and 1
            desired_steering_angle = np.clip(desired_steering_angle, -1.0, 1.0)

            # Denormalize speed
            desired_speed = preprocessing.denormalize_speed(desired_speed)

            # desired_speed = 0.8*desired_speed
            # desired_speed = min(desired_speed, 20.0)

            # Control speed
            pid_controller.set_desired(desired_speed)
            throttle = pid_controller.update(speed)

            # # Calculate the throttle value and decide when to go to neutral or break.
            # # Break is activated by negative throttle
            # if pid_controller.get_error() < - 3:
            #     # Break
            #     throttle = -1
            # elif speed_error < 0:
            #     # Neutral
            #     throttle = 0
            # else:
            #     throttle = Kp * (speed_error + speed_integrated * Ki)

            logging.info("sa: {:.4f}  \tacc: {:.4f}  \tv_err: {:.4f}  \tv_current: {:.4f}   \tv_int: {:.4f}".format(
                desired_steering_angle, throttle, pid_controller.get_error(), speed, pid_controller.get_integrator()))

            send_control(desired_steering_angle, throttle)

            # Extra gui
            if EXTRA_GUI:
                # im = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
                im = preprocessing.process_img_for_visualization(image, angle=desired_steering_angle)  # TODO maybe in img source
                cv2.imshow('Center camera', cv2.cvtColor(np.asarray(image_src), cv2.COLOR_RGB2BGR))
                cv2.imshow('CNN input', im/255)
                cv2.waitKey(1)
        except Exception as e:
            logging.info(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    logging.info("connect " + sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


def run(path_to_model):
    global sio
    global app
    global model
    global EXTRA_GUI

    if path_to_model == 'main':
        logging.info("Can't run file as main. Exiting")
        sys.exit()

    logging.info("Loading model at: " + path_to_model)
    model = load_model(path_to_model)
    model.summary()

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
    # Logging
    global use_NVIDIA
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading example...')
    use_NVIDIA = True

    # run('C:/Users/timmy/Documents/Dev/Thesis/code/logs/montr_val_3/model-005.h5')
    # run('C:/ProgramData/Thesis/code/logs/1517044193.003092/model-053.h5')
    run('C:/Users/timmy/Documents/Dev/Thesis/code/logs/FINAL_nvidia_adam_split/model-009.h5')
    # run('C:/Users/timmy/Documents/Dev/Thesis/code/logs/FINAL_electron_nadam_val_split/model-099.h5')

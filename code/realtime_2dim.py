import base64 #decoding camera images
import os #reading and writing files
import shutil #high level file operations
import numpy as np
import socketio #real-time server
import eventlet #concurrent networking
import eventlet.wsgi #web server gateway interface
from PIL import Image
from flask import Flask
from io import BytesIO #input output
from keras.models import load_model
import preprocessing
import logging
import sys
import cv2

# initialize server & Flask app
sio = socketio.Server()
app = Flask(__name__)

model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED

EXTRA_GUI = True

# PI controller
Kp = 1/30.0 * 2.5   # P gain
Ki = 0.01 # I gain
# Integration buffer for the PI speed controller
speed_integrated = 0

# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    global Kp
    global Ki
    global speed_integrated
    global speed_limit
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        img_str = data["image"]
        image_src = Image.open(BytesIO(base64.b64decode(img_str)))
        try:
            image = np.asarray(image_src)       # from PIL image to numpy array
            image = preprocessing.preprocess(image)  # apply the preprocessing
            #image = np.asarray(image, dtype=np.float32)       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle, speed_pred = model.predict(image[None, :, :, :], batch_size=1)[0]
            logging.info(model.predict(image[None, :, :, :], batch_size=1)[0])


            # Source Lectron Castle
            # https://github.com/electroncastle/behavioral_cloning/blob/master/drive.py
            # Convert the normalized speed to mph
            max_speed = 50.0 #also seen in preprocessing.normalize_speed()
            speed_pred = (speed_pred + 0.5) * max_speed

            # Get the difference between the desired speed and the current car speed
            speed_error = speed_pred - speed

            # Integrate the speed errors
            speed_integrated += speed_error
            # windup limiter
            speed_integrated = max(0, speed_integrated)

            # Calculate the throttle value and decide when to go to neutral or break.
            # Break is activated by negative throttle
            if speed_error < - 3:
                # Break
                throttle = -1
            elif speed_error < 0:
                # Neutral
                throttle = 0
            else:
                throttle = Kp * (speed_error + speed_integrated * Ki)

            logging.info("sa: {:.4f}  \tacc: {:.4f}  \tv_err: {:.4f}  \tv_current: {:.4f}   \tv_int: {:.4f}".format(steering_angle, throttle, speed_error, speed, speed_integrated))

            send_control(steering_angle, throttle)

            # Extra gui
            if EXTRA_GUI:
                cv2.imshow('Center camera', cv2.cvtColor(np.asarray(image_src), cv2.COLOR_RGB2BGR))
                cv2.imshow('CNN input', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
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
        #shutil.rmtree('./data/')
        #os.makedirs('./data/')
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
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading example...')
    run('C:/ProgramData/Thesis/code/logs/1516290512.150514/model-003.h5')

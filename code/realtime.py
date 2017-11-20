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

# initialize server & Flask app
sio = socketio.Server()
app = Flask(__name__)

model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = preprocessing.preprocess(image)  # apply the preprocessing
            image = np.array([image])       # the model expects 4D array

            logging.info(image.shape)

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1)[0])
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            logging.info('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            logging.info(e)

        ## save frame
        #if args.image_folder != '':
        #    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        #    image_filename = os.path.join(args.image_folder, timestamp)
        #    image.save('{}.jpg'.format(image_filename))
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

    if path_to_model == 'main':
        logging.info("Can't run file as main. Exiting")
        sys.exit()
    logging.info("Loading model at: " + path_to_model)
    model = load_model(path_to_model)
    logging.info("Creating image folder at {}".format('./data/'))
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    else:
        shutil.rmtree('./data/')
        os.makedirs('./data/')
        logging.info("RECORDING THIS RUN ...")

    # wrap Flask application with engineIO's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

if __name__ == '__main__':
    # Logging
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading example...')
    run('C:\ProgramData\Thesis\code\logs\model.h5')

import train, realtime
import threading
import multiprocessing
import os

import kivy
kivy.require('1.10.0')

from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView

import logging
logging.basicConfig(level=logging.INFO)


#############################################################
class ScreenManagement(ScreenManager):
    pass


class ScreenMain(Screen):
    pass


class ScreenSettingsTraining(Screen):
    pass


class ScreenSettingsRealtime(Screen):
    pass


class MyButton(Button):
    enabled = BooleanProperty(True)

    def on_enabled(self, instance, value):
        if value:
            self.background_color = [1,1,1,1]
            self.color = [1,1,1,1]
        else:
            self.background_color = [1,1,1,.3]
            self.color = [1,1,1,.5]

    def on_touch_down( self, touch ):
        if self.enabled:
            return super(self.__class__, self).on_touch_down(touch)


class RealtimeThread(threading.Thread):
    """ Threaded realtime contro """
    stopper = None

    def __init__(self, stopper, path):
        threading.Thread.__init__(self)
        self.stopper = stopper
        self.path = path

    def run(self):
        while not self.stopper.is_set():
            realtime.run(self.path)

    def stop(self):
        self.stopper.set()


class TrainThread(threading.Thread):
    """ Threaded realtime contro """
    stopper = None

    def __init__(self, params):
        threading.Thread.__init__(self)
        self.params = params

    def run(self):
        train.run(self.params)


class RealtimeProcess(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
        return

    def run(self):
        realtime.run()
        return


#############################################################
class GuiThesisApp(App):
    local_path = os.path.abspath(os.path.join(os.path.curdir, 'logs')) # dont change because of kivy bug!!!

    def build(self):
        return ScreenManagement()

    def train_ext(self, params):
        logging.info("Start training")
        #logging.info(params)

        worker = TrainThread(params)
        worker.start()
        content = Button(text='Neural Network is training...')
        popup = Popup(title='Starting Training...',
                      content=content,
                      size_hint=(None, None), size=(400, 400),
                      auto_dismiss=False)
        content.bind(on_press=popup.dismiss)
        popup.open(self)

    def realtime_ext(self, curr_path, selection):
        logging.info("Start real-time control")
        stopper = threading.Event()

        if not selection:
            path = os.path.normpath(os.path.join(curr_path, 'model-005.h5'))
        else:
            path = os.path.normpath(" ".join(str(x) for x in selection))

        worker = RealtimeThread(stopper, path)
        worker.start()
        content = Button(text='Cancel')
        popup = Popup(title='Looking for Unity IO Socket...',
                      content=content,
                      size_hint=(None, None), size=(400, 400),
                      auto_dismiss=False)
        content.bind(on_press=popup.dismiss)
        popup.bind(on_dismiss=lambda x: worker.stop())
        popup.open(self)
#############################################################


if __name__ == "__main__":
    thesis_app = GuiThesisApp()
    thesis_app.run()

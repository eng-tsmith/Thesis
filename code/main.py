import train, realtime
import threading
import multiprocessing

import kivy
kivy.require('1.10.0')

from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen

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


class RealtimeThread(threading.Thread):
    """ Threaded realtime contro """
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        realtime.main()


class RealtimeProcess(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
        return

    def run(self):
        realtime.main()
        return


#############################################################
class TimmyApp(App):
    def build(self):
        return ScreenManagement()

    def train_ext(self, *args):
        content = Button(text='Not implemented yet!')
        popup = Popup(title='Test popup',
                      content=content,
                      size_hint=(None, None), size=(400, 400),
                      auto_dismiss=False)
        content.bind(on_press=popup.dismiss)
        popup.open(self)

    def realtime_ext(self, *args):
        logging.info("Start real-time control")
        worker = RealtimeProcess()
        worker.start()
        content = Button(text='Stop')
        popup = Popup(title='Looking for Unity IO Socket...',
                      content=content,
                      size_hint=(None, None), size=(400, 400),
                      auto_dismiss=False)
        content.bind(on_press=popup.dismiss)
        popup.bind(on_dismiss=lambda x: worker.terminate())
        popup.open(self)
#############################################################


if __name__ == "__main__":
    timmy_app = TimmyApp()
    timmy_app.run()

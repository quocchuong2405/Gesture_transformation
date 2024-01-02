import pyautogui
import numpy as np

class ZoomControl:
    def __init__(self, model):
        self.model = model
        self.gestures = ['A', 'V', 'Q']
        
    def zoom_control(self, buffer):
        """
        Predict Gesture and perform corresponding zoom control action
        """
        buffer = np.expand_dims(buffer, axis=0)
        res = self.model.predict(buffer, verbose=False)[0]
        pred = np.argmax(res)
        gesture = self.gestures[pred]
        
        if gesture == 'A': # Alt A, microphone keybine, toggle on/off
            pyautogui.hotkey("altleft", 'a') 
        elif gesture == 'V': # Alt V, video keybind, toggle on/off
            pyautogui.hotkey("altleft", 'v') 
        elif gesture == 'Q': # Alt Q, quit keybind, quit current instance
            pyautogui.hotkey("altleft", 'q') 
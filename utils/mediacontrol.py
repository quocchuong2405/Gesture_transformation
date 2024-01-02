import pyautogui
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

class MediaControl:
    def __init__(self, model):
        self.model = model
        self.gestures = ['P', 'M', 'C', 'L', 'J']
    
    def media_control(self, buffer):
        """
        Predict Gesture and perform corresponding media control action
        """
        buffer = np.expand_dims(buffer, axis=0)
        res = self.model.predict(buffer, verbose=False)[0]
        pred = np.argmax(res)
        gesture = self.gestures[pred]
        
        if gesture == 'P': # Play/Pause
            # pyautogui.press("playpause")
            pyautogui.press("k")
        elif gesture == 'M': # Mute Toggle
            # pyautogui.press("volumemute")
            pyautogui.press("m")
        elif gesture == 'J': # Forwards in vid, j is hotkey
            # pyautogui.press("nexttrack")
            pyautogui.press("j")
        elif gesture == 'L': # Backwards in vid, L is hotkey
            # pyautogui.press("prevtrack")
            pyautogui.press("l")
        elif gesture == 'C': # subtitles
            pyautogui.press("c")
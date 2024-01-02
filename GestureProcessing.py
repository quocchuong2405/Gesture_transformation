import time
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

from utils.landmark import Landmark
from utils.collection import Collection
from utils.quickmenu import QuickMenu
from utils.mouse import Mouse
from utils.mediacontrol import MediaControl
from utils.zoomcontrol import ZoomControl
from utils.qm_utils import *

# USER PREFERENCES
MIRROR = True
DRAW_LANDMARKS = False
QM_HAND = 'left'
CURSOR_HAND = 'right'

# QUICK MENU MODES
CURSOR = 0
MEDIA = 1
ZOOM = 2
MODE_SELECT = 3

qm_modes = QMModes()

def Model(classes):
    # creates LSTM model and adds layers
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    return model

def setup_quickmenus():
    # QUICK MENU: MODE SELECT
    qm_mode_select = QuickMenu(3*np.pi/6)
    qm_mode_select.add_action("Cursor Mode", qm_modes.set_mode_cursor)
    qm_mode_select.add_action("Media Mode", qm_modes.set_mode_media)
    qm_mode_select.add_action("Zoom Mode", qm_modes.set_mode_zoom)
    
    # QUICK MENU: CURSOR MODE
    qm_cursor = QuickMenu(3*np.pi/6)
    qm_cursor.add_action("Change Mode", qm_modes.set_mode_select)
    qm_cursor.add_action("Copy", mouseCopy)
    qm_cursor.add_action("Paste", mousePaste)
    qm_cursor.add_action("OSK", openOSK)
    
    # QUICK MENU: MEDIA MODE
    qm_media = QuickMenu(3*np.pi/6)
    qm_media.add_action("Cursor Mode", qm_modes.set_mode_cursor)
    qm_media.add_action("Zoom Mode", qm_modes.set_mode_zoom)
    qm_media.add_action("Input Gesture", qm_modes.input_gesture)
    
    # QUICK MENU: ZOOM MODE
    qm_zoom = QuickMenu(3*np.pi/6)
    qm_zoom.add_action("Cursor Mode", qm_modes.set_mode_cursor)
    qm_zoom.add_action("Media Mode", qm_modes.set_mode_media)
    qm_zoom.add_action("Input Gesture", qm_modes.input_gesture)
    
    return [qm_cursor, qm_media, qm_zoom, qm_mode_select]
    

# main loop of program
def ProcessGesture():
    # Classes that help with processing mediapipe landmarks
    landmark = Landmark()
    collection = Collection()
    
    # Setup quick menus
    qm_modes.current_mode = MODE_SELECT
    quick_menus = setup_quickmenus()
    
    # Setup mouse control
    mouse = Mouse(cursor_speed=4)
    
    # Setup zoom and media gesture recognition
    zoom_model = Model(3)
    zoom_model.load_weights(r"model/models/Gesture_Model_Zoom.keras")
    zoom_control = ZoomControl(zoom_model)
    
    media_model = Model(5)
    media_model.load_weights(r"model/models/Gesture_Model_Media.keras")
    media_control = MediaControl(media_model)
    
    landmark_buffer = []
    
    """
    IMPORTANT!!!
    Change video capture dependent on what webcam you're using.
    Generally, system webcam will be 0 and external one will be 1 (can be > 1)
    e.g. cap = cv2.VideoCapture(2)
    
    If camera doesn't work or an error occurs, remove/add cv2.CAP_DSHOW
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    
    
    with landmark.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read Camera feed
            ret, frame = cap.read()

            # Make detections
            image, results = landmark.mediapipe_detection(frame, holistic)
            landmark.extract_keypoints(results, image)
            
            # Draw landmarks
            if DRAW_LANDMARKS:
                landmark.draw_styled_landmarks(image, results)
            
            if MIRROR:
                image = cv2.flip(image, 1)
            
            # Calculate Landmark properties
            landmark.extract_keypoints(results, image)
            landmark.calc_fingers_raised()
            landmark.calc_hand_rotations(MIRROR)
            
            # Handles recording gestures in MEDIA MODE and ZOOM MODE
            if qm_modes.record_next == True:
                keypoints = collection.extract_keypoints(results)
                landmark_buffer.append(keypoints)
            else:
                landmark_buffer = []
            
            # Handles quick menu animations
            if qm_modes.animate == True:
                quick_menus[qm_modes.current_mode].animation = True
                quick_menus[qm_modes.current_mode].animation_start = time.time()    
                qm_modes.animate = False
            
            # Display quick menu
            quick_menus[qm_modes.current_mode].quickmenu(landmark, image, QM_HAND)
            
            """
            MODE HANDLING
            """
            
            # CURSOR MODE
            if qm_modes.current_mode == CURSOR:
                mouse.mouse(image, landmark, CURSOR_HAND)    
                
            # MEDIA MODE
            elif qm_modes.current_mode == MEDIA:
                if len(landmark_buffer) == 40:
                    media_control.media_control(landmark_buffer[-30:])
                    qm_modes.record_next = False
                
            # ZOOM MODE
            elif qm_modes.current_mode == ZOOM:
                if len(landmark_buffer) == 40:
                    zoom_control.zoom_control(landmark_buffer[-30:])
                    qm_modes.record_next = False


            ret,buffer=cv2.imencode('.jpg',image)
            image=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

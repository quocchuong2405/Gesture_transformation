import pyautogui
import win32api, win32con
import numpy as np
import time
import cv2
from utils.landmark import Landmark
import threading
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

class Mouse():
    # Mouse gestures:
    NO_INPUT = 0
    CURSOR = 1
    LEFT_CLICK = 2
    RIGHT_CLICK = 3
    DRAG = 4
    SCROLL = 5
    
    def __init__(self, cursor_speed):
        self.last_pos = None
        self.initial_pos = None
        self.cursor_speed = cursor_speed
        self.gesture = None
        self.last_gesture = None
        self.drag_start_time = 0
    
    def mouse(self, image, landmark : Landmark, handedness):
        """
        Calculates mouse gestures and performs the corresponding action
        """
        # Retreive Hand Landmark Information
        if handedness == 'left':
            cursor_hand_keypoints = landmark.lh
            raised = landmark.lh_raised
        elif handedness == 'right':
            cursor_hand_keypoints = landmark.rh
            raised = landmark.rh_raised
        
        if cursor_hand_keypoints is None:
            self.last_gesture = None
            self.last_pos = None
            return
        
        # Pointer is middle of palm
        pointer = [(cursor_hand_keypoints[0][0] + cursor_hand_keypoints[5][0] + cursor_hand_keypoints[17][0])//3, 
                   (cursor_hand_keypoints[0][1] + cursor_hand_keypoints[5][1] + cursor_hand_keypoints[17][1])//3]
        
        index_mcp_x = cursor_hand_keypoints[5][0]
        pinky_mcp_x = cursor_hand_keypoints[17][0]
        
        # Only show if the front of the hand is showing
        if index_mcp_x < pinky_mcp_x:
            return
        
        # CURSOR
        if raised == [True, False, False, False, True] or raised == [True, True, False, False, True]:
            if self.last_gesture == self.LEFT_CLICK:
                pyautogui.mouseUp(_pause = False)
            self.move_mouse(pointer)
            self.last_gesture = self.CURSOR
            return
        
        invert = lambda x : [image.shape[1] - x[0] - 1, x[1]]
        # LEFT CLICK
        if raised == [True, False, False, False, False]:
            if self.last_gesture == self.CURSOR:
                self.initial_pos = pointer
                self.last_pos = pointer
                print("Click!")
                pyautogui.mouseDown(_pause = False)
                self.drag_start_time = time.time()
            
            # If left click has been held down for at least 0.2 seconds,
            elif self.last_gesture == self.LEFT_CLICK and time.time() - self.drag_start_time >= 0.2:
                self.move_mouse(pointer)
                self.initial_pos = self.initial_pos if self.initial_pos is not None else pointer
                
                # Draw user feedback
                cv2.circle(image, invert(self.initial_pos),
                           5, (255,255,255), -1, cv2.LINE_AA)
                cv2.line(image, invert(pointer), invert(self.initial_pos),
                         (255,255,255), 2, cv2.LINE_AA)
            
            self.last_gesture = self.LEFT_CLICK
            return
        else:
            pyautogui.mouseUp(_pause = False)
        
        # RIGHT CLICK
        if raised == [True, True, False, False, False] and self.last_gesture == self.CURSOR:
            self.last_pos = None
            print("Right Click!")
            pyautogui.rightClick(_pause = False)
            self.last_gesture = self.RIGHT_CLICK
            return
        
        # SCROLLING
        if raised == [True, True, True, False, False]:
            if self.last_gesture == self.CURSOR:
                self.initial_pos = pointer
                self.last_pos = pointer
                print("Scroll!")
                
            elif self.last_gesture == self.SCROLL:
                self.scroll_mouse(pointer)
                
                # Generate user feed back
                cv2.circle(image, invert(self.initial_pos),
                           5, (255,255,255), -1, cv2.LINE_AA)
                cv2.line(image, invert(pointer), invert(self.initial_pos),
                         (255,255,255), 2, cv2.LINE_AA)
                
            self.last_gesture = self.SCROLL
            return
        
        # NO INPUT
        self.drag_start_time = 0
        pyautogui.mouseUp(_pause = False)
        self.last_pos = None
        self.initial_pos = None
    
    def move_mouse(self, pointer_pos):
        """
        Move mouse relate to how much it moved from the previous position
        """
        if self.last_pos is None:
            self.last_pos = pointer_pos
        else:
            dir = np.array(pointer_pos) - np.array(self.last_pos)
            threading.Thread(target = pyautogui.moveRel, daemon=True, args = (self.cursor_speed * -1 * dir[0], self.cursor_speed * dir[1], 0), kwargs={"_pause" : False}).start()
            self.last_pos = pointer_pos
            
    def scroll_mouse(self, pointer):
        """
        Scroll the mouse in the direction the hand moved
        """
        if self.initial_pos is None:
            self.initial_pos = pointer
        else:
            direction = int((self.initial_pos[1] - pointer[1])/abs(self.initial_pos[1] - pointer[1])) if int(self.initial_pos[1] - pointer[1]) != 0 else 0
            # Scale speed to how far the hand is from the initla position
            speed = self.cursor_speed/6 * np.linalg.norm(np.array(pointer) - np.array(self.initial_pos))
            clicks = 1 * int(speed)
            if clicks != 0:
                threading.Thread(target = scroll, daemon=True, args = (direction*clicks, 0, 0, 0.05/clicks)).start()
    
def scroll(clicks=0, delta_x=0, delta_y=0, delay_between_ticks=0):
    """
    Source: https://learn.microsoft.com/en-gb/windows/win32/api/winuser/nf-winuser-mouse_event?redirectedfrom=MSDN
    """
    if clicks > 0:
        increment = win32con.WHEEL_DELTA//12
    else:
        increment = win32con.WHEEL_DELTA//12 * -1

    for _ in range(abs(clicks)):
        win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, delta_x, delta_y, increment, 0)
        time.sleep(delay_between_ticks)
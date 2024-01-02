import numpy as np
from utils.landmark import Landmark
import cv2
import time

# COLORS = [RED, ORANGE, GREEN, BLUE, PURPLE]
COLORS = [(60, 20, 220), (0,165,255), (0,252,124), (255,144,30), (226,43,138)]

# A quick menu that will apear as a circle around the non-dominant 
# hand which can be selected from rotating the hand
class QuickMenu():
    HOVER_TIME = 1.0
    SELECTION_BUFFER_TIME = 2.0
    ANIMATION_TIME = 0.5
    PRECISION =  np.pi/36
    
    def __init__(self, menu_degrees : float):
        self.num_actions = 0
        self.action_names = []
        self.action_functions = {}
        
        self.button_angles = []
        self.menu_degrees = menu_degrees # The angle of the slice of the menu circle
        
        self.selection_buffer_time = 0  # The time of when the last selection was made
        self.selection_hold_time = 0    # The time of when the cursor first hovers on the button
        self.selection_hover = False    # If a button is being hovered (user is in the middle of selecting button)
        self.selected = False           # If a button has been selected
        self.selected_name = None       # Name of selected action
        
        self.animation = False
        self.animation_start = 0
    
    def quickmenu(self, landmark : Landmark, image, handedness):
        """
        Handles quick menu logic and displays it
        """
        # Retreive Hand Landmark Information
        if handedness == 'left':
            qm_hand_keypoints = landmark.lh
            raised = landmark.lh_raised
            rotation = landmark.lh_angle
        elif handedness == 'right':
            qm_hand_keypoints = landmark.rh
            raised = landmark.rh_raised
            rotation = landmark.rh_angle
        
        if qm_hand_keypoints is None:
            self.selection_hold_time = 0
            self.selection_hover = False
            return
        
        pointing_to = self.pointing_to(rotation)
        
        index_mcp_x = qm_hand_keypoints[5][0]
        pinky_mcp_x = qm_hand_keypoints[17][0]
        
        # Handles starting animation
        if self.animation == True and time.time() - self.animation_start < self.ANIMATION_TIME:
            self.animate(image, qm_hand_keypoints, x_flip=True)
            return
        else:
            self.animation_start = 0
            self.animation = False
        
        # Only show if all fingers but the thumb are raised and that the front of the hand is showing
        if all(raised[:4]) and (index_mcp_x < pinky_mcp_x):
            # Handle action selection logic
            if time.time() - self.selection_buffer_time >= self.SELECTION_BUFFER_TIME: # A selection can only be made every 2 seconds
                self.selected = False
                # If thumb is not raised - a selection has been made
                if pointing_to is not None and raised[4] == False:
                    if self.selection_hover == False:
                        # Start timing hover time
                        self.selection_hold_time = time.time()
                        self.selection_hover = True
                    
                    # Wait till user has hovered on the button for at least HOVER_TIME
                    if time.time() - self.selection_hold_time >= self.HOVER_TIME:
                        # Handle selection
                        self.selected = True   
                        self.selected_name = self.action_names[pointing_to]
                        
                        # Run selected action
                        func = self.action_functions[self.selected_name]
                        if func is not None:
                            func()
                        
                        self.selection_buffer_time = time.time()
                        self.selection_hover = False
                else:
                    self.selection_hover = False  
                    self.selection_hold_time = time.time()
            
            self.draw_menu(image, qm_hand_keypoints, rotation, pointing_to, x_flip=True)
        
    def add_action(self, name : str, function):
        """
        Adds an action button to the the quickmenu
        """
        self.action_names.append(name)
        self.action_functions[name] = function
        self.num_actions += 1
        self.calculate_button_angles()

    def calculate_button_angles(self):
        """
        Evenly distributes the positions of the buttons across self.menu_degrees
        """
        left_angle = np.pi/2 + self.menu_degrees/2        
        self.button_angles = [left_angle - self.menu_degrees/(self.num_actions + 1) * i for i in range(1, self.num_actions + 1)]
    
    def pointing_to(self, hand_rotation):
        """
        Returns the button the hand is pointing to, within given precision angle. 
        None if not pointing to anything
        """
        button_angles_diff = np.array(self.button_angles)
        button_angles_diff = np.abs(button_angles_diff - hand_rotation)
        pointing_to = np.where(button_angles_diff <= self.PRECISION)[0]
        if len(pointing_to) == 0:
            return None
        return pointing_to[0]
    
    def draw_menu(self, image, hand_landmarks, hand_rotation, pointing_to, x_flip = False):
        """
        Draws the quick menu
        """
        base = hand_landmarks[0]
        palm = [(hand_landmarks[0][0] + hand_landmarks[5][0] + hand_landmarks[17][0])//3, (hand_landmarks[0][1] + hand_landmarks[5][1] + hand_landmarks[17][1])//3]        
        
        # Menu radius is relative to distance from camera 
        circle_radius = np.linalg.norm(np.array(hand_landmarks[9]) - np.array(hand_landmarks[0])) * 2.0
        
        # Flip x coordinates if camera is mirrored
        if x_flip == True:
            base[0] = image.shape[1] - base[0] - 1
            palm[0] = image.shape[1] - palm[0] - 1     
        
        if pointing_to is None:
            # Draw cursor
            cv2.circle(image, [int(palm[0] - circle_radius * np.cos(hand_rotation)), int(palm[1] - circle_radius * np.sin(hand_rotation))], 5, (255,255,255), -1, cv2.LINE_AA)
        elif self.selected == False:
            # Draw the action name of the button the cursor is pointing to
            textsize = cv2.getTextSize(self.action_names[pointing_to], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX = (base[0] - textsize[0]// 2) 
            textY = (base[1] + textsize[1]// 2)
            padding = 10
            cv2.rectangle(image, 
                          (base[0] - textsize[0]//2 - padding, base[1] - textsize[1]//2 - padding), (base[0] + textsize[0]//2 + padding, base[1] + textsize[1]//2 + padding), 
                          color=(255,255,255), 
                          thickness=-1, 
                          lineType=cv2.LINE_AA)
            cv2.putText(image, self.action_names[pointing_to], (textX , textY) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
        # Draw the name of the selected action
        if self.selected == True:
            textsize = cv2.getTextSize(self.selected_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX = (base[0] - textsize[0]// 2) 
            textY = (base[1] + textsize[1]// 2)
            padding = 10
            cv2.rectangle(image, 
                          (base[0] - textsize[0]//2 - padding, base[1] - textsize[1]//2 - padding), (base[0] + textsize[0]//2 + padding, base[1] + textsize[1]//2 + padding), 
                          color=(0,0,0), 
                          thickness=-1, 
                          lineType=cv2.LINE_AA)
            cv2.putText(image, self.selected_name, (textX , textY) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # Draw all action buttons
        for i, angle in enumerate(self.button_angles):
            if i == pointing_to:
                # Draw button user is pointing to:
                # Inner Colored Circle
                cv2.circle(image, [int(palm[0] - circle_radius * np.cos(angle)), int(palm[1] - circle_radius * np.sin(angle))], 
                           5, COLORS[i], -1, cv2.LINE_AA) 
                # Outline
                cv2.circle(image, [int(palm[0] - circle_radius * np.cos(angle)), int(palm[1] - circle_radius * np.sin(angle))],
                            9, (255,255,255), 2, cv2.LINE_AA) 
                
                # Draw progress if button is being hovered (Coloring in the outer ring)
                if self.selection_hover == True:
                    cv2.ellipse(image, 
                                [int(palm[0] - circle_radius * np.cos(angle)), int(palm[1] - circle_radius * np.sin(angle))],
                                axes = (9,9), 
                                angle = 0,
                                startAngle=180,
                                endAngle=180 + 360 * (time.time() - self.selection_hold_time)/self.HOVER_TIME,
                                color=COLORS[i],
                                thickness=2,
                                lineType=cv2.LINE_AA)
                continue
            # Draw action buttons
            cv2.circle(image, [int(palm[0] - circle_radius * np.cos(angle)), int(palm[1] - circle_radius * np.sin(angle))], 8, COLORS[i], -1, cv2.LINE_AA)
    
    
    def animate(self, image, qm_hand_keypoints, x_flip):
        """
        Animate quickmenu start up
        """
        circle_radius = np.linalg.norm(np.array(qm_hand_keypoints[9]) - np.array(qm_hand_keypoints[0])) * 1.7
        palm = [(qm_hand_keypoints[0][0] + qm_hand_keypoints[5][0] + qm_hand_keypoints[17][0])//3, (qm_hand_keypoints[0][1] + qm_hand_keypoints[5][1] + qm_hand_keypoints[17][1])//3]        
        
        if x_flip == True:
            palm[0] = image.shape[1] - palm[0] - 1  
            
        for i, angle in enumerate(self.button_angles):
            cv2.circle(image, [int(palm[0] - circle_radius * np.cos(angle) * (time.time() - self.animation_start)/self.ANIMATION_TIME), 
                               int(palm[1] - circle_radius * np.sin(angle) * (time.time() - self.animation_start)/self.ANIMATION_TIME)], 8, COLORS[i], -1, cv2.LINE_AA)
            
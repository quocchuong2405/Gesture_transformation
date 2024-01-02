import cv2
import mediapipe as mp
import numpy as np

# class that creates all the landmarks on the body.
class Landmark():
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        
        self.lh, self.rh, self.pose, self.face = None, None, None, None
        
        self.rh_raised = None
        self.lh_raised = None
        
        self.rh_angle = None
        self.lh_angle = None


    # Source: https://github.com/nicknochnack/ActionDetectionforSignLanguage    
    def mediapipe_detection(self, image, model):
        """
        Retrieve's landmarks from a given image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    # Source: https://github.com/nicknochnack/ActionDetectionforSignLanguage
    def draw_landmarks(self, image, results):
        """
        Draws the given landmarks on the given image
        """
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

    # Source: https://github.com/nicknochnack/ActionDetectionforSignLanguage
    def draw_styled_landmarks(self, image, results):
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
    
    # Source: https://github.com/nicknochnack/ActionDetectionforSignLanguage
    def extract_keypoints(self, results, image): 
        image_width, image_height = image.shape[1], image.shape[0]          
        lh, rh, pose, face = None, None, None, None
        if results.left_hand_landmarks:
            lh = np.array([[min(int(res.x * image_width), image_width - 1), min(int(res.y * image_height), image_height - 1)] for res in results.left_hand_landmarks.landmark])
        if results.right_hand_landmarks:
            rh = np.array([[min(int(res.x * image_width), image_width - 1), min(int(res.y * image_height), image_height - 1)] for res in results.right_hand_landmarks.landmark])
        if results.pose_landmarks:
            pose = np.array([[min(int(res.x * image_width), image_width - 1), min(int(res.y * image_height), image_height - 1)] for res in results.pose_landmarks.landmark])
        if results.face_landmarks:
            face = np.array([[min(int(res.x * image_width), image_width - 1), min(int(res.y * image_height), image_height - 1)] for res in results.face_landmarks.landmark])
        self.lh = lh
        self.rh = rh
        self.pose = pose
        self.face = face
    
    def calc_hand_rotations(self, x_flip=False):
        """
        Get rotation of both hands (in radians)
        """
        for hand_landmarks, handedness in [(self.lh, 'left'),  (self.rh, 'right')]:
            if hand_landmarks is None and handedness == 'left':
                self.lh_angle = None
                continue
            elif hand_landmarks is None and handedness == 'right':
                self.rh_angle = None
                continue

            # Angle is given by vector between wrist to base of middle finger
            vector = (np.array(hand_landmarks[9]) - np.array(hand_landmarks[0]))            
            if handedness == 'left':
                self.lh_angle = vector_angle(vector, x_flip=x_flip)
            elif handedness == 'right':
                self.rh_angle = vector_angle(vector, x_flip=x_flip)
    
    # Source: https://github.com/OwenTalmo/finger-counter (for all fingers except thumb)
    def calc_fingers_raised(self):
        """
        Calculates if a finger is raised or not
        """
        for hand_landmarks, handedness in [(self.lh, 'left'),  (self.rh, 'right')]:
            # raised = [Index, Middle, Ring, Pinky, Thumb]
            raised = [False, False, False, False, False]
            
            if hand_landmarks is None and handedness == 'left':
                self.lh_raised = None
                continue
            elif hand_landmarks is None and handedness == 'right':
                self.rh_raised = None
                continue
            
            for i in range(4):
                # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
                tip_y = hand_landmarks[i*4 + 8][1]
                dip_y = hand_landmarks[i*4 + 7][1]
                pip_y = hand_landmarks[i*4 + 6][1]
                mcp_y = hand_landmarks[i*4 + 5][1]
                if tip_y < min(dip_y,pip_y,mcp_y):
                    raised[i] = True
                
            # Check if back of hand 
            index_mcp_x = hand_landmarks[5][0]
            pinky_mcp_x = hand_landmarks[17][0]
                
            # Thumb
            # Not Raised when tip of thumb crosses line between index mcp and thumb mcp
            thumb_tip = np.array(hand_landmarks[4]) - np.array(hand_landmarks[1])
            hand_edge = np.array(hand_landmarks[5]) - np.array(hand_landmarks[1])
            hand_edge_angle = vector_angle(hand_edge)
            tip_angle = vector_angle(thumb_tip)
            
            # If left hand is flipped or if right hand is NOT flipped
            if (handedness == "left" and index_mcp_x < pinky_mcp_x) or (handedness == "right" and index_mcp_x < pinky_mcp_x):
                if tip_angle < hand_edge_angle:
                    raised[4] = True
            # If left hand is NOT flipped or if right hand is flipped
            elif (handedness == "left" and index_mcp_x > pinky_mcp_x) or (handedness == "right" and index_mcp_x > pinky_mcp_x):
                if tip_angle > hand_edge_angle:
                    raised[4] = True
            
            if handedness == 'left':
                self.lh_raised = raised
            else:
                self.rh_raised = raised
    
    # Depreciated
    def fingers_raised(self, hand_landmarks, handedness):
        # raised = [Index, Middle, Ring, Pinky, Thumb]
        raised = [False, False, False, False, False]
        for i in range(4):
            # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
            tip_y = hand_landmarks[i*4 + 8][1]
            dip_y = hand_landmarks[i*4 + 7][1]
            pip_y = hand_landmarks[i*4 + 6][1]
            mcp_y = hand_landmarks[i*4 + 5][1]
            if tip_y < min(dip_y,pip_y,mcp_y):
                raised[i] = True
                
        # Check if back of hand 
        index_mcp_x = hand_landmarks[5][0]
        pinky_mcp_x = hand_landmarks[17][0]
                
        # Thumb
        # Not Raised when tip of thumb crosses line between keypoints 5 and 1
        thumb_tip = np.array(hand_landmarks[4]) - np.array(hand_landmarks[1])
        hand_edge = np.array(hand_landmarks[5]) - np.array(hand_landmarks[1])
        hand_edge_angle = vector_angle(hand_edge)
        tip_angle = vector_angle(thumb_tip)
        
        # If left hand is flipped or if right hand is NOT flipped
        if (handedness == "left" and index_mcp_x < pinky_mcp_x) or (handedness == "right" and index_mcp_x < pinky_mcp_x):
            if tip_angle < hand_edge_angle:
                raised[4] = True
        # If left hand is NOT flipped or if right hand is flipped
        elif (handedness == "left" and index_mcp_x > pinky_mcp_x) or (handedness == "right" and index_mcp_x > pinky_mcp_x):
            if tip_angle > hand_edge_angle:
                raised[4] = True
        
        return raised

    # Depreciated
    def get_hand_rotation(self, hand_landmarks, handedness='right', flip=False):
        # Get rotation of the hand specified by handedness (in radians)
        if handedness not in ['left', 'right']:
            raise Exception("Invalid value of handedness, expected 'left' or 'right'")
               
        if hand_landmarks is None:
            return None

        # Angle is given by vector between wrist to base of middle finger
        vector = (np.array(hand_landmarks[9]) - np.array(hand_landmarks[0]))            
        return vector_angle(vector, flip=flip)
    
    # Depreciated
    def extract_keypoints_hand(self, results, image, handedness='right'): 
        image_width, image_height = image.shape[1], image.shape[0]          
        hand = None
        if handedness == 'left' and results.left_hand_landmarks:
            hand = np.array([[min(int(res.x * image_width), image_width - 1), min(int(res.y * image_height), image_height - 1)] for res in results.left_hand_landmarks.landmark])
        if handedness == 'right' and results.right_hand_landmarks:
            hand = np.array([[min(int(res.x * image_width), image_width - 1), min(int(res.y * image_height), image_height - 1)] for res in results.right_hand_landmarks.landmark])
        return hand
    
def vector_angle(vector, x_flip=False):
    """
    Returns the angle between the given vector and the x-axis 
    """
    if vector[1] > 0:
        if x_flip:
            return np.pi - (-1 * np.arccos(-1*vector[0]/np.linalg.norm(vector)))
        return -1 * np.arccos(-1*vector[0]/np.linalg.norm(vector))
    if x_flip:
        return np.pi - np.arccos(-1*vector[0]/np.linalg.norm(vector))
    return np.arccos(-1*vector[0]/np.linalg.norm(vector))
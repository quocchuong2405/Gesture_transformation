import os
import numpy as np
import cv2
from utils.landmark import Landmark
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Class to handle all data preprocessing/collection logic
class Collection():
    def __init__(self):
        self.letters = np.array(['P', 'M', 'C', 'L', 'J']) # Mediacontrol letters P, M, C, L, J
        # self.letters = np.array(['A', 'V', 'Q'])
        self.data_path = os.path.join(os.getcwd(), "Letters") # Zoom is A, V, Q
        self.no_sequences = 30
        self.sequence_length = 30
    
    # creates folder in current working directory, and creates a
    # folder for each letter in the letters needed. 
    # creates 30 sub-folders inside for training and testing data
    def CreateLetterFolders(self):
        for letter in self.letters:
            for sequence in range(self.no_sequences):
                try:
                    os.makedirs(os.path.join(self.data_path, letter, str(sequence)))
                except:
                    pass

    # NOTE: I THINK THIS SHOULD BE MOVED TO THE LANDMARK CLASS - AARON
    # extracts position of key values (fingers, face etc.)
    def extract_keypoints(self, results):
        #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    # loop to collect the data for each letter
    def CollectData(self):
        landmark = Landmark()
        cap = cv2.VideoCapture(0)

        # Set mediapipe model
        with landmark.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            # NEW LOOP
            # Loop through letters
            for letter in self.letters:
                # Loop through sequences aka videos
                for sequence in range(self.no_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(self.sequence_length):

                        # Read feed
                        ret, frame = cap.read()

                        # Make detections
                        image, results = landmark.mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        landmark.draw_styled_landmarks(image, results)

                        # NEW Apply wait logic
                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(letter, sequence), (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            #cv2.waitKey(2000)
                        else:
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(letter, sequence), (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)

                        # NEW Export keypoints
                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(self.data_path, letter, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

            cap.release()
            cv2.destroyAllWindows()

    # creates a train test split from the data
    def Preprocessing(self):
        label_map = {label:num for num, label in enumerate(self.letters)}
        sequences, labels = [], []

        for letter in self.letters:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.data_path, letter, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[letter])
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        return X_train, X_test, y_train, y_test
    
    # TRAINS a newral network model and saves the outcome as a .h5 file. 
    # differs from main,py as main.py loads, whereas this actually trains. 
    def model(self):
        X_train, X_test, y_train, y_test = self.Preprocessing()
        
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.letters.shape[0], activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        

        #batch_size = 128
        epochs = 50

        history = model.fit(X_train, y_train, epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[tb_callback])

        # Save the entire model, including architecture and weights
        model.save("Gesture_Model_Zoom.keras")

        # Plot the evaluation curve
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        return model

# this main loop is recundant, was created for testing. 
# if you want to train your own neural network, call collect data to create 
# your own data, then precrocessing, then model. 
def main():
    collection = Collection()
    #collection.CreateLetterFolders()
    #collection.CollectData()
    model = collection.model()
    #model = Sequential()
    #model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    #model.add(LSTM(128, return_sequences=True, activation='relu'))
    #model.add(LSTM(64, return_sequences=False, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(len(collection.letters), activation='softmax'))
    #model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])    
    #model.load_weights("word_test.keras")
    
    
    
if __name__ == "__main__":
    main()
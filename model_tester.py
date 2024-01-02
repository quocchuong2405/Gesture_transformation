import cv2
from utils.landmark import Landmark
from utils.collection import Collection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np


# gestures to implement
# pause
# mute
# forward
# backwards
# subtitles
# close






colors = [(245,117,16), (117,245,16), (16,117,245), (199, 27, 45), (33, 34, 60), (253, 97, 8)]

# function that for each letter, displays what letter it is plus associated propability
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame

# function that loads the weights from trained model
def Model():
    collection = Collection()

    # creates LSTM model and adds layers
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(collection.letters.shape[0], activation='softmax'))
    #model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights("model/models/Gesture_Model_Media.keras")
    # model.load_weights("Gesture_Model_Zoom.keras")
    return model

# main loop of program
def main():
    landmark = Landmark()
    collection = Collection()
    sequence = []
    sentence = []
    threshold = 0.8
    model = Model()

    # IMPORTANT!!!
    # need to change video capture dependant on what webcam using
    # generally, system webcam will be 0 and external one will be 1
    # need to play around with different integer values until work for own system
    # can be > 1
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with landmark.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = landmark.mediapipe_detection(frame, holistic)

            # Draw landmarks
            landmark.draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = collection.extract_keypoints(results)
            #  sequence.insert(0,keypoints)
            #  sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(collection.letters[np.argmax(res)])


            #3. Viz logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if collection.letters[np.argmax(res)] != sentence[-1]:
                            sentence.append(collection.letters[np.argmax(res)])
                    else:
                        sentence.append(collection.letters[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, collection.letters, image, colors)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Gesture - A universal gestural language
A desktop application that uses AI detected gestures to control your computer.

## Requirements
Windows operating system
Camera

### Dependencies
  * opencv-python 
  * mediapipe 
  * matplotlib 
  * numpy 
  * tensofrlow 
  * scikit-learn
  * pyautogui
  * psutil
  * flask
  * ctypes
  * keyboard

## Files
  Host.py
    This is the Main Program and everything will be run by launching Host.py 

## Running the Program

To run the program, launch Host.py with a virtual python environment that has the packages mentioned above.

If the camera doesn't appear or the wrong camera is used, refer to GestureProcessing.py - Line 104 and Host.py - Line 14:

```python
"""
IMPORTANT!!!
Change video capture dependent on what webcam you're using.
Generally, system webcam will be 0 and external one will be 1 (can be > 1)
e.g. cap = cv2.VideoCapture(2)

If camera doesn't work or an error occurs, remove/add cv2.CAP_DSHOW
"""
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    
```





<!--
  main.py: 
    The main logic that runs the program. Everything should be run through main.py. 
    
    !!!IMPORTANT!!!
    In line 102 of GestureProcessing.py and line 12 of Host.py the device chosen for cv2
    was device 1 as the model was trained on an external camera. 
    If using on own system, make sure you choose the correct camera device. 
    If its the system camera it should be camera 0, and an external camera should be camera 1. 
    However it is important to test this, and camera devices can be > 0. 

  collection.py:
    Class that handles the data process, from collection to training the neural network. 
    
    The most important function would be the collect data function, which runs for 30 frames for each folder for each letter. 
    Also the model function, which TRAINS (differs from main.py model) the neural network. 
    
    Generally, unless you want to override the existing model (which is in letter.h5), dont touch this class. 

  landmark.py:
    this class creates the landmarks on the face and body. Might be a redundant class so possibly might remove later on, but its a good way to visualise the data process. 
  
Notebooks:
    Spelling Checker - checks spelling for words and adds predictive corrections
    Text Correction Model - the neural neteork for text correction

Flask:
  Since No exe is provided, flask is run through visual studio or any other valid way of running flask (we have only tested Visual Studio).
  This tutorial should help to set up flask on visual studio: https://code.visualstudio.com/docs/python/tutorial-flask
  Once running Flask on Visual Studio in a virual enviroment you may need to reinstall some files for the VE.
--!>

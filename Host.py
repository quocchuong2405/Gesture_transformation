from flask import Flask, render_template,Response,request

from flaskwebgui import FlaskUI
import GestureProcessing
import cv2
import ctypes
import keyboard


app = Flask(__name__)
# accesses Camera
camera=cv2.VideoCapture(0, cv2.CAP_DSHOW)
#acesses Windows OS functionality
user32 = ctypes.windll.user32
# res[0] = x axis length res[1] = y axis height
res = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))

# How Users camera is read into the GUI
def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        # sends camera info to Webpage GUI
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Loads Launch Page of GUI
@app.route('/')
def homepage():
    # set specifc size and postion for Window housing GUI
    handle = user32.FindWindowW(None, u'PIP Page')
    if(handle):
        user32.MoveWindow(handle, 600, 300, 1250, 1000, True)
    # load GUI
    return render_template('huaban_3.html')

# Loads GUI for Gesture Detection functionality
@app.route('/pip')
def pip():
    # set specifc size and postion for Window housing GUI
    handle = user32.FindWindowW(None, u'Gesture')
    if(handle):
        user32.MoveWindow(handle, res[0]-500, res[1]-525, 500, 500, True)
    # load GUI
    return render_template('pip.html')

# used to load values from Gesture Processing code to GUI
@app.route('/video')
def video():
    return Response(GestureProcessing.ProcessGesture(),mimetype='multipart/x-mixed-replace; boundary=frame')

# used to load webcam data to GUI
@app.route('/cam')
def cam():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

#Closes the program
@app.route('/shutdown')
def shutdown1():
    keyboard.press_and_release('alt+f4')

# launches program
if __name__ == '__main__':
    # res[0] = x axis length res[1] = y axis height
    # launches program
    FlaskUI(app=app, server="flask",width=res[0] - 1250,height=res[1] - 500,fullscreen=False).run()

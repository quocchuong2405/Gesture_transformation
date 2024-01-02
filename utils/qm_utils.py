import pyautogui
import subprocess
import psutil
import threading

oskActive = False
procOSK = None #the osk process

def openOSK():
    """
    Opens the on-screen keyboard
    (and kills it if it is already open)
    """
    global procOSK, oskActive
    if not oskActive:
        procOSK = subprocess.Popen('C:\\Windows\\System32\\osk.exe', shell=True)
        oskActive = True
 
    elif oskActive:
        oskActive = False
        threading.Thread(target = killOSK, daemon=True).start()

def killOSK():
    """
    find on-screen keyboard process and kill it
    """
    process = psutil.Process(procOSK.pid)
    process.terminate()
    for proc in (process for process in psutil.process_iter() 
                            if process.name()=="osk.exe"):
        proc.kill()

def mouseCopy():
    pyautogui.hotkey('ctrl', 'c')
    
def mousePaste():
    pyautogui.hotkey('ctrl', 'v')

# QUICK MENU MODES
CURSOR = 0
MEDIA = 1
ZOOM = 2
MODE_SELECT = 3

class QMModes():
    """
    Stores current quick menu mode and states
    """
    def __init__(self):
        self.current_mode = None
        self.animate = False
        self.record_next = False
        
    def set_mode_cursor(self):
        self.current_mode = CURSOR
        self.animate = True
        
    def set_mode_media(self):
        self.current_mode = MEDIA
        self.animate = True
        
    def set_mode_zoom(self):
        self.current_mode = ZOOM
        self.animate = True
        
    def set_mode_select(self):
        self.current_mode = MODE_SELECT
        self.animate = True
    
    def input_gesture(self):
        if self.record_next == False:
            self.record_next = True
            
    
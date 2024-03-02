import cv2
from gaze_tracking import GazeTracking
import time
import random
import joblib
import pandas as pd
import numpy as np

RF_model = joblib.load('E:\\ADHD_Classification_Program\\RF_model_1.pkl')
scaler = joblib.load('E:\\ADHD_Classification_Program\\scaler.pkl')

# Capture video from camera
cap = cv2.VideoCapture(0)  



width = 640
height = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cat = cv2.imread('E:\\ADHD_Classification_Program\\cat.jpg')
cat = cv2.resize(cat, (width, height))
neutral = cv2.imread('E:\\ADHD_Classification_Program\\Neutral.jpg')
neutral = cv2.resize(neutral, (width, height))

# Define grid parameters
cell_width = width // 4  # Adapt width based on your camera resolution
cell_height = height // 4  # Adapt height based on your camera resolution

gaze = GazeTracking()

distraction_type = {
    1: 'grid',
    2: 'flat',
    3: 'cat',
    4: 'neutral'
}



def get_array_configurations(load):
    arrays = []
    for i in range(3):
        if load == 1:
            arrays.append([random.randint(1,16)])
        elif load == 2:
            arrays.append([random.randint(1,16), random.randint(1,16)]) 

    return arrays         


def draw_grid(frame):

    frame_width, frame_height, offset_x, offset_y = cal_dim(frame)

    for i in range(1, 4):
        cv2.line(frame, (i * cell_width + offset_x, 0), (i * cell_width + offset_x, frame_height), (0, 0, 0), 4)  # Vertical lines
        cv2.line(frame, (0, i * cell_height + offset_y), (frame_width, i * cell_height + offset_y), (0, 0, 0), 4)  # Horizontal lines
    # Draw circles in the middle of each cell
        
    return frame

def cal_dim(frame):
    # Calculate actual frame width and height
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Calculate offsets for centered grid
    offset_x = (frame_width - 4 * cell_width) // 2
    offset_y = (frame_height - 4 * cell_height) // 2

    return frame_width, frame_height, offset_x, offset_y

def draw_fix_cross(frame):
    center_x = width // 2 
    center_y = height // 2
    cv2.line(frame, (center_x - 50 , center_y), (center_x + 50, center_y), (0, 0, 0), 3)  # Vertical line
    cv2.line(frame, (center_x, center_y + 50), (center_x, center_y - 50), (0, 0, 0), 3)  # Horizontal line
    return frame

def draw_circles(frame, n):
    for i in n:
        draw_circle(frame, i)
    return frame    

def draw_distract(frame,type):
    if type == 'grid':
        frame = draw_grid(frame)
    elif type == 'flat':
        cv2.rectangle(frame, (0, 0), (width, height), (255,0,0), thickness=cv2.FILLED)
    elif type == 'cat':
        frame = cat
    elif type == 'neutral':
        frame = neutral

    return frame

def draw_circle(frame,n):
    frame = draw_grid(frame)
    for i in range(4):
        for j in range(4):
            center_x = cell_width * (j + 0.5)  # Calculate center of each cell
            center_y = cell_height * (i + 0.5)
            radius = min(cell_width, cell_height) // 4  # Set circle radius
            if ((i*4) + (j+1)) == n:
                cv2.circle(frame, (int(center_x), int(center_y)), radius, (0, 0, 255), -1)  # Draw filled circle
    return frame


def cross(current_time,frame,current_stage):
    frame = draw_fix_cross(frame)
    stage = current_stage
    if current_stage == 0:
        t = 1000
    else:
        t = 500    
    if int(time.perf_counter() * 1000) - current_time > t:
        current_time = int(time.perf_counter() * 1000)
        stage = current_stage + 1
    return stage, current_time

def array(current_time,frame, array,current_stage):
    frame = draw_circles(frame, array)
    stage = current_stage
    if int(time.perf_counter() * 1000) - current_time > 750:
        current_time = int(time.perf_counter() * 1000)
        stage = current_stage + 1
    return stage, current_time


def distract(current_time, frame, distraction, current_stage):
    frame = draw_distract(frame, distraction)
    stage = current_stage
    if int(time.perf_counter() * 1000) - current_time > 500:
        current_time = int(time.perf_counter() * 1000)
        stage = current_stage + 1
    return frame, stage, current_time

def check_test(arrays, test):
    for i in arrays:
        for j in test:
            if j in i:
                return True
    return False


task_data = []

for i in range(5):
    #time.sleep(1)
    distraction = distraction_type[random.randint(1,4)]
    load = random.randint(1,2)
    arrays = get_array_configurations(load)

    if random.randint(1,2) == 1:
        test = get_array_configurations(load)[0]
    else:
        test = arrays[2]   

    start_time = int(time.perf_counter() * 1000)
    current_time = start_time
    stage = 0
    decision = None
    test_result = None

    while True:
        # Capture a frame
        
        ret,frame = cap.read()
        gaze.refresh(frame)
        left_eye = gaze.pupil_left_coords()
        right_eye = gaze.pupil_right_coords()
        if left_eye and right_eye:
            x = (left_eye[0] + right_eye[0])/2
            y = (left_eye[1] + right_eye[1])/2
            #print("X:", x)
            #print("Y:", y)
        else:
            x = -1
            y = -1
        
        task_data.append([current_time - start_time, x, y])
        
        if stage == 0:
            # 1 - cross 
            stage, current_time = cross(current_time,frame, 0)
        elif stage == 1:
            # 2 - array 1 
            stage, current_time = array(current_time,frame, arrays[0],1)    
        elif stage == 2:
            # 3 - cross and 500 ms delay
            stage, current_time = cross(current_time,frame, 2)
        elif stage == 3:
            # 4 - array 2 and 750 ms delay
            stage, current_time = array(current_time,frame, arrays[1],3)
        elif stage == 4:
            # 5 - cross and 500 ms delay
            stage, current_time = cross(current_time,frame, 4)    
        elif stage == 5:
            # 4 - array 2 and 750 ms delay
            stage, current_time = array(current_time,frame, arrays[2],5)
        elif stage == 6:
            frame, stage, current_time = distract(current_time, frame, distraction, 6)
        elif stage == 7:
            frame = draw_circles(frame, test)
            test_result = check_test(arrays, test)
            key_event = cv2.waitKey(3) & 0xFF
            if int(time.perf_counter() * 1000) - current_time < 1500:
                if key_event == ord("y"):
                    decision = True
                    print("Pressed Yes")
                    print("Test Result:", test_result)
                    #break
                if key_event == ord("n"):
                    decision = False
                    print("Pressed No")
                    print("Test Result:", test_result)
                    #break
            else:
                break 

        
        # Display the resulting frame
        cv2.imshow("ADHD Test", frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(decision)
            break

# Release the capture and close all windows

task_data = pd.DataFrame(task_data)
task_data = scaler.transform(task_data)
prediction = RF_model.predict(task_data)
count_1 = np.sum(prediction == 1)
count_0 = np.sum(prediction == 0)

if count_1 > count_0:
    print("ADHD")
else:
    print("Non-ADHD")


#print()
cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2
import dlib
import imutils
import math
import keyboard
import time
import pyaudio
import cython_part

from multiprocessing import Process, Queue, Value

time_ms = lambda: int(round(time.time() * 1000))

def get_bbox(tc1, tc2, tc3):
    bbox = np.array([
        min(tc1[1], tc2[1], tc3[1]),
        max(tc1[1], tc2[1], tc3[1])-min(tc1[1], tc2[1], tc3[1]),
        min(tc1[0], tc2[0], tc3[0]),
        max(tc1[0], tc2[0], tc3[0])-min(tc1[0], tc2[0], tc3[0])],
        dtype = np.int_)
    return bbox

def display(img, name):

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.75
    fontColor              = (0,0,255)
    lineType               = 2

    cv2.putText(img, "Press E to exit", 
        (10,20), 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.imshow(name, img)
    cv2.waitKey(1)

def audio(stop_parameter, amplitude):
    CHUNK = 1024

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=22050,
        input=True,
        output=True, 
        frames_per_buffer=CHUNK) 

    while stop_parameter.value == 0:
        data = stream.read(CHUNK)
        #stream.write(data, CHUNK)

        avg = 0
        for i in range(0,1024,2):
            avg += float(int.from_bytes(data[i:i+2], byteorder='little', signed=False)) / 512 
        amplitude.value = int(avg)

    stream.close()
    p.terminate()
    
def main():

    v = Value('i', 0)
    amp = Value('i', 0)
    audio_thread = Process(target=audio, args=(v,amp,))
    audio_thread.start()

    size = (480, 640, 3)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    aug_state = 0

    groups = np.array([
    # TOP LEFT EYE
    [0,19,36],[19,36,38],[19,21,38],[21,38,39],[21,27,39],
    # TOP RIGHT EYE
    [16,24,45],[24,43,45],[22,24,43],[22,42,43],[22,27,42],
    # TOP "THIRD" EYE
    [21,22,27],
    # BOT LEFT EYE
    [0,2,36],[2,36,40],[2,30,40],[30,39,40],[27,30,39],
    # BOT RIGHT EYE
    [14,16,45],[14,45,47],[14,30,47],[30,42,47],[27,30,42],
    # MIDDLE FACE
    [2,30,48],[30,48,51],[2,4,48],[14,30,54],[30,51,54],[12,14,54],
    # BOT LIPS
    [4,6,48],[6,48,51],[6,8,51],[10,12,54],[10,54,51],[8,10,51]
    # LEFT EYE
    #[36,38,40],[38,39,40],
    # RIGHT EYE
    #[42,43,47],[43,45,47],
    # LIPS
    #[48,51,57],[51,54,57],
    ])   
    
    """
    groups = np.array([
        [0,19,27],[27,24,16],[19,24,27],
        [0,3,27],[16,27,13],[3,27,33],[27,33,13],
        [3,33,57],[33,57,13],[13,10,57],[3,6,57],
        [6,8,57],[10,8,57]
    ]) 
    """

    camera_1_points = np.ndarray((68,2), dtype = np.int_)
    camera_2_points = np.ndarray((68,2), dtype = np.int_)
    
    camera_1 = cv2.VideoCapture(701)
    camera_2 = cv2.VideoCapture(0)

    time_prev = time_ms()
    
    triangles_0 = 0
    triangles_1 = 0
    triangles_2 = 0

    time_duration = 10

    while(True):

        _, camera_frame_1 = camera_1.read()
        _, camera_frame_2 = camera_2.read() 
        
        camera_copy_1 = np.copy(camera_frame_1)
        camera_copy_2 = np.copy(camera_frame_2)

        if time_prev + time_duration < time_ms():
            time_prev = time_ms()

            if amp.value > 1000:
                if triangles_0 < len(groups):
                    triangles_0 += 3
                    if triangles_0 > len(groups):
                        triangles_0 = len(groups)

                if triangles_1 < len(groups):
                    triangles_1 += 2
                    if triangles_1 > len(groups):
                        triangles_1 = len(groups)

                if triangles_2 < len(groups):
                    triangles_2 += 1
                    if triangles_2 > len(groups):
                        triangles_2 = len(groups)

            else:
                if triangles_0 > 0:
                    triangles_0 -= 1
                    if triangles_0 < 0:
                        triangles_0 = 0

                if triangles_1 > 0:
                    triangles_1 -= 2
                    if triangles_1 < 0:
                        triangles_1 = 0

                if triangles_2 > 0:
                    triangles_2 -= 3
                    if triangles_2 < 0:
                        triangles_2 = 0

        camera_frame_gray_1 = cv2.cvtColor(camera_frame_1, cv2.COLOR_BGR2GRAY)    
        camera_frame_gray_2 = cv2.cvtColor(camera_frame_2, cv2.COLOR_BGR2GRAY)  
        camera_face_1 = detector(camera_frame_gray_1)
        camera_face_2 = detector(camera_frame_gray_2)

        #camera_marks_1 = False
        for face in camera_face_1:
            landmarks = predictor(image=camera_frame_gray_1, box=face)
            for n in range(0, 68):
                #camera_marks_1 = True
                camera_1_points[n,0] = landmarks.part(n).y
                camera_1_points[n,1] = landmarks.part(n).x

        #camera_marks_2 = False
        for face in camera_face_2:
            landmarks = predictor(image=camera_frame_gray_2, box=face)
            for n in range(0, 68):
                #camera_marks_2 = True
                camera_2_points[n,0] = landmarks.part(n).y
                camera_2_points[n,1] = landmarks.part(n).x

        """
        if not camera_marks_1:
            for n in range(0, 68):
                camera_1_points[n,0] = 0
                camera_1_points[n,1] = 0

        if not camera_marks_2:
            for n in range(0, 68):
                camera_1_points[n,0] = 0
                camera_1_points[n,1] = 0

        camera_dx_1 = (camera_1_points[36,0] - camera_1_points[45,0])
        camera_dy_1 = (camera_1_points[36,1] - camera_1_points[45,1])

        camera_dx_2 = (camera_2_points[36,0] - camera_2_points[45,0])
        camera_dy_2 = (camera_2_points[36,1] - camera_2_points[45,1])

        camera_1_face_angle = math.degrees(math.atan(camera_dy_1/camera_dx_1))  
        camera_2_face_angle = math.degrees(math.atan(camera_dy_2/camera_dx_2)) 

        camera_copy_1 = imutils.rotate_bound(camera_copy_1, camera_2_face_angle)  
        camera_copy_2 = imutils.rotate_bound(camera_copy_2, camera_1_face_angle)       
        """

        for i in range(triangles_0):
            group = groups[i]
        
            point_camera_1_1 = camera_1_points[group[0]]
            point_camera_1_2 = camera_1_points[group[1]]
            point_camera_1_3 = camera_1_points[group[2]]
            
            point_camera_2_1 = camera_2_points[group[0]]
            point_camera_2_2 = camera_2_points[group[1]]
            point_camera_2_3 = camera_2_points[group[2]]
        
            bbox_camera_1 = get_bbox(point_camera_1_1, point_camera_1_2, point_camera_1_3) 
            bbox_camera_2 = get_bbox(point_camera_2_1, point_camera_2_2, point_camera_2_3)              
            
            points_c1 = np.ndarray((3,2), dtype = np.int_)
            points_c1[0,:] = point_camera_1_1
            points_c1[1,:] = point_camera_1_2  
            points_c1[2,:] = point_camera_1_3         

            points_c2 = np.ndarray((3,2), dtype = np.int_)
            points_c2[0,:] = point_camera_2_1
            points_c2[1,:] = point_camera_2_2  
            points_c2[2,:] = point_camera_2_3      

            if i < triangles_2:
                camera_frame_1 = cython_part.map_face_check(camera_frame_1, camera_copy_2, bbox_camera_1, bbox_camera_2, points_c1)
                camera_frame_2 = cython_part.map_face_check(camera_frame_2, camera_copy_1, bbox_camera_2, bbox_camera_1, points_c2)

            elif i < triangles_1:

                center_x = max(0, min(639, int((point_camera_1_1[1] + point_camera_1_2[1] + point_camera_1_3[1]) / 3)))
                center_y = max(0, min(479, int((point_camera_1_1[0] + point_camera_1_2[0] + point_camera_1_3[0]) / 3)))
                avg_2 = camera_frame_1[center_y, center_x, :]

                center_x = max(0, min(639, int((point_camera_2_1[1] + point_camera_2_2[1] + point_camera_2_3[1]) / 3)))
                center_y = max(0, min(479, int((point_camera_2_1[0] + point_camera_2_2[0] + point_camera_2_3[0]) / 3)))
                avg_1 = camera_frame_2[center_y, center_x, :]

                camera_frame_1 = cython_part.map_avg_check(camera_frame_1, avg_1, bbox_camera_1, bbox_camera_2, points_c1)
                camera_frame_2 = cython_part.map_avg_check(camera_frame_2, avg_2, bbox_camera_2, bbox_camera_1, points_c2)

            else:
                p1 = (point_camera_1_1[1], point_camera_1_1[0])
                p2 = (point_camera_1_2[1], point_camera_1_2[0])
                p3 = (point_camera_1_3[1], point_camera_1_3[0])

                cv2.line(camera_frame_1, p1, p2, (0, 255, 0), 2) 
                cv2.line(camera_frame_1, p2, p3, (0, 255, 0), 2) 
                cv2.line(camera_frame_1, p3, p1, (0, 255, 0), 2) 

                p1 = (point_camera_2_1[1], point_camera_2_1[0])
                p2 = (point_camera_2_2[1], point_camera_2_2[0])
                p3 = (point_camera_2_3[1], point_camera_2_3[0])

                cv2.line(camera_frame_2, p1, p2, (0, 255, 0), 2) 
                cv2.line(camera_frame_2, p2, p3, (0, 255, 0), 2) 
                cv2.line(camera_frame_2, p3, p1, (0, 255, 0), 2)

        display(camera_frame_1, "F1")
        display(camera_frame_2, "F2")
        
        if keyboard.is_pressed('q'):
            face_index = (face_index + 1) % len(faces)

        if keyboard.is_pressed('w'):
            aug_state = (aug_state + 1) % 3
            
        if keyboard.is_pressed('e'):
            break

    v.value = 1            
    camera_1.release()
    camera_2.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()

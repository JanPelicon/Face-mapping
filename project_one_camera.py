import numpy as np
import cv2
import dlib
import imutils
import math
import keyboard
import os
import time
import pyaudio
import threading
import cython_part
import sys

from multiprocessing import Process, Queue, Value

time_ms = lambda: int(round(time.time() * 1000))

def inside(A,B,C,P):
    v0 = C - A
    v1 = B - A
    v2 = P - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) and (v >= 0) and (u + v < 1)

def get_bbox(tc1, tc2, tc3):
    bbox = np.array([
        min(tc1[1], tc2[1], tc3[1]),
        max(tc1[1], tc2[1], tc3[1])-min(tc1[1], tc2[1], tc3[1]),
        min(tc1[0], tc2[0], tc3[0]),
        max(tc1[0], tc2[0], tc3[0])-min(tc1[0], tc2[0], tc3[0])],
        dtype = np.int_)
    return bbox

def load_face(index, faces, size):
    image = cv2.imread("./faces/" + faces[index])

    scale = image.shape[1] / size[1]
    width = int(image.shape[1] / scale)
    height = int(image.shape[0] / scale)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    return image

def display(img, name):

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2

    cv2.putText(img, "Press Q to change face", 
        (10,30), 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.putText(img, "Press E to exit", 
        (10,60), 
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

    faces = os.listdir("./faces")
    face_index = 0

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
        [0,3,27],[16,27,13],[3,27,33],[27,33,13],
        [3,33,57],[33,57,13],[13,10,57],[3,6,57],
        [0,19,27],[27,24,16],[19,24,27],
        [6,8,57],[10,8,57]
    ]) 
    """

    camera_points = np.ndarray((68,2), dtype = np.int_)
    image_points = np.ndarray((68,2), dtype = np.int_)

    camera = cv2.VideoCapture(int(sys.argv[1]))
    
    time_prev = time_ms()
    
    triangles_0 = 0
    triangles_1 = 0
    triangles_2 = 0

    time_duration = 10

    while(True):
        _, camera_frame = camera.read() 
        
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
        
        camera_frame_gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)    
        camera_face = detector(camera_frame_gray)
        
        for face in camera_face:
            landmarks = predictor(image=camera_frame_gray, box=face)
            for n in range(0, 68):
                camera_points[n,0] = landmarks.part(n).y
                camera_points[n,1] = landmarks.part(n).x
            
        image = load_face(face_index, faces, size)

        dx = (camera_points[36,1] - camera_points[45,1])
        dy = (camera_points[36,0] - camera_points[45,0])
        if dx == 0:
            dx = 0.01
            
        camera_face_angle = math.degrees(math.atan(dy/dx))   
        image_rotated = imutils.rotate_bound(image, camera_face_angle)       
        image_rotated_gray = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2GRAY)  
        image_face = detector(image_rotated_gray)
        
        for face in image_face:
            landmarks = predictor(image=image_rotated_gray, box=face)
            for n in range(0, 68):
                image_points[n,0] = landmarks.part(n).y
                image_points[n,1] = landmarks.part(n).x
        
        for i in range(triangles_0):
            group = groups[i]

            point_camera_1 = camera_points[group[0]]
            point_camera_2 = camera_points[group[1]]
            point_camera_3 = camera_points[group[2]]
            
            point_face_1 = image_points[group[0]]
            point_face_2 = image_points[group[1]]
            point_face_3 = image_points[group[2]]
        
            bbox_frame = get_bbox(point_camera_1, point_camera_2, point_camera_3) 
            bbox_face = get_bbox(point_face_1, point_face_2, point_face_3)              

            points_c1 = np.ndarray((3,2), dtype = np.int_)
            points_c1[0,:] = point_camera_1
            points_c1[1,:] = point_camera_2  
            points_c1[2,:] = point_camera_3     

            if i < triangles_2:
                #camera_frame = cython_part.map_face(camera_frame, image_rotated, bbox_frame, bbox_face)
                camera_frame = cython_part.map_face_check(camera_frame, image_rotated, bbox_frame, bbox_face, points_c1)
            elif i < triangles_1:
                center_x = max(0, min(639, int((point_camera_1[1] + point_camera_2[1] + point_camera_3[1]) / 3)))
                center_y = max(0, min(479, int((point_camera_1[0] + point_camera_2[0] + point_camera_3[0]) / 3)))
                avg = camera_frame[center_y, center_x, :]
                camera_frame = cython_part.map_avg_check(camera_frame, avg, bbox_frame, bbox_face, points_c1)
            
            else:
                p1 = (point_camera_1[1], point_camera_1[0])
                p2 = (point_camera_2[1], point_camera_2[0])
                p3 = (point_camera_3[1], point_camera_3[0])

                cv2.line(camera_frame, p1, p2, (0, 255, 0), 2) 
                cv2.line(camera_frame, p2, p3, (0, 255, 0), 2) 
                cv2.line(camera_frame, p3, p1, (0, 255, 0), 2) 

        display(camera_frame, "F1")
        
        if keyboard.is_pressed('q'):
            face_index = (face_index + 1) % len(faces)
            
        if keyboard.is_pressed('e'):
            break

    v.value = 1        
    camera.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()

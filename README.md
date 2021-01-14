#In order for program to work you first need to download:

shape_predictor_68_face_landmarks.dat.bz2 (https://github.com/davisking/dlib-models) and extract it.

#2 different programs:

project_one_camera.py camera1_index

project_two_camera.py camera1_index camera2_index

#Obtain camera index-es with:

get_camera_index.py 

#Requirements:

pip install -r requirements.txt

pip install pipwin

pipwin install pyaudio

python setup.py build_ext --inplace

**or if that does not work:**

pip install opencv-python

pip install dlib

pip install imutils

pip install keyboard

pip install cython

pip install pipwin

pipwin install pyaudio

python setup.py build_ext --inplace

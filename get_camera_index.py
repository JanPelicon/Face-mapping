import cv2
print("Wait a few moments for script to finnish...")

available = []

for index in range(1000):
    camera = cv2.VideoCapture(index)
    if camera.isOpened():
        available.append(index)
    camera.release()
    cv2.destroyAllWindows()
    
for camera in available:
    print("Found camera on index", camera)
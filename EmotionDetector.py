import numpy as np
import cv2
import torch
import cv2
#Importing the model

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/last.pt', force_reload=True)

# Sample test run

# img = os.path.join('data', 'images', 'awake.8b4294e5-c7f0-11ee-84ae-b40ede9bed9d.jpg')
# results = model(img)
# results.print()

# To rune the new trained model
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
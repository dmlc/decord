"""Benchmark using opencv's VideoCapture"""
import time
import cv2

cap = cv2.VideoCapture('test2.mp4')
tic = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
print('elapsed time for sequential read: ', time.time() - tic)

cap.release()
cv2.destroyAllWindows()
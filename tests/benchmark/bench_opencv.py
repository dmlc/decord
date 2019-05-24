"""Benchmark using opencv's VideoCapture"""
import time
import random
import numpy as np
import cv2

test_video = '/tmp/testsrc_h264_100s_default.mp4'

def cv2_seek_frame(cap, pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

class CV2VideoLoader(object):
    def __init__(self, fns, shape, interval, skip, shuffle):
        self._shape = shape
        self._interval = interval
        self._skip = skip
        self._shuffle = shuffle
        self._cap = [cv2.VideoCapture(fn) for fn in fns]
        # for cap in self._cap:
        #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, shape[2])
        #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[1])
        self._frame_len = sum([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self._cap])
        self._curr = 0
        self._init_orders()
        self.reset()

    def __del__(self):
        for cap in self._cap:
            cap.release()
    
    def _init_orders(self):
        self._orders = []
        for i, cap in enumerate(self._cap):
            l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            order = np.arange(0, l, self._shape[0] * (1 + self._interval) - self._interval + self._skip)
            for o in order:
                self._orders.append((i, o))
        
    def reset(self):
        self._curr = 0
        random.shuffle(self._orders)

    def __len__(self):
        return len(self._orders)

    def __next__(self):
        if self._curr >= self.__len__():
            raise StopIteration

        i, o = self._orders[self._curr]
        cap = self._cap[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, o)
        data = np.empty(shape=self._shape)
        for j in range(self._shape[0]):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError
            data[j][:] = cv2.resize(frame, (self._shape[2], self._shape[1])) 
            for k in range(self._interval):
                ret, frame = cap.read()
        self._curr += 1
        return 

    def __iter__(self):
        return self

cap = cv2.VideoCapture(test_video)
tic = time.time()
cnt = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    cnt += 1
print(cnt, ' frames. Elapsed time for sequential read: ', time.time() - tic)

cap.release()
cv2.destroyAllWindows()


# vl = CV2VideoLoader([test_video], (2, 320, 240, 3), 1, 5, 1)
# cnt = 0
# tic = time.time()
# for batch in vl:
#     cnt += 1

# print(cnt, ' batches. Elapsed time for (not accurate) random access: ', time.time() - tic)

"""Benchmark using opencv's VideoCapture"""
import time
import random
import numpy as np
import cv2
import argparse
import warnings
import numpy as np

parser = argparse.ArgumentParser("OpenCV benchmark")
parser.add_argument('--file', type=str, default='/tmp/testsrc_h264_100s_default.mp4', help='Test video')
parser.add_argument('--seed', type=int, default=666, help='numpy random seed for random access indices')
parser.add_argument('--random-frames', type=int, default=300, help='number of random frames to run')
parser.add_argument('--width', type=int, default=320, help='resize frame width')
parser.add_argument('--height', type=int, default=240, help='resize frame height')

args = parser.parse_args()

def cv2_seek_frame(cap, pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    #print(pos, cap.get(cv2.CAP_PROP_POS_FRAMES))

class CV2VideoReader(object):
    def __init__(self, fn, width, height):
        self._cap = cv2.VideoCapture(fn)
        self._len = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = width
        self._height = height
        self._curr = 0

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        cv2_seek_frame(self._cap, idx)
        self._curr = idx
        return self.__next__()

    def __del__(self):
        self._cap.release()

    def __next__(self):
        if self._curr >= self.__len__():
            raise StopIteration

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError
        ret = cv2.resize(frame, (self._width, self._height))
        self._curr += 1
        return ret

    def next(self):
        return self.__next__()


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

vr = CV2VideoReader(args.file, width=args.width, height=args.height)
tic = time.time()
cnt = 0
while True:
    try:
        frame = vr.__next__()
        cnt += 1
    except:
        break
print(cnt, ' frames. Elapsed time for sequential read: ', time.time() - tic)

vr = CV2VideoReader(args.file, width=args.width, height=args.height)
np.random.seed(args.seed)  # fix seed for all random tests
acc_indices = np.arange(len(vr))
np.random.shuffle(acc_indices)
if args.random_frames > len(vr):
    warnings.warn('Number of random frames reduced to {} to fit test video'.format(len(vr)))
    args.random_frames = len(vr)
indices = acc_indices[:args.random_frames]

tic = time.time()
for idx in indices:
    frame = vr[idx]

print(len(indices), ' frames, elapsed time for random access(not accurate): ', time.time() - tic)

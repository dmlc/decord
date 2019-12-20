"""Benchmark using opencv's VideoCapture"""
import time
import random
import numpy as np
import av
import argparse
import warnings
import numpy as np
import pims
import cv2

parser = argparse.ArgumentParser("PyAV benchmark")
parser.add_argument('--file', type=str, default='/tmp/testsrc_h264_100s_default.mp4', help='Test video')
parser.add_argument('--seed', type=int, default=666, help='numpy random seed for random access indices')
parser.add_argument('--random-frames', type=int, default=300, help='number of random frames to run')
parser.add_argument('--width', type=int, default=320, help='resize frame width')
parser.add_argument('--height', type=int, default=240, help='resize frame height')

args = parser.parse_args()


class PyAVVideoReader(object):
    def __init__(self, fn, width, height, any_frame=False):
        self._cap = pims.Video(fn)
        self._len = len(self._cap)
        self._width = width
        self._height = height
        self._curr = 0

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._curr = idx
        frame = self._cap[idx]
        frame = cv2.resize(frame, (self._width, self._height))
        return frame

    def __next__(self):
        if self._curr >= self.__len__():
            raise StopIteration

        return self.__getitem__(self._curr + 1)

    def next(self):
        return self.__next__()

vr = PyAVVideoReader(args.file, width=args.width, height=args.height)
tic = time.time()
cnt = 0
while True:
    try:
        frame = vr.__next__()
        cnt += 1
    except:
        break
print(cnt, ' frames. Elapsed time for sequential read: ', time.time() - tic)

vr = PyAVVideoReader(args.file, width=args.width, height=args.height)
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

print(len(indices), ' frames, elapsed time for random access(accurate): ', time.time() - tic)































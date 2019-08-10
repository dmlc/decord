"""Benchmark using opencv's VideoCapture"""
import time
import sys
import os
import argparse
import warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import decord as de

parser = argparse.ArgumentParser("Decord benchmark")
parser.add_argument('--gpu', type=int, default=-1, help='context to run, use --gpu=-1 to use cpu only')
parser.add_argument('--file', type=str, default='/tmp/testsrc_h264_100s_default.mp4', help='Test video')
parser.add_argument('--seed', type=int, default=666, help='numpy random seed for random access indices')
parser.add_argument('--random-frames', type=int, default=300, help='number of random frames to run')
parser.add_argument('--width', type=int, default=320, help='resize frame width')
parser.add_argument('--height', type=int, default=240, help='resize frame height')

args = parser.parse_args()

test_video = args.file
if args.gpu > -1:
    ctx = de.gpu(args.gpu)
else:
    ctx = de.cpu()

vr = de.VideoReader(test_video, ctx, width=args.width, height=args.height)
cnt = 0
tic = time.time()
while True:
    try:
        frame = vr.next()
    except StopIteration:
        break
    cnt += 1
print(cnt, ' frames, elapsed time for sequential read: ', time.time() - tic)

np.random.seed(args.seed)  # fix seed for all random tests
acc_indices = np.arange(len(vr))
np.random.shuffle(acc_indices)
if args.random_frames > len(vr):
    warnings.warn('Number of random frames reduced to {} to fit test video'.format(len(vr)))
    args.random_frames = len(vr)
indices = acc_indices[:args.random_frames]

vr.seek(0)
tic = time.time()
for idx in indices:
    frame = vr[idx]

print(len(indices), ' frames, elapsed time for random access: ', time.time() - tic)

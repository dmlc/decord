"""Benchmark using opencv's VideoCapture"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import decord as de

test_video = '/tmp/testsrc_h264_100s_default.mp4'

vr = de.VideoReader(test_video, de.cpu(), -1, -1)
cnt = 0
tic = time.time()
while True:
    try:
        frame = vr.next()
    except StopIteration:
        break
    cnt += 1
frame = frame.asnumpy()
print(cnt, ' frames, elapsed time for sequential read: ', time.time() - tic)

# vl = de.VideoLoader([test_video], ctx=de.cpu(), shape=(2, 180, 320, 3), interval=1, skip=5, shuffle=2)

# cnt = 0
# tic = time.time()
# for batch in vl:
#     cnt += 1

# batch[0].asnumpy()
# print(cnt, ' batches, elapsed time for random access: ', time.time() - tic)
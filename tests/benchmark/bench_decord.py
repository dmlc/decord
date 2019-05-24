"""Benchmark using opencv's VideoCapture"""
import time
import os
import decord as de

curr_dir = os.path.dirname(__file__)

test_video = os.path.join(curr_dir, '..', 'testsrc_h264_100s_default.mp4')

vr = de.VideoReader(test_video, -1, -1)
cnt = 0
tic = time.time()
while True:
    try:
        frame = vr.next()
    except StopIteration:
        break
    cnt += 1
print(cnt, ' frames, elapsed time for sequential read: ', time.time() - tic)

vl = de.VideoLoader([test_video], shape=(2, 320, 240, 3), interval=1, skip=5, shuffle=2)

cnt = 0
tic = time.time()
for batch in vl:
    cnt += 1

batch[0].asnumpy()
print(cnt, ' batches, elapsed time for random access: ', time.time() - tic)
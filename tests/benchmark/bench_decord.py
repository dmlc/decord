"""Benchmark using opencv's VideoCapture"""
import time
import decord as de

vr = de.VideoReader('/Users/zhiz/Dev/decord/build/test2.mp4', -1, -1)
print(vr.num_frame)
print(vr.get_key_indices())
tic = time.time()
while True:
    try:
        frame = vr.next()
    except StopIteration:
        break
print('elapsed time for sequential read: ', time.time() - tic)


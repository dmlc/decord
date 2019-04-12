"""Benchmark using opencv's VideoCapture"""
import time
import decord as de

vr = de.VideoReader('/Users/zhiz/Dev/decord/build/test2.mp4', -1, -1)
print(vr.num_frame)
print(vr.get_key_indices())
cnt = 0
tic = time.time()
while True:
    try:
        frame = vr.next()
    except StopIteration:
        break
    cnt += 1
print(cnt, ' frames, elapsed time for sequential read: ', time.time() - tic)

vl = de.VideoLoader(['/Users/zhiz/Dev/decord/build/test2.mp4'], shape=(2, 320, 240, 3), interval=1, skip=5, shuffle=2)

cnt = 0
tic = time.time()
for batch in vl:
    cnt += 1
    pass

print(cnt, ' batches, elapsed time for random access: ', time.time() - tic)
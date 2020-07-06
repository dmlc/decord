import av
import numpy as np
import decord


def analysis(filename):
    container = decord.VideoReader(filename)
    container_av = av.open(filename)
    imgs = list()
    for frame in container_av.decode(video=0):
        imgs.append(frame.to_rgb().to_ndarray())
    frame_list = np.random.randint(1, len(imgs), size=10)
    frame_list = [38, 58, 70]
    frame_list = [30, 18]
    incorrect = False
    print("==============", frame_list)
    for frame_idx in frame_list:
        print("======", frame_idx)
        frame_decord = container[frame_idx].asnumpy()
        frame_pyav = imgs[frame_idx]
        if np.sum(abs(frame_pyav - frame_decord)) != 0:
            print("XXXXXXXXXX _ XXXXXXXXX")
            print("index ", frame_idx, " has different result: ", np.sum(abs(frame_pyav - frame_decord)))
            incorrect = True
    return incorrect


print(analysis("decordvid/3N9LxIDyfMk_000008_000018.mp4"))

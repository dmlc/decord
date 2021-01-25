import os
from decord import AudioReader, cpu, gpu
from decord.base import DECORDError

CTX = cpu(0)

def test_audio_reader():
    ar = AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'audio', 'test.mov'), CTX)

if __name__ == '__main__':
    import nose
    nose.runmodule()

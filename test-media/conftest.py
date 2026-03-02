"""Shared fixtures for decord tests."""
import os
import pytest
import numpy as np

from decord import VideoReader, AudioReader, AVReader, cpu


MEDIA_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(MEDIA_DIR, '..', 'tests', 'test_data')

# Primary test file: big_buck_bunny.mp4
# - 1440 video frames, 640x360, ~24fps, ~60s
# - AAC stereo audio at 22050 Hz
BBB_PATH = os.path.join(MEDIA_DIR, 'big_buck_bunny.mp4')

# Small test video (video-only fixtures)
PANCAKE_PATH = os.path.join(MEDIA_DIR, '..', 'examples', 'flipping_a_pancake.mkv')

# MP3 audio-only file: ~878s, mono, 44100 Hz
MP3_PATH = os.path.join(MEDIA_DIR, '26_Universität_Wien_Informationen-Audioversion-ElevenLabs_20260123_final.mp3')

# Corrupted video
CORRUPTED_PATH = os.path.join(TEST_DATA_DIR, 'corrupted.mp4')

# Rotation test videos
ROTATION_VIDEOS = {
    rot: os.path.join(TEST_DATA_DIR, f'video_{rot}.mov')
    for rot in [0, 90, 180, 270]
}

# Unordered PTS video
UNORDERED_PATH = os.path.join(TEST_DATA_DIR, 'unordered.mov')

# Video-only file (no audio stream)
VIDEO_ONLY_PATH = os.path.join(TEST_DATA_DIR, 'video_0.mov')

CTX = cpu(0)


@pytest.fixture
def bbb_video():
    """VideoReader for big_buck_bunny.mp4."""
    return VideoReader(BBB_PATH, ctx=CTX)


@pytest.fixture
def bbb_audio():
    """AudioReader for big_buck_bunny.mp4 (mono)."""
    return AudioReader(BBB_PATH, ctx=CTX, mono=True)


@pytest.fixture
def bbb_audio_stereo():
    """AudioReader for big_buck_bunny.mp4 (stereo)."""
    return AudioReader(BBB_PATH, ctx=CTX, mono=False)


@pytest.fixture
def bbb_av():
    """AVReader for big_buck_bunny.mp4."""
    return AVReader(BBB_PATH, ctx=CTX)


@pytest.fixture
def pancake_video():
    """VideoReader for flipping_a_pancake.mkv."""
    return VideoReader(PANCAKE_PATH, ctx=CTX)


@pytest.fixture
def mp3_audio():
    """AudioReader for the MP3 test file (mono, 44100 Hz)."""
    return AudioReader(MP3_PATH, ctx=CTX, mono=True)

# Decord

![symbol](docs/symbol.png)

`Decord` is a reverse procedure of `Record`. It provides convenient video slicing methods based on a thin wrapper on top of hardware accelerated video decoders, e.g.

-   FFMPEG/LibAV(Done)
-   Nvidia Codecs(Done)
-   Intel Codecs

`Decord` was designed to handle awkward video shuffling experience in order to provide smooth experiences similar to random image loader for deep learning.

Bridges for deep learning frameworks:

-   Apache MXNet (Done)

## Installation

### Install via pip

TODO

### Install from source

#### Linux

Install the system packages for building the shared library, for Debian/Ubuntu users, run:

```bash
# official PPA comes with ffmpeg 2.8, which lacks tons of features, we use ffmpeg 4.0 here
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-setuptools make cmake
libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
# note: make sure you have cmake 3.8 or later, you can install from cmake official website if it's too old
```

Clone the repo recursively(important)

```bash
git clone --recursive https://github.com/zhreshold/decord
```

Build the shared library in source root directory, you can specify `-DUSE_CUDA=1` to enable NVDEC hardware accelerated decoding:

```bash
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=0
make
```

Install python bindings:

```bash
cd ../python
# option 1: add python path to $PYTHONPATH, you will need to install numpy separately
pwd=$PWD
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc
source ~/.bashrc
# option 2: install with setuptools
python3 setup.py install --user
```

#### Mac OS

Installation on macOS is similar to Linux. But macOS users need to install building tools like clang, GNU Make, cmake first.

Tools like clang and GNU Make are packaged in _Command Line Tools_ for macOS. To install:

```bash
xcode-select --install
```

To install other needed packages like cmake, we recommend first installing Homebrew, which is a popular package manager for macOS. Detailed instructions can be found on its [homepage](https://brew.sh/).

After installation of Homebrew, install cmake by:

```bash
brew install cmake
# note: make sure you have cmake 3.8 or later, you can install from cmake official website if it's too old
```

Clone the repo recursively(important)

```bash
git clone --recursive https://github.com/zhreshold/decord
```

Then go to root directory build shared library:

```bash
cd decord
mkdir build && cd build
cmake ..
make
```

Install python bindings:

```bash
cd ../python
# option 1: add python path to $PYTHONPATH, you will need to install numpy separately
pwd=$PWD
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bash_profile
source ~/.bash_profile
# option 2: install with setuptools
python3 setup.py install --user
```

#### Windows

For windows, you will need CMake and Visual Studio for C++ compilation.

-   First, install `git`, `cmake`, `ffmpeg` and `python`. You can use [Chocolatey](https://chocolatey.org/) to manage packages similar to Linux/Mac OS.
-   Second, install [`Visual Studio 2017 Community`](https://visualstudio.microsoft.com/), this my take some time.

When dependencies are ready, open command line prompt:

```bash
cd your-workspace
git clone --recursive https://github.com/zhreshold/decord
cd decord
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="/DDECORD_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" -G "Visual Studio 15 2017 Win64" ..
# open `decord.sln` and build project
```

## Usage

Decord provides minimal API set for bootstraping. You can also check out jupyter notebook [examples](examples/).

### VideoReader

VideoReader is used to access frames directly from video files.

```python
from decord import VideoReader
from decord import cpu, gpu

vr = VideoReader('xxx.mp4', ctx=cpu(0))
print('video frames:', len(reader))
batch = vr.next()
print('frame shape:', batch.shape)
print('numpy frames:', batch.asnumpy())

# skip 100 frames
vr.skip_frames(1000)
# seek to start
vr.seek(0)

# Another way is to directly access frames
for i in range(len(vr)):
    # the video reader will handle seeking and skipping in the most efficient manner
    frame = vr[i]

# To get multiple frames at once, use get_batch
frames = vf.get_batch([1, 3, 5, 7, 9])
print(frames.shape)
# (5, 240, 320, 3)

```

### VideoLoader

VideoLoader is designed for training deep learning models with tons of video files.
It provides smart video shuffle techniques in order to provide high random access performance (We know that seeking in video is super slow and redundant).
The optimizations are underlying in the C++ code, which are invisible to user.

```python
from decord import VideoLoader
from decord import cpu, gpu

vl = VideoLoader(['1.mp4', '2.avi', '3.mpeg'], ctx=[cpu(0)], shape=(2, 320, 240, 3), interval=1, skip=5, shuffle=1)
print('Total batches:', len(vl))

for batch in vl:
    print(batch.shape)
```

Shuffling video can be tricky, thus we provide various modes:

```python
shuffle = -1  # smart shuffle mode, based on video properties, (not implemented yet)
shuffle = 0  # all sequential, no seeking, following initial filename order
shuffle = 1  # random filename order, no random access for each video, very efficient
shuffle = 2  # random order
shuffle = 3  # random frame access in each video only
```

## Preliminary Benchmarks

| Setting                             | OpenCV VideoCapture | NVVL | Decord |
| ----------------------------------- | ------------------- | ---- | ------ |
| CPU sequential read                 | 1.0x                | -    | 1.1x   |
| CPU random access(no accurate seek) | 0.08x               | -    | 0.23x  |
| CPU random access (accurate seek)   | -                   |      | 0.06x  |
| GPU sequential                      | -                   | TODO | TODO   |
| GPU random access                   | -                   | TODO | TODO   |

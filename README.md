# Decode

![symbol](docs/symbol.png)

`Decord` is a reverse procedure of `Record`. It provides convenient video slicing methods based on a thin wrapper on top of hardware accelerated video decoders, e.g.

- FFMPEG/LibAV(On going)
- Nvidia Codecs(Planed)
- Intel Codecs

`Decord` was designed to handle awkward video shuffling experience in order to provide smooth experiences similar to random image loader for deep learning.

## Installation

### Install via pip
TODO

### Install from source

#### Linux

Install the system packages for building the shared library, for Debian/Ubuntu users, run:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-setuptools make cmake libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
```

Build the shared library in source root directory:

```bash
mkdir build
cd build
cmake ..
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

Tools like clang and GNU Make are packaged in *Command Line Tools* for macOS. To install:

```bash
xcode-select --install
```

To install other needed packages like cmake, we recommend first installing Homebrew, which is a popular package manager for macOS. Detailed instructions can be found on its [homepage](https://brew.sh/).

After installation of Homebrew, install cmake by:

```bash
brew install cmake
```

Then go to root directory build shared library:

```bash
mkdir build
cd build
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

TODO


## Usage

Decord provides minimal API set for bootstraping.

### VideoReader

VideoReader is used to access frames directly from video files.

```python
from decord import VideoReader

reader = VideoReader('xxx.mp4')
print('video frames:', len(reader))
print('frame shape:', vr.next().asnumpy().shape)

# skip 100 frames
vr.skip_frames(1000)
# seek to start
vr.seek(0)

```

### VideoLoader

VideoLoader is designed for training deep learning models with tons of video files. 
It provides smart video shuffle techniques in order to provide high random access performance (We know that seeking in video is super slow and redundant).
The optimizations are underlying in the C++ code, which are invisible to user.

```python
from decord import VideoLoader

vl = VideoLoader(['1.mp4', '2.avi', '3.mpeg'], shape=(2, 320, 240, 3), interval=1, skip=5, shuffle=1)
print('Total batches:', len(vl))

for batch in vl:
    print(batch.shape)
```

Shuffling video can be tricky, thus we provide various modes:

```python
shuffle = -1  # smart shuffle mode, based on video properties, not implemented yet
shuffle = 0  # all sequential, no seeking, following initial filename order
shuffle = 1  # random filename order, no random access for each video, very efficient
shuffle = 2  # random order
shuffle = 3  # random frame access in each video only
```

## Preliminary Benchmarks

| Setting             | OpenCV VideoCapture | NVVL | Decord |
|---------------------|---------------------|------|--------|
| CPU sequential read | 1.0x                | -    | 1.1x   |
| CPU random acess(no accurate seek)  | 0.08x                | -    | 0.23x  |
| CPU random acess (accurate seek)                    |    -                 |      |  0.06x  |
| GPU sequential                    |       -              |  TODO    |    TODO    |
| GPU random acess                   |      -               |  TODO    |   TODO     |
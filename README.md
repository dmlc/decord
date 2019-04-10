# Decode

![symbol](docs/symbol.png)

`Decord` is a reverse procedure of `Record`. It provides convenient video slicing methods based on a thin wrapper on top of hardware accelerated video decoders, e.g.

- FFMPEG
- LibAV
- Nvidia Codecs
- Intel Codecs


## Installation

### Install via pip
TODO

### Install from source

#### Linux

Install the system packages for building the shared library, for Debian/Ubuntu users, run:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev make cmake libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
```


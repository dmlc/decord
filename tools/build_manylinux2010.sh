#!/bin/bash

set -e

# build tools
yum install -y autoconf automake bzip2 bzip2-devel freetype-devel gcc gcc-c++ git libtool make mercurial pkgconfig zlib-devel

# cmake
cd ~
curl -O -L https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1-Linux-x86_64.sh
chmod +x ./cmake-3.19.1-Linux-x86_64.sh
./cmake-3.19.1-Linux-x86_64.sh --skip-license --prefix=/usr/local
/usr/local/bin/cmake -version

# workspace
mkdir ~/ffmpeg_sources

# nasm
cd ~/ffmpeg_sources
curl -O -L https://github.com/dmlc/decord/files/5685923/nasm-2.14.02.zip
unzip nasm-2.14.02.zip
cd nasm-2.14.02
./autogen.sh
./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin"
make
make install

# yasm
cd ~/ffmpeg_sources
curl -O -L https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
tar xzf yasm-1.3.0.tar.gz
cd yasm-1.3.0
./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin"
make
make install

# libx264
cd ~/ffmpeg_sources
git clone --depth 1 https://code.videolan.org/videolan/x264.git
cd x264
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-shared --enable-pic
make
make install

# libvpx
cd ~/ffmpeg_sources
git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git
cd libvpx
PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --disable-examples --disable-unit-tests --enable-vp9-highbitdepth --as=yasm --enable-shared --enable-pic
make
make install

# ffmpeg
cd ~/ffmpeg_sources
curl -O -L https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
tar xjf ffmpeg-snapshot.tar.bz2
cd ffmpeg
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs=-lpthread \
  --extra-libs=-lm \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-libvpx \
  --enable-libx264 \
  --enable-nonfree \
  --disable-static \
  --enable-shared \
  --enable-pic
make
make install

# test ffmpeg
ffmpeg -version

# decord
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pushd $DIR/..
mkdir build
cd build
/usr/local/bin/cmake .. -DUSE_CUDA=0
make -j$(nproc)
popd

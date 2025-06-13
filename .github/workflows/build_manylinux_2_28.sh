#!/bin/bash

# this file is actually for building decord for manylinux2010 on github action

set -e

# pwd
pwd

# build tools
yum install -y autoconf automake bzip2 bzip2-devel freetype-devel gcc gcc-c++ git libtool make mercurial pkgconfig zlib-devel

# cmake
pushd ~
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
make -j$(nproc)
make install

# yasm
cd ~/ffmpeg_sources
curl -O -L https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
tar xzf yasm-1.3.0.tar.gz
cd yasm-1.3.0
./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin"
make -j$(nproc)
make install

# libx264
cd ~/ffmpeg_sources
git clone --depth 1 https://code.videolan.org/videolan/x264.git
cd x264
export PATH="$HOME/bin:$PATH"
PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-shared --enable-pic
make -j$(nproc)
make install

# libvpx
cd ~/ffmpeg_sources
git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git
cd libvpx
export PATH="$HOME/bin:$PATH"
./configure --prefix="$HOME/ffmpeg_build" --disable-examples --disable-unit-tests --enable-vp9-highbitdepth --as=yasm --enable-shared --enable-pic
make -j$(nproc)
make install

# ffmpeg
cd ~/ffmpeg_sources
curl -O -L https://ffmpeg.org/releases/ffmpeg-4.1.6.tar.bz2
tar xjf ffmpeg-4.1.6.tar.bz2
cd ffmpeg-4.1.6
export PATH="$HOME/bin:$PATH"
PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
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
make -j$(nproc)
make install

# build libs
ls ~/ffmpeg_build/lib

# decord
popd
pwd
ls ..
mkdir -p ../build
pushd ../build
/usr/local/bin/cmake .. -DUSE_CUDA=0 -DFFMPEG_DIR=~/ffmpeg_build
make -j$(nproc)
cp libdecord.so /usr/local/lib/
popd

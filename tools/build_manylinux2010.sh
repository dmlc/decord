#!/bin/bash

set -e

# build tools
yum install -y autoconf automake gcc gcc-c++ git libtool make nasm pkgconfig wget opencv zlib-devel dbus-devel lua-devel zvbi libdvdread-devel  libdc1394-devel libxcb-devel xcb-util-devel libxml2-devel mesa-libGLU-devel pulseaudio-libs-devel alsa-lib-devel libgcrypt-devel qt-devel
yum --enablerepo=epel install -y yasm libva-devel libass-devel libkate-devel libbluray-devel libdvdnav-devel libcddb-devel libmodplug-devel
yum --enablerepo=rpmforge install -y a52dec-devel libmpeg2-devel

# cmake
cd ~
curl -O -L https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1-Linux-x86_64.sh
./cmake-3.19.1-Linux-x86_64.sh --skip-license --prefix=/usr/local/bin
cmake -version

# workspace
mkdir ~/ffmpeg_sources

# libx264
cd ~/ffmpeg_sources
git clone --depth 1 https://code.videolan.org/videolan/x264.git
cd x264
PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-shared --enable-pic
make
make install

# libvpx
cd ~/ffmpeg_sources
git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git
cd libvpx
./configure --prefix="$HOME/ffmpeg_build" --disable-examples --disable-unit-tests --enable-vp9-highbitdepth --as=yasm --enable-shared --enable-pic
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
cmake .. -DUSE_CUDA=0
make -j$(nproc)
popd

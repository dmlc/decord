#!/bin/bash

# build tools
yum install autoconf automake gcc gcc-c++ git libtool make nasm pkgconfig wget opencv zlib-devel dbus-devel lua-devel zvbi libdvdread-devel  libdc1394-devel libxcb-devel xcb-util-devel libxml2-devel mesa-libGLU-devel pulseaudio-libs-devel alsa-lib-devel libgcrypt-devel qt-devel
yum --enablerepo=epel install yasm libva-devel libass-devel libkate-devel libbluray-devel libdvdnav-devel libcddb-devel libmodplug-devel
yum --enablerepo=rpmforge install a52dec-devel libmpeg2-devel

# cmake
cd ~
curl -O -L https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1-Linux-x86_64.tar.gz
tar xvf cmake-3.19.1-Linux-x86_64.tar.gz
cp cmake-3.19.1-Linux-x86_64/bin/cmake /usr/bin/cmake
cmake -version

# workspace
mkdir -p /opt/source/ffmpeg

# libx264
cd /opt/source/ffmpeg/
git clone git://git.videolan.org/x264
cd x264
./configure --enable-shared --enable-pic
make
make install

# libvpx
cd /opt/source/ffmpeg/
git clone http://git.chromium.org/webm/libvpx.git
cd libvpx
./configure --enable-shared --enable-pic
make
make install

# ffmpeg
cd /opt/source/ffmpeg/
git clone git://source.ffmpeg.org/ffmpeg
cd ffmpeg
./configure --enable-gpl --enable-libvpx --enable-libx264 --enable-nonfree --disable-static --enable-shared --enable-pic
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

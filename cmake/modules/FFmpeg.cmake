# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(FFMPEG_FOUND)
  # always set the includedir when ffmpeg is available
  # avoid global retrigger of cmake
  message("FFMPEG_INCLUDE_DIR = ${FFMPEG_INCLUDE_DIR} ")
  message("FFMPEG_LIBRARIES = ${FFMPEG_LIBRARIES} ")
  include_directories(${FFMPEG_INCLUDE_DIR})
  file(GLOB DECORD_FFMPEG_SRCS src/video/ffmpeg/*.cc)
  list(APPEND DECORD_LINKER_LIBS ${FFMPEG_LIBRARIES})
else()
  message( FATAL_ERROR "Unable to find FFMPEG automatically, please specify FFMPEG location" )
endif(FFMPEG_FOUND)

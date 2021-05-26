cmake_minimum_required(VERSION 3.17)
project(SkeletonAnimationDemo)

set(CMAKE_CXX_STANDARD 14)

LINK_DIRECTORIES(vendor/opencv/lib)
add_executable(SkeletonAnimationDemo main.cpp)

find_library(COCOA_LIBRARY Cocoa)
target_include_directories(SkeletonAnimationDemo
        PUBLIC
        vendor/math/include
        vendor/opencv/include
        )
TARGET_LINK_LIBRARIES(SkeletonAnimationDemo
                      libopencv_core.a
                      libopencv_imgproc.a
                      libopencv_highgui.a
#                      libIlmImf.a
                      liblibjpeg.a
                      liblibpng.a
                      liblibtiff.a
                      libzlib.a
                      libopencv_video.a
                      libopencv_contrib.a
                      libopencv_calib3d.a
                      libopencv_features2d.a
                      libopencv_flann.a
                      ${COCOA_LIBRARY})


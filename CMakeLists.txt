cmake_minimum_required(VERSION 3.13)
project(Feature_Detection_and_Matching)

set(CMAKE_CXX_STANDARD 14)

include_directories(SIFT)

add_executable(Feature_Detection_and_Matching
        SIFT/ANMS.cpp
        SIFT/ANMS.h
        SIFT/Descriptor.h
        SIFT/GenerateDescriptor.cpp
        SIFT/GenerateDescriptor.h
        SIFT/HarrisCorner.cpp
        SIFT/HarrisCorner.h
        SIFT/Key_Point.h
        SIFT/Main.cpp
        SIFT/SSD.cpp
        SIFT/SSD.h)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Feature_Detection_and_Matching ${OpenCV_LIBS})

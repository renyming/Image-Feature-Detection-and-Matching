cmake_minimum_required(VERSION 3.13)
project(Feature_Detection_and_Matching)

set(CMAKE_CXX_STANDARD 14)

include_directories(SIFT)

add_executable(Feature_Detection_and_Matching
        src/ANMS.cpp
        src/ANMS.h
        src/Descriptor.h
        src/GenerateDescriptor.cpp
        src/GenerateDescriptor.h
        src/HarrisCorner.cpp
        src/HarrisCorner.h
        src/Key_Point.h
        src/Main.cpp
        src/SSD.cpp
        src/SSD.h)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Feature_Detection_and_Matching ${OpenCV_LIBS})

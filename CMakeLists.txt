cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0074 NEW)
project(fisheye_project)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE "Debug")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O0 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 3 REQUIRED)
#find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

list( APPEND CMAKE_MODULE_PATH /home/hl18/Downloads/g2o/cmake_modules )
#set(G2O_ROOT /usr/local/include/g2o)
#find_package(G2O REQUIRED)
find_package(g2o REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
)

add_library(BA BA_func.cpp)

add_executable(all_ue all_ue.cpp)
target_link_libraries(all_ue ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(BA_test_real BA_test_real.cpp)
target_link_libraries(BA_test_real ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(disparity_real disparity_real.cpp)
target_link_libraries(disparity_real ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

target_link_libraries( BA_test_real Sophus::Sophus BA)
target_link_libraries( all_ue Sophus::Sophus BA)

target_link_libraries( BA Sophus::Sophus g2o_core g2o_stuff ${OpenCV_LIBS})



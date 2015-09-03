cmake_minimum_required(VERSION 2.8)
project(APAP)

find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(DevIL REQUIRED)
find_package(Eigen3 REQUIRED)

set(VLFEAT_INCLUDE ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/include)
set(VLFEAT_LIB ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/lib/libvl.so)
set(SIFTGPU_INCLUDE ${PROJECT_SOURCE_DIR}/thirdparty/SiftGPU/include)
set(SIFTGPU_LIB ${PROJECT_SOURCE_DIR}/thirdparty/SiftGPU/lib/libsiftgpu.so)

add_executable(APAP ${PROJECT_SOURCE_DIR}/src/APAP.cpp
                    ${PROJECT_SOURCE_DIR}/src/Math.cpp
                    ${PROJECT_SOURCE_DIR}/src/CVUtility.cpp
                    ${PROJECT_SOURCE_DIR}/src/SiftGPUWrapper.cpp
                    ${PROJECT_SOURCE_DIR}/src/VLFeatSiftWrapper.cpp)

add_definitions(-std=c++11)

include_directories(
  ${GLEW_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIRS}
  ${IL_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${VLFEAT_INCLUDE}
  ${SIFTGPU_INCLUDE})

target_link_libraries(APAP 
  ${GLEW_LIBRARIES}
  ${GLUT_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${IL_LIBRARIES}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${SIFTGPU_LIB}
  ${VLFEAT_LIB})

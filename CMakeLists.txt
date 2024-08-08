cmake_minimum_required(VERSION 3.12)
project(yolov9-bytetrack-trt)

find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DAPI_EXPORTS)

# Find Eigen library
find_package(Eigen3 3.3 REQUIRED)


# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src/)
include_directories(bytetrack/)


# Add source files
set(SOURCES
    main.cpp
    src/yolov9.cpp
    bytetrack/BYTETracker.cpp
    bytetrack/KalmanFilter.cpp
    bytetrack/lapjv.cpp
    bytetrack/Object.cpp
    bytetrack/Rect.cpp
    bytetrack/STrack.cpp    
)

# Add CUDA source files
set(CUDA_SOURCES src/preprocess.cu)

# Add headers
set(HEADERS
    bytetrack/BYTETracker.h
    src/yolov9.h
    src/macros.h
    src/logging.h
    src/cuda_utils.h
    src/preprocess.h
)

file(GLOB BYTETRACK_HEADERS "bytetrack/*.h")

# Find and include OpenCV
set(OpenCV_DIR "/usr/include/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "/usr/src/TensorRT-8.6.1.6")

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

# Include MongoDB C++ driver
include_directories(/usr/local/include/mongocxx/v_noabi)
include_directories(/usr/local/include/bsoncxx/v_noabi)

# Create an executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${CUDA_SOURCES} ${BYTETRACK_HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}  
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
    mongocxx
    bsoncxx
)
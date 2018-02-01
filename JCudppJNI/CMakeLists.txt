cmake_minimum_required(VERSION 3.1)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../jcuda-common/JCudaCommon_CMake_flags.txt)

project(JCudpp)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../jcuda-common/JCudaCommon_CMake.txt)

if (NOT TARGET JCudaCommonJNI)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../jcuda-common/JCudaCommonJNI
        ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

set (BUILD_SHARED_LIBS ON)

find_path(CUDA_CUDPP_INCLUDE_DIR
  cudpp.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc/cudpp"
  DOC "Location of cudpp.h"
  NO_DEFAULT_PATH
)

find_path(CUDA_CUDPP_INCLUDE_DIR cudpp.h DOC "Location of cudpp.h and cudpp_hash.h")

mark_as_advanced(CUDA_CUDPP_INCLUDE_DIR)

if(CMAKE_HOST_WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(cuda_cudpp_name cudpp64)
    set(cuda_cudpp_hash_name cudpp_hash64)
  else()
    set(cuda_cudpp_name cudpp32)
    set(cuda_cudpp_hash_name cudpp_hash32)
  endif()

  find_library(CUDA_CUDPP_LIBRARY
    NAMES cutil ${cuda_cudpp_name}
    PATHS ${CUDA_SDK_SEARCH_PATH}
    PATH_SUFFIXES "common/lib" "lib"
    DOC "Location of cudpp library"
    NO_DEFAULT_PATH
  )

  find_library(CUDA_CUDPP_HASH_LIBRARY
    NAMES cutil ${cuda_cudpp_name}
    PATHS ${CUDA_SDK_SEARCH_PATH}
    PATH_SUFFIXES "common/lib" "lib"
    DOC "Location of cudpp hash library"
    NO_DEFAULT_PATH
  )
elseif(CMAKE_HOST_UNIX)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(cuda_cudpp_name cudpp_x86_64)
    set(cuda_cudpp_hash_name cudpp_hash_x86_64)
  else()
    set(cuda_cudpp_name cudpp_i386)
    set(cuda_cudpp_hash_name cudpp_hash_i386)
  endif()

  find_library(CUDA_CUDPP_LIBRARY
    NAMES cutil ${cuda_cudpp_name}
    PATHS ${CUDA_SDK_SEARCH_PATH}
    PATH_SUFFIXES "common/lib/darwin" "common/lib/linux" "common/lib" "lib"
    DOC "Location of cudpp library"
    NO_DEFAULT_PATH
  )

  find_library(CUDA_CUDPP_HASH_LIBRARY
    NAMES cutil ${cuda_cudpp_hash_name}
    PATHS ${CUDA_SDK_SEARCH_PATH}
    PATH_SUFFIXES "common/lib/darwin" "common/lib/linux" "common/lib" "lib"
    DOC "Location of cudpp hash library"
    NO_DEFAULT_PATH
  )
endif()

find_library(CUDA_CUDPP_LIBRARY NAMES cudpp ${cuda_cudpp_name} DOC "Location of cudpp library")
mark_as_advanced(CUDA_CUDPP_LIBRARY)
set(CUDA_CUDPP_LIBRARIES ${CUDA_CUDPP_LIBRARY})

find_library(CUDA_CUDPP_HASH_LIBRARY NAMES cudpp ${cuda_cudpp_name} DOC "Location of cudpp hash library")
mark_as_advanced(CUDA_CUDPP_HASH_LIBRARY)
set(CUDA_CUDPP_HASH_LIBRARIES ${CUDA_CUDPP_HASH_LIBRARY})

include_directories (
    src/
    ${CUDA_CUDPP_INCLUDE_DIR}
    ${JCudaCommonJNI_INCLUDE_DIRS}
    ${JNI_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(${PROJECT_NAME}
    src/JCudpp.cpp
)

cuda_add_cusparse_to_target(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    JCudaCommonJNI
    ${CUDA_CUDPP_LIBRARIES}
    ${CUDA_CUDPP_HASH_LIBRARIES}
)


set_target_properties(${PROJECT_NAME}
    PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-${JCUDA_VERSION}-${JCUDA_OS}-${JCUDA_ARCH})

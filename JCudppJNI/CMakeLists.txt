cmake_minimum_required(VERSION 2.8)

# Overriding the CMake flags to use static runtime libraries
# See http://www.cmake.org/Wiki/CMake_FAQ#
#     How_can_I_build_my_MSVC_application_with_a_static_runtime.3F
set(CMAKE_USER_MAKE_RULES_OVERRIDE
   ${CMAKE_CURRENT_SOURCE_DIR}/CMake_c_flag_overrides.cmake)
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
   ${CMAKE_CURRENT_SOURCE_DIR}/CMake_cxx_flag_overrides.cmake)
   
project(JCuda-JCudpp)

set(CMAKE_MODULE_PATH 
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH})

find_package(CUDA 7.0 REQUIRED)
find_package(JNI REQUIRED)

if(CMAKE_HOST_WIN32)
  set(CMAKE_HOST "windows")
elseif(CMAKE_HOST_APPLE)
  set(CMAKE_HOST "apple")
  set(CMAKE_SKIP_RPATH FALSE)
elseif(CMAKE_HOST_UNIX)
  set(CMAKE_HOST "linux")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(CMAKE_ARCH "x86_64")
else()
  set(CMAKE_ARCH "x86")
endif()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(CommonJNI)
add_subdirectory(JCudppJNI)


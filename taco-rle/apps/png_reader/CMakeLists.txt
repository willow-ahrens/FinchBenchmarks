cmake_minimum_required(VERSION 2.8.12)
project(png_reader)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
file(GLOB SOURCE_CODE ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.c)
add_executable(${PROJECT_NAME} ${SOURCE_CODE})

#target_link_libraries(${PROJECT_NAME} LINK_PUBLIC lodepng)

# To let the app be a standalone project 
if (NOT TACO_INCLUDE_DIR)
  if (NOT DEFINED ENV{TACO_INCLUDE_DIR} OR NOT DEFINED ENV{TACO_LIBRARY_DIR})
    message(FATAL_ERROR "Set the environment variables TACO_INCLUDE_DIR and TACO_LIBRARY_DIR")
  endif ()
  set(TACO_INCLUDE_DIR $ENV{TACO_INCLUDE_DIR})
  set(TACO_LIBRARY_DIR $ENV{TACO_LIBRARY_DIR})
  find_library(taco taco ${TACO_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${taco})
else()
  set_target_properties("${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "taco-${PROJECT_NAME}")
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC taco)
endif ()

find_package(benchmark REQUIRED)
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC benchmark::benchmark)

# Include taco headers
include_directories(${TACO_INCLUDE_DIR})

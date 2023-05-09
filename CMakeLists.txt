cmake_minimum_required( VERSION 3.0 )
project(Project3)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable( Project3 ImgDB.cpp functions.h functions.cpp csv_util.cpp csv_util.h)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )
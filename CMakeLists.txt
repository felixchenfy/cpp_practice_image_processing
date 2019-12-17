cmake_minimum_required(VERSION 2.8)
project( my_project )

# -- Compile settings
set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Debug" )
# set( CMAKE_BUILD_TYPE "Release" ) # *** stack smashing detected ***: <unknown> terminated
# set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )


# -- Folders
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
include_directories( ${PROJECT_SOURCE_DIR}/include )

# -- External libraries
find_package( OpenCV REQUIRED )
set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
)

# -- My library
add_subdirectory( src/my_cv )

# -- Merge libraries
set( ALL_LIBS 
    ${OpenCV_LIBS}
    filters
    geometry
    maths
)

# -- Tests
add_subdirectory( tests )

# -- Executables
# add_executable( main src/canny.cpp )
# target_link_libraries( main ${OpenCV_LIBS} )




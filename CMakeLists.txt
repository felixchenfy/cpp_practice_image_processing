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
# include_directories(
#         ${OpenCV_INCLUDE_DIRS}
# )

# -- My library
add_subdirectory( src/my_cv )

set( ALL_LIBS 
    ${OpenCV_LIBS}
    filters
    geometry
    keypoints
)

# ================================= Main scripts ================================= #

# ---------------------------- Image processing ---------------------------- #

add_executable( test_sobel_and_gaussion tests/test_sobel_and_gaussion.cpp )
target_link_libraries( test_sobel_and_gaussion ${ALL_LIBS})

add_executable( test_canny tests/test_canny.cpp )
target_link_libraries( test_canny ${ALL_LIBS})

add_executable( test_hough_line tests/test_hough_line.cpp )
target_link_libraries( test_hough_line ${ALL_LIBS})

add_executable( test_harris tests/test_harris.cpp )
target_link_libraries( test_harris ${ALL_LIBS})

# ---------------------------- RANSAC ---------------------------- #
# If you don't have eigen library, 
# you may install it, or comment out all following lines.

include_directories(
    "/usr/include/eigen3/" # I use eigen in RASNAC algorithm. You may change this to your directory.
)

add_executable( test_fit_2d_line tests/test_fit_2d_line.cpp )
target_link_libraries( test_fit_2d_line ${ALL_LIBS})

add_executable( test_svd tests/others/test_svd.cpp )
target_link_libraries( test_svd ${ALL_LIBS})







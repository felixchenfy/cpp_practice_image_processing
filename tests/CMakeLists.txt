
add_executable( test_sobel_and_gaussion test_sobel_and_gaussion.cpp )
target_link_libraries( test_sobel_and_gaussion ${ALL_LIBS})

add_executable( test_canny test_canny.cpp )
target_link_libraries( test_canny ${ALL_LIBS})

add_executable( test_hough_line test_hough_line.cpp )
target_link_libraries( test_hough_line ${ALL_LIBS})


add_executable( test_harris test_harris.cpp )
target_link_libraries( test_harris ${ALL_LIBS})

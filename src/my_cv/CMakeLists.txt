

add_library( geometry geometry.cpp )
target_link_libraries( geometry ${THIRD_PARTY_LIBS})

add_library( filters filters.cpp )
target_link_libraries( filters ${THIRD_PARTY_LIBS})

add_library( keypoints keypoints.cpp )
target_link_libraries( keypoints ${THIRD_PARTY_LIBS} filters geometry)

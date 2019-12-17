
add_library( filters filters.cpp )
target_link_libraries( filters ${THIRD_PARTY_LIBS})

add_library( geometry geometry.cpp )
target_link_libraries( geometry ${THIRD_PARTY_LIBS})

add_library( maths maths.cpp )
target_link_libraries( maths ${THIRD_PARTY_LIBS})
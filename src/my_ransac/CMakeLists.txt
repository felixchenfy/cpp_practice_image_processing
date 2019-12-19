

add_library(models 
    model_2d_line.cpp
)
target_link_libraries( models ${THIRD_PARTY_LIBS})

add_library(ransac ransac.cpp )
target_link_libraries( ransac models ${THIRD_PARTY_LIBS})

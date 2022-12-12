cmake_minimum_required(VERSION 3.9)

project(EIGEN_PARDISO LANGUAGES CXX C)

# Create variables
###############################################################################################
set(EIGEN_PARDISO_SOURCES ${PROJECT_NAME}_sources)
set(EIGEN_PARDISO_HEADERS ${PROJECT_NAME}_headers)
set(EIGEN_PARDISO_INCLUDE ${PROJECT_NAME}_include)
set(EIGEN_PARDISO_LINKED_LIBRARIES ${PROJECT_NAME}_linked_libraries)
set(EIGEN_PARDISO_COMPILE_DEFINITIONS ${PROJECT_NAME}_compile_definitions)



if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

set(ENABLE_MKL OFF CACHE BOOL "Enable MKL extension")

# Add dependencies
###############################################################################################

## Eigen3
find_package(Eigen3 CONFIG REQUIRED)
list(APPEND ${EIGEN_PARDISO_LINKED_LIBRARIES} Eigen3::Eigen)

## MKL
if(${ENABLE_MKL})
    set(MKL_INTERFACE lp64)
    find_package(MKL CONFIG REQUIRED)
    list(APPEND ${EIGEN_PARDISO_LINKED_LIBRARIES} MKL::MKL)
    # ENABLE THE NEXT TO LET EIGEN PERFORM ALL OPERATIONS WITH MKL
    #list(APPEND ${EIGEN_PARDISO_COMPILE_DEFINITIONS} EIGEN_USE_MKL_ALL)
endif()

# Create public headers
###############################################################################################
set(EIGEN_PARDISO_ADDITIONAL_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR}/${EIGEN_PARDISO_ADDITIONAL_INCLUDE})

list(APPEND ${EIGEN_PARDISO_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/Macro.hpp.in)

foreach(header ${${EIGEN_PARDISO_HEADERS}})
    get_filename_component(headerDirectory ${header} DIRECTORY)
    get_filename_component(headerFileName ${header} NAME_WE)
    get_filename_component(headerExtension ${header} EXT)

    if(${headerExtension} MATCHES ".hpp.in")
        set(headerExtension ".hpp")
        set(headerDirectory ${EIGEN_PARDISO_ADDITIONAL_INCLUDE_PATH})

        configure_file(${header} ${headerDirectory}/${headerFileName}${headerExtension} @ONLY)
    elseif(NOT ${headerExtension} MATCHES ".hpp" AND NOT ${headerExtension} MATCHES ".h")
        message(SEND_ERROR "Header extension not recognized for file ${headerFileName}${headerExtension}")
    endif()

    list(APPEND ${EIGEN_PARDISO_PUBLIC_HEADERS} ${headerDirectory}/${headerFileName}${headerExtension})
endforeach()

list(APPEND ${EIGEN_PARDISO_INCLUDE} ${EIGEN_PARDISO_ADDITIONAL_INCLUDE_PATH})

# Create library test executable
###############################################################################################
add_executable(${PROJECT_NAME} main.cpp ${${EIGEN_PARDISO_SOURCES}} ${${EIGEN_PARDISO_HEADERS}})

target_link_libraries(${PROJECT_NAME} ${${EIGEN_PARDISO_LINKED_LIBRARIES}})
target_include_directories(${PROJECT_NAME} PRIVATE ${${EIGEN_PARDISO_INCLUDE}})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${${EIGEN_PARDISO_COMPILE_DEFINITIONS}})

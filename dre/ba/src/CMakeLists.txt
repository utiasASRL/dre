# Collect all source files
file(GLOB_RECURSE BA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Create the library
add_library(ba ${BA_SOURCES})

# Include directories
target_include_directories(ba
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)

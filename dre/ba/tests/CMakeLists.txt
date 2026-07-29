find_package(GTest REQUIRED)

add_executable(test_local_map_scan test_local_map_scan.cpp)
add_executable(test_voxel_map test_voxel_map.cpp)

target_link_libraries(test_local_map_scan
    PRIVATE
        ba
        GTest::GTest
        GTest::Main
)
target_link_libraries(test_voxel_map
    PRIVATE
        ba
        GTest::GTest
        GTest::Main
)

# Register with CTest
add_test(NAME LocalMapScanTests COMMAND test_local_map_scan)
add_test(NAME VoxelMapTests COMMAND test_voxel_map)
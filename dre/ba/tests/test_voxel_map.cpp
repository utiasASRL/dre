#include <gtest/gtest.h>
#include <ba/map/voxel_map.hpp>
#include <iostream>
#include <lgmath/se2/Transformation.hpp>

TEST(VoxelMapTests, ValidateObjectCreation) {
    double resolution = 0.5;
    ba::VoxelMap voxel_map(resolution);
    EXPECT_DOUBLE_EQ(voxel_map.res(), resolution);
    EXPECT_EQ(voxel_map.size(), 0);
}

TEST(VoxelMapTests, ValidateAddAndAccessVoxels) {
    ba::VoxelMap voxel_map(1.0);
    voxel_map.add_single_voxel(0, 0, 1.5);
    voxel_map.add_single_voxel(1.0, 1.0, 2.5); // Should map to (1,1)

    EXPECT_EQ(voxel_map.size(), 2);
    EXPECT_DOUBLE_EQ(voxel_map.at({0, 0}), 1.5);
    EXPECT_DOUBLE_EQ(voxel_map.at({1, 1}), 2.5);
}

TEST(VoxelMapTests, ValidateIndexConversion) {
    ba::VoxelMap voxel_map(0.5);
    auto idx1 = voxel_map.index(1.0, 1.0);
    EXPECT_EQ(idx1.first, 2);
    EXPECT_EQ(idx1.second, 2);

    auto idx2 = voxel_map.index(-0.1, -0.1);
    EXPECT_EQ(idx2.first, -1);
    EXPECT_EQ(idx2.second, -1);
}

TEST(VoxelMapTests, ValidateInitMap) {
    ba::VoxelMap voxel_map(1.0);
    lgmath::se2::Transformation pose(Eigen::Vector3d(0.0, 0.0, 0.0));
    double max_dist = 2.0;
    voxel_map.init_map(pose, max_dist, 0);

    // Expect voxels to be initialized in a square window around (0,0) within max_dist
    int expected_voxel_count = 0;
    for (int32_t a = -2; a <= 2; ++a) {
        for (int32_t b = -2; b <= 2; ++b) {
            if (std::sqrt(a * a + b * b) <= max_dist) {
                expected_voxel_count++;
                EXPECT_TRUE(voxel_map.contains({a, b}));
                EXPECT_DOUBLE_EQ(voxel_map.at({a, b}), 0.0);
            }
        }
    }
    EXPECT_EQ(voxel_map.size(), expected_voxel_count);
}

TEST(VoxelMapTests, ValidateZeroOutAndRandomize) {
    ba::VoxelMap voxel_map(1.0);
    voxel_map.add_single_voxel(0, 0, 1.0);
    voxel_map.add_single_voxel(1, 1, 2.0);

    voxel_map.zero_out();
    EXPECT_DOUBLE_EQ(voxel_map.at({0, 0}), 0.0);
    EXPECT_DOUBLE_EQ(voxel_map.at({1, 1}), 0.0);

    voxel_map.randomize(42); // Fixed seed for deterministic test
    EXPECT_GE(voxel_map.at({0, 0}), 0.0);
    EXPECT_LE(voxel_map.at({0, 0}), 1.0);
    EXPECT_GE(voxel_map.at({1, 1}), 0.0);
    EXPECT_LE(voxel_map.at({1, 1}), 1.0);
}

TEST(VoxelMapTests, ValidateContains) {
    ba::VoxelMap voxel_map(1.0);
    voxel_map.add_single_voxel(0, 0, 1.0);

    EXPECT_TRUE(voxel_map.contains({0, 0}));
    EXPECT_FALSE(voxel_map.contains({1, 1}));
}

TEST(VoxelMapTests, ValidateAtThrows) {
    ba::VoxelMap voxel_map(1.0);
    voxel_map.add_single_voxel(0, 0, 1.0);

    EXPECT_NO_THROW(voxel_map.at({0, 0}));
    EXPECT_THROW(voxel_map.at({1, 1}), std::out_of_range);
}

TEST(VoxelMapTests, ValidateAddSingleVoxelDoesNotOverwrite) {
    ba::VoxelMap voxel_map(1.0);
    voxel_map.add_single_voxel(0, 0, 1.0);
    voxel_map.add_single_voxel(0, 0, 2.0); // Should not overwrite

    EXPECT_DOUBLE_EQ(voxel_map.at({0, 0}), 1.0);
}

TEST(VoxelMapTests, ValidateVoxelSorting) {
    ba::VoxelMap voxel_map(1.0);
    voxel_map.add_single_voxel(2, 2, 1.0);
    voxel_map.add_single_voxel(0, 0, 1.0);
    voxel_map.add_single_voxel(1, 1, 1.0);

    auto sorted_keys = voxel_map.get_sorted_keys_downsampled(1.0);
    ASSERT_EQ(sorted_keys.size(), 3);
    EXPECT_EQ(sorted_keys[0], (ba::VoxelMap::Index{0, 0}));
    EXPECT_EQ(sorted_keys[1], (ba::VoxelMap::Index{1, 1}));
    EXPECT_EQ(sorted_keys[2], (ba::VoxelMap::Index{2, 2}));
}

TEST(VoxelMapTests, ValidateVoxelSortingWithDownsampling) {
    ba::VoxelMap voxel_map(1.0);
    voxel_map.add_single_voxel(2, 2, 1.0);
    voxel_map.add_single_voxel(0, 0, 1.0);
    voxel_map.add_single_voxel(1, 1, 1.0);
    voxel_map.add_single_voxel(3, 3, 1.0);

    auto downsampled_keys = voxel_map.get_sorted_keys_downsampled(1.0);
    ASSERT_EQ(downsampled_keys.size(), 4);
    downsampled_keys = voxel_map.get_sorted_keys_downsampled(0.5);
    ASSERT_EQ(downsampled_keys.size(), 2);
    downsampled_keys = voxel_map.get_sorted_keys_downsampled(0.25);
    ASSERT_EQ(downsampled_keys.size(), 1);
}
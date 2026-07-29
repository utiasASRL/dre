#include <gtest/gtest.h>
#include <ba/scans/local_map_scan.hpp>
#include <lgmath/se3/Transformation.hpp>
#include <iostream>
#include <ba/utils/ba_config.hpp>

// Create constant Options for tests
ba::OptimizationOptions create_test_options() {
    ba::OptimizationOptions opts;
    opts.cumul_thresh = 0.8;
    opts.range_factor = 0.1;
    opts.meas_std = 0.5;
    return opts;
}

TEST(LocalMapScanTests, ValidateCoordToPixel) {
    // Create a simple local map (3x3)
    // Here the pixel coordinates line up with the radar coordinates
    Eigen::MatrixXf local_map(3, 3);

    // Define scan parameters
    int64_t timestamp = 1;
    int scan_id = 1;
    lgmath::se3::Transformation pose;
    ba::OptimizationOptions opts = create_test_options();

    // Create LocalMapScan instance
    ba::LocalMapScan scan(timestamp, scan_id, 1.0, opts, pose, pose, local_map);

    // Test point at in world coordinates
    // This point should be top left corner of the image
    double x_world = 1.0;
    double y_world = -1.0;
    auto px = scan.coord_to_pixel(x_world, y_world);

    // Validate results
    EXPECT_EQ(px.first, 0.0);
    EXPECT_EQ(px.second, 0.0);

    // Now let's try an imperfect alignment (even number of pixels)
    local_map = Eigen::MatrixXf(4, 4);

    // The same world coordinates should now map to pixel (0.5, 0.5)
    ba::LocalMapScan scan2(timestamp, scan_id, 1.0, opts, pose, pose, local_map);
    px = scan2.coord_to_pixel(x_world, y_world);
    EXPECT_EQ(px.first, 0.5);
    EXPECT_EQ(px.second, 0.5);
}

TEST(LocalMapScanTests, ValidateRootPixelCoords) {
    // Create a simple local map (3x3)
    Eigen::MatrixXf local_map(3, 3);

    // Define scan parameters
    int64_t timestamp = 1;
    int scan_id = 1;
    lgmath::se3::Transformation pose;
    ba::OptimizationOptions opts = create_test_options();

    // Create LocalMapScan instance
    ba::LocalMapScan scan(timestamp, scan_id, 1.0, opts, pose, pose, local_map);

    // First try the top left corner, this should correspond to pixel (0,0)
    // so the root pixel coords should also be (0,0)
    double x_world = 1.0;
    double y_world = -1.0;
    auto root_px = scan.get_root_pixel_coords(x_world, y_world);
    EXPECT_EQ(root_px.first, 0.0);
    EXPECT_EQ(root_px.second, 0.0);

    // Now let's have a point that lies between pixel (0,0) and (1,1)
    // This should have the same (0,0) root pixel coords
    x_world = 0.5;
    y_world = -0.5;
    root_px = scan.get_root_pixel_coords(x_world, y_world);
    EXPECT_EQ(root_px.first, 0.0);
    EXPECT_EQ(root_px.second, 0.0);

    // Now let's try a point between pixel (1,1) and (2,2)
    // This should have root pixel coords (1,1)
    x_world = -0.5;
    y_world = 0.5;
    root_px = scan.get_root_pixel_coords(x_world, y_world);
    EXPECT_EQ(root_px.first, 1.0);
    EXPECT_EQ(root_px.second, 1.0);
}

TEST(LocalMapScanTests, ValidateCoverageCheck) {
    // Create a simple local map (11x11) with arbitrary values
    // This spans from -5 to +5 in both x and y in world coordinates
    Eigen::MatrixXf local_map(11, 11);
    local_map.setRandom();

    // Define scan parameters
    int64_t timestamp = 1;
    int scan_id = 1;
    lgmath::se3::Transformation pose;
    ba::OptimizationOptions opts = create_test_options();

    // Create LocalMapScan instance
    ba::LocalMapScan scan(timestamp, scan_id, 1.0, opts, pose, pose, local_map);

    // Test point well within bounds
    double x_in = 0.0;
    double y_in = 0.0;
    EXPECT_TRUE(scan.check_coverage_at_point(x_in, y_in));

    // Test point near edge, but within
    double x_edge = 4.9;
    double y_edge = 4.9;
    EXPECT_TRUE(scan.check_coverage_at_point(x_edge, y_edge));

    // Test point right on the top left edge, we can still interpolate this
    // so it should be valid
    double x_on_edge = 5.0;
    double y_on_edge = -5.0;
    EXPECT_TRUE(scan.check_coverage_at_point(x_on_edge, y_on_edge));

    // Test point right on the top right edge, we can't interpolate this one
    // since we always look to the right/down from the root coords
    double x_on_edge2 = 5.0;
    double y_on_edge2 = 5.0;
    EXPECT_FALSE(scan.check_coverage_at_point(x_on_edge2, y_on_edge2));

    // Test point outside bounds
    double x_out = 5.0;
    double y_out = 5.1;
    EXPECT_FALSE(scan.check_coverage_at_point(x_out, y_out));
}

TEST(LocalMapScanTests, ValidateInterpolation) {
    // Define scan parameters
    int64_t timestamp = 1;
    int scan_id = 1;
    lgmath::se3::Transformation pose;
    ba::OptimizationOptions opts = create_test_options();

    // Create a simple local map (3x3) with identical intensity values
    Eigen::MatrixXf local_map(3, 3);
    local_map << 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0;
    ba::LocalMapScan scan(timestamp, scan_id, 1.0, opts, pose, pose, local_map);
    // Test interpolation within the local map, it should produce 1.0 everywhere
    double x_query = 0.5;
    double y_query = 0.5;
    std::optional<ba::Scan::Measurement> meas = scan.interpolate(x_query, y_query);
    ASSERT_TRUE(meas.has_value());
    EXPECT_FLOAT_EQ(meas->intensity, 1.0);

    // Test another point within map
    x_query = -0.5;
    y_query = -0.5;
    meas = scan.interpolate(x_query, y_query);
    ASSERT_TRUE(meas.has_value());
    EXPECT_FLOAT_EQ(meas->intensity, 1.0);

    // Make the map non-uniform
    local_map << 0.0, 1.0, 0.0,
                1.0, 0.0, 1.0,
                0.0, 1.0, 0.0;
    ba::LocalMapScan scan2(timestamp, scan_id, 1.0, opts, pose, pose, local_map);

    // Test point at center (0,0), should be exactly 0.0
    x_query = 0.0;
    y_query = 0.0;
    meas = scan2.interpolate(x_query, y_query);
    ASSERT_TRUE(meas.has_value());
    EXPECT_FLOAT_EQ(meas->intensity, 0.0);

    // Test point at (0.5, 0.5), should be 0.5
    x_query = 0.5;
    y_query = 0.5;
    meas = scan2.interpolate(x_query, y_query);
    ASSERT_TRUE(meas.has_value());
    EXPECT_FLOAT_EQ(meas->intensity, 0.5);

    // Test point at (0.5, 0.0), should also be 0.5
    x_query = 0.5;
    y_query = 0.0;
    meas = scan2.interpolate(x_query, y_query);
    ASSERT_TRUE(meas.has_value());
    EXPECT_FLOAT_EQ(meas->intensity, 0.5);

    // Test point at (0.0, 0.25), should be 0.25
    x_query = 0.0;
    y_query = 0.25;
    meas = scan2.interpolate(x_query, y_query);
    ASSERT_TRUE(meas.has_value());
    EXPECT_FLOAT_EQ(meas->intensity, 0.25);

    // Test point outside bounds, should return nullopt
    x_query = 2.0;
    y_query = 2.0;
    meas = scan2.interpolate(x_query, y_query);
    EXPECT_FALSE(meas.has_value());
}
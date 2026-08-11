#ifndef COMPRESSION_STREAM_HPP
#define COMPRESSION_STREAM_HPP

#include <cstdint>

constexpr int IMAGE_WIDTH = 512;
constexpr int IMAGE_HEIGHT = 512;
constexpr int IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr int BLOCK_SIZE = 8;
constexpr int BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE;
constexpr int BLOCKS_X = IMAGE_WIDTH / BLOCK_SIZE;
constexpr int BLOCKS_Y = IMAGE_HEIGHT / BLOCK_SIZE;
constexpr int NUM_BLOCKS = BLOCKS_X * BLOCKS_Y;
constexpr int MAX_RLE_PAIRS_PER_BLOCK = BLOCK_ELEMENTS - 1;
constexpr int TOTAL_RLE_PAIR_SLOTS = NUM_BLOCKS * MAX_RLE_PAIRS_PER_BLOCK;
constexpr std::uint32_t GSC1_HEADER_BYTES = 152;

constexpr std::uint16_t QUANTISATION_MATRIX[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

extern "C" void compression_stream_kernel(
    const std::uint8_t* input_pixels,
    std::int16_t* dc_values,
    std::uint8_t* pair_counts,
    std::uint8_t* zero_runs,
    std::int16_t* pair_values
);

#endif
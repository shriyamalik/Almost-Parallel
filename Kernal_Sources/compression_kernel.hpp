#ifndef COMPRESSION_KERNEL_HPP
#define COMPRESSION_KERNEL_HPP

#include <cstdint>

constexpr int IMAGE_WIDTH = 512;
constexpr int IMAGE_HEIGHT = 512;
constexpr int IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

constexpr int BLOCK_SIZE = 8;
constexpr int BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE;
constexpr int BLOCKS_X = IMAGE_WIDTH / BLOCK_SIZE;
constexpr int BLOCKS_Y = IMAGE_HEIGHT / BLOCK_SIZE;
constexpr int NUM_BLOCKS = BLOCKS_X * BLOCKS_Y;

// The theoretical maximum for 63 AC coefficients is below 182 bytes.
// A 192-byte slot leaves a small safety margin and gives simple addressing.
constexpr int MAX_AC_BYTES_PER_BLOCK = 192;
constexpr int TOTAL_AC_STORAGE_BYTES =
    NUM_BLOCKS * MAX_AC_BYTES_PER_BLOCK;

constexpr std::uint32_t GSC2_HEADER_BYTES = 152;

extern "C" void compression_kernel(
    const std::uint8_t* input_pixels,
    std::int16_t* dc_values,
    std::uint16_t* ac_bit_counts,
    std::uint8_t* ac_data,
    std::uint32_t* compact_size_bytes,
    std::uint32_t* status
);

#endif

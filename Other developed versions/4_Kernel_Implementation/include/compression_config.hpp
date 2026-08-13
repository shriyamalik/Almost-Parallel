#ifndef COMPRESSION_CONFIG_HPP
#define COMPRESSION_CONFIG_HPP

#include <cstdint>

constexpr int IMAGE_WIDTH = 512;
constexpr int IMAGE_HEIGHT = 512;
constexpr int IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

constexpr int BLOCK_SIZE = 8;
constexpr int BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE;
constexpr int BLOCKS_X = IMAGE_WIDTH / BLOCK_SIZE;
constexpr int BLOCKS_Y = IMAGE_HEIGHT / BLOCK_SIZE;
constexpr int NUM_BLOCKS = BLOCKS_X * BLOCKS_Y;

// Every intermediate buffer is block-major: all 64 values for block 0,
// followed by all 64 values for block 1, and so on.
constexpr int TOTAL_COEFFICIENTS = NUM_BLOCKS * BLOCK_ELEMENTS;

// The theoretical maximum for 63 AC coefficients is below 182 bytes.
// A 192-byte slot leaves a small safety margin and simplifies addressing.
constexpr int MAX_AC_BYTES_PER_BLOCK = 192;
constexpr int TOTAL_AC_STORAGE_BYTES =
    NUM_BLOCKS * MAX_AC_BYTES_PER_BLOCK;

constexpr std::uint32_t GSC2_HEADER_BYTES = 152;

using shifted_storage_t = std::int16_t;
using dct_storage_t = std::int64_t;
using quantised_storage_t = std::int16_t;

extern "C" void load_shift_kernel(
    const std::uint8_t* input_pixels,
    shifted_storage_t* shifted_pixels
);

extern "C" void dct_kernel(
    const shifted_storage_t* shifted_pixels,
    dct_storage_t* dct_values
);

extern "C" void quantise_kernel(
    const dct_storage_t* dct_values,
    quantised_storage_t* quantised_values
);

extern "C" void encode_kernel(
    const quantised_storage_t* quantised_values,
    std::int16_t* dc_values,
    std::uint16_t* ac_bit_counts,
    std::uint8_t* ac_data,
    std::uint32_t* compact_size_bytes,
    std::uint32_t* status
);

#endif

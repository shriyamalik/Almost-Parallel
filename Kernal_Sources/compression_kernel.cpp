#include "compression_kernel.hpp"

#include <ap_fixed.h>
#include <ap_int.h>
#include <cstdint>

namespace {

using shifted_pixel_t = ap_int<9>;
using dct_coefficient_t =
    ap_fixed<18, 2, AP_RND_CONV, AP_SAT>;
using first_stage_t =
    ap_fixed<28, 12, AP_RND_CONV, AP_SAT>;
using dct_value_t =
    ap_fixed<36, 14, AP_RND_CONV, AP_SAT>;
using quantisation_ratio_t =
    ap_fixed<40, 16, AP_TRN, AP_SAT>;
using quantised_t = ap_int<16>;

constexpr int QUANTISATION_FRACTION_BITS =
    40 - 16;

static const dct_coefficient_t DCT_MATRIX[8][8] = {
    {
        0.353553390593,  0.353553390593,
        0.353553390593,  0.353553390593,
        0.353553390593,  0.353553390593,
        0.353553390593,  0.353553390593
    },
    {
        0.490392640202,  0.415734806151,
        0.277785116510,  0.097545161008,
       -0.097545161008, -0.277785116510,
       -0.415734806151, -0.490392640202
    },
    {
        0.461939766256,  0.191341716183,
       -0.191341716183, -0.461939766256,
       -0.461939766256, -0.191341716183,
        0.191341716183,  0.461939766256
    },
    {
        0.415734806151, -0.097545161008,
       -0.490392640202, -0.277785116510,
        0.277785116510,  0.490392640202,
        0.097545161008, -0.415734806151
    },
    {
        0.353553390593, -0.353553390593,
       -0.353553390593,  0.353553390593,
        0.353553390593, -0.353553390593,
       -0.353553390593,  0.353553390593
    },
    {
        0.277785116510, -0.490392640202,
        0.097545161008,  0.415734806151,
       -0.415734806151, -0.097545161008,
        0.490392640202, -0.277785116510
    },
    {
        0.191341716183, -0.461939766256,
        0.461939766256, -0.191341716183,
       -0.191341716183,  0.461939766256,
       -0.461939766256,  0.191341716183
    },
    {
        0.097545161008, -0.277785116510,
        0.415734806151, -0.490392640202,
        0.490392640202, -0.415734806151,
        0.277785116510, -0.097545161008
    }
};

static const ap_uint<8> QUANTISATION_MATRIX[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

// Standard low-frequency to high-frequency traversal.
static const ap_uint<6> ZIGZAG_ORDER[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

void load_level_shifted_block(
    const std::uint8_t* input_pixels,
    int block_x,
    int block_y,
    shifted_pixel_t block[8][8]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=block complete dim=0

    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 8; ++column) {
#pragma HLS PIPELINE II=1
            const int image_x = block_x * 8 + column;
            const int image_y = block_y * 8 + row;
            const int input_index =
                image_y * IMAGE_WIDTH + image_x;

            block[row][column] =
                static_cast<int>(input_pixels[input_index]) - 128;
        }
    }
}

void perform_dct(
    const shifted_pixel_t input[8][8],
    dct_value_t output[8][8]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=input complete dim=0
#pragma HLS ARRAY_PARTITION variable=output complete dim=0
#pragma HLS ARRAY_PARTITION variable=DCT_MATRIX complete dim=0

    first_stage_t temporary[8][8];
#pragma HLS ARRAY_PARTITION variable=temporary complete dim=0

    for (int frequency_row = 0;
         frequency_row < 8;
         ++frequency_row) {
        for (int column = 0; column < 8; ++column) {
#pragma HLS PIPELINE II=1
            first_stage_t sum = 0;

            for (int row = 0; row < 8; ++row) {
#pragma HLS UNROLL
                sum +=
                    DCT_MATRIX[frequency_row][row] *
                    input[row][column];
            }

            temporary[frequency_row][column] = sum;
        }
    }

    for (int frequency_row = 0;
         frequency_row < 8;
         ++frequency_row) {
        for (int frequency_column = 0;
             frequency_column < 8;
             ++frequency_column) {
#pragma HLS PIPELINE II=1
            dct_value_t sum = 0;

            for (int column = 0; column < 8; ++column) {
#pragma HLS UNROLL
                sum +=
                    temporary[frequency_row][column] *
                    DCT_MATRIX[frequency_column][column];
            }

            output[frequency_row][frequency_column] = sum;
        }
    }
}

quantised_t quantise_and_round(
    dct_value_t value,
    ap_uint<8> divisor
) {
#pragma HLS INLINE

    const quantisation_ratio_t ratio =
        value / divisor;

    const ap_int<40> raw =
        ratio.range(39, 0);

    const bool negative = raw < 0;

    ap_uint<40> magnitude =
        negative
            ? ap_uint<40>(-raw)
            : ap_uint<40>(raw);

    const ap_uint<40> half =
        ap_uint<40>(1) <<
        (QUANTISATION_FRACTION_BITS - 1);

    magnitude += half;

    const ap_uint<16> rounded_magnitude =
        magnitude >> QUANTISATION_FRACTION_BITS;

    ap_int<17> signed_result =
    ap_int<17>(rounded_magnitude);

    if (negative) {
        signed_result = -signed_result;
    }

    return quantised_t(signed_result);
}

void quantise_block(
    const dct_value_t dct[8][8],
    quantised_t quantised[8][8]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=dct complete dim=0
#pragma HLS ARRAY_PARTITION variable=quantised complete dim=0
#pragma HLS ARRAY_PARTITION variable=QUANTISATION_MATRIX complete dim=0

    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 8; ++column) {
#pragma HLS PIPELINE II=1
            quantised[row][column] =
                quantise_and_round(
                    dct[row][column],
                    QUANTISATION_MATRIX[row][column]
                );
        }
    }
}

ap_uint<5> coefficient_size(quantised_t value) {
#pragma HLS INLINE

    if (value == 0) {
        return 0;
    }

    const ap_int<17> extended = value;

    ap_uint<16> magnitude =
        extended < 0
            ? ap_uint<16>(-extended)
            : ap_uint<16>(extended);

    ap_uint<5> size = 0;

    for (int bit = 0; bit < 16; ++bit) {
#pragma HLS UNROLL
        if (magnitude[bit]) {
            size = bit + 1;
        }
    }

    return size;
}

ap_uint<16> encode_amplitude(
    quantised_t value,
    ap_uint<5> size
) {
#pragma HLS INLINE

    if (value >= 0) {
        return ap_uint<16>(value);
    }

    const ap_uint<16> mask =
        (ap_uint<16>(1) << size) - 1;

    const ap_int<17> encoded =
        ap_int<17>(value) + ap_int<17>(mask);

    return ap_uint<16>(encoded);
}

bool write_bits(
    ap_uint<8> output[MAX_AC_BYTES_PER_BLOCK],
    ap_uint<16>& bit_count,
    ap_uint<16> value,
    ap_uint<5> number_of_bits
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=8 dim=1

    for (int emitted = 0; emitted < 16; ++emitted) {
#pragma HLS PIPELINE II=2
        if (emitted < number_of_bits) {
            if (
                bit_count >=
                MAX_AC_BYTES_PER_BLOCK * 8
            ) {
                return false;
            }

            const ap_uint<5> source_bit =
                number_of_bits - 1 - emitted;

            const ap_uint<8> next_bit =
                value[source_bit];

            const ap_uint<8> byte_index =
                bit_count >> 3;

            const ap_uint<3> bit_in_byte =
                bit_count.range(2, 0);

            const ap_uint<3> destination_bit =
                7 - bit_in_byte;

            output[byte_index][destination_bit] =
                next_bit[0];

            ++bit_count;
        }
    }

    return true;
}

bool encode_ac_coefficients(
    const quantised_t quantised[8][8],
    ap_uint<16>& bit_count,
    ap_uint<8> packed_bytes[MAX_AC_BYTES_PER_BLOCK],
    ap_uint<32>& status_flags
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=quantised complete dim=0
#pragma HLS ARRAY_PARTITION variable=ZIGZAG_ORDER complete dim=1
#pragma HLS ARRAY_PARTITION variable=packed_bytes cyclic factor=8 dim=1

    for (int index = 0;
         index < MAX_AC_BYTES_PER_BLOCK;
         ++index) {
#pragma HLS PIPELINE II=1
        packed_bytes[index] = 0;
    }

    bit_count = 0;
    ap_uint<6> zero_run = 0;

    for (int zigzag_index = 1;
         zigzag_index < 64;
         ++zigzag_index) {
        const ap_uint<6> matrix_index =
            ZIGZAG_ORDER[zigzag_index];

        const ap_uint<3> row =
            matrix_index >> 3;

        const ap_uint<3> column =
            matrix_index.range(2, 0);

        const quantised_t value =
            quantised[row][column];

        if (value == 0) {
            ++zero_run;
            continue;
        }

        for (int zrl_index = 0;
             zrl_index < 3;
             ++zrl_index) {
            if (zero_run >= 16) {
                if (!write_bits(
                        packed_bytes,
                        bit_count,
                        0xF0,
                        8
                    )) {
                    status_flags |= 2;
                    return false;
                }

                zero_run -= 16;
            }
        }

        const ap_uint<5> size =
            coefficient_size(value);

        if (size == 0 || size > 15) {
            status_flags |= 1;
            return false;
        }

        const ap_uint<8> run_size_symbol =
            (ap_uint<8>(zero_run) << 4) |
            ap_uint<8>(size);

        if (!write_bits(
                packed_bytes,
                bit_count,
                run_size_symbol,
                8
            )) {
            status_flags |= 2;
            return false;
        }

        if (!write_bits(
                packed_bytes,
                bit_count,
                encode_amplitude(value, size),
                size
            )) {
            status_flags |= 2;
            return false;
        }

        zero_run = 0;
    }

    if (zero_run > 0) {
        if (!write_bits(
                packed_bytes,
                bit_count,
                0x00,
                8
            )) {
            status_flags |= 2;
            return false;
        }
    }

    return true;
}

} // namespace

extern "C" void compression_kernel(
    const std::uint8_t* input_pixels,
    std::int16_t* dc_values,
    std::uint16_t* ac_bit_counts,
    std::uint8_t* ac_data,
    std::uint32_t* compact_size_bytes,
    std::uint32_t* status
) {
#pragma HLS INTERFACE m_axi port=input_pixels offset=slave bundle=gmem0 depth=IMAGE_PIXELS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=dc_values offset=slave bundle=gmem1 depth=NUM_BLOCKS
#pragma HLS INTERFACE m_axi port=ac_bit_counts offset=slave bundle=gmem1 depth=NUM_BLOCKS
#pragma HLS INTERFACE m_axi port=ac_data offset=slave bundle=gmem1 depth=TOTAL_AC_STORAGE_BYTES max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=compact_size_bytes offset=slave bundle=gmem1 depth=1
#pragma HLS INTERFACE m_axi port=status offset=slave bundle=gmem1 depth=1

#pragma HLS INTERFACE s_axilite port=input_pixels bundle=control
#pragma HLS INTERFACE s_axilite port=dc_values bundle=control
#pragma HLS INTERFACE s_axilite port=ac_bit_counts bundle=control
#pragma HLS INTERFACE s_axilite port=ac_data bundle=control
#pragma HLS INTERFACE s_axilite port=compact_size_bytes bundle=control
#pragma HLS INTERFACE s_axilite port=status bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    ap_uint<32> compact_size = GSC2_HEADER_BYTES;
    ap_uint<32> status_flags = 0;

    for (int block_y = 0;
         block_y < BLOCKS_Y;
         ++block_y) {
        for (int block_x = 0;
             block_x < BLOCKS_X;
             ++block_x) {
            shifted_pixel_t pixels[8][8];
            dct_value_t dct[8][8];
            quantised_t quantised[8][8];
            ap_uint<8> packed_bytes[
                MAX_AC_BYTES_PER_BLOCK
            ];
            ap_uint<16> bit_count = 0;

#pragma HLS ARRAY_PARTITION variable=pixels complete dim=0
#pragma HLS ARRAY_PARTITION variable=dct complete dim=0
#pragma HLS ARRAY_PARTITION variable=quantised complete dim=0
#pragma HLS ARRAY_PARTITION variable=packed_bytes cyclic factor=8 dim=1

            load_level_shifted_block(
                input_pixels,
                block_x,
                block_y,
                pixels
            );

            perform_dct(pixels, dct);
            quantise_block(dct, quantised);

            encode_ac_coefficients(
                quantised,
                bit_count,
                packed_bytes,
                status_flags
            );

            const int block_index =
                block_y * BLOCKS_X + block_x;

            dc_values[block_index] =
                static_cast<std::int16_t>(
                    quantised[0][0]
                );

            ac_bit_counts[block_index] =
                static_cast<std::uint16_t>(
                    bit_count
                );

            const ap_uint<16> byte_count =
                (bit_count + 7) >> 3;

            const int output_base =
                block_index *
                MAX_AC_BYTES_PER_BLOCK;

            for (int byte_index = 0;
                 byte_index < MAX_AC_BYTES_PER_BLOCK;
                 ++byte_index) {
#pragma HLS PIPELINE II=1
                if (byte_index < byte_count) {
                    ac_data[output_base + byte_index] =
                        static_cast<std::uint8_t>(
                            packed_bytes[byte_index]
                        );
                }
            }

            compact_size += 4 + byte_count;
        }
    }

    compact_size_bytes[0] =
        static_cast<std::uint32_t>(compact_size);

    status[0] =
        static_cast<std::uint32_t>(status_flags);
}
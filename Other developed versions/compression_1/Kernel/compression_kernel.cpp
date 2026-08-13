#include "compression_kernal.hpp"

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

namespace {

using pixel_t = ap_int<9>;
using dct_coefficient_t = ap_fixed<18, 2, AP_RND_CONV, AP_SAT>;
using first_stage_t = ap_fixed<25, 11, AP_RND_CONV, AP_SAT>;
using second_stage_accumulator_t = ap_fixed<44, 14, AP_RND_CONV, AP_SAT>;
using dct_value_t = ap_fixed<32, 13, AP_RND_CONV, AP_SAT>;
using reciprocal_t = ap_ufixed<24, 1, AP_RND_CONV, AP_SAT>;
using quantisation_product_t = ap_fixed<56, 14, AP_RND_CONV, AP_SAT>;

constexpr int QUANTISATION_PRODUCT_WIDTH = 56;
constexpr int QUANTISATION_PRODUCT_INTEGER_BITS = 14;
constexpr int QUANTISATION_PRODUCT_FRACTION_BITS =
    QUANTISATION_PRODUCT_WIDTH - QUANTISATION_PRODUCT_INTEGER_BITS;

struct RLEPair {
    std::uint8_t zero_run;
    std::int16_t value;
};

static const dct_coefficient_t DCT_MATRIX[8][8] = {
    { 0.353553391,  0.353553391,  0.353553391,  0.353553391,  0.353553391,  0.353553391,  0.353553391,  0.353553391 },
    { 0.490392640,  0.415734806,  0.277785117,  0.097545161, -0.097545161, -0.277785117, -0.415734806, -0.490392640 },
    { 0.461939766,  0.191341716, -0.191341716, -0.461939766, -0.461939766, -0.191341716,  0.191341716,  0.461939766 },
    { 0.415734806, -0.097545161, -0.490392640, -0.277785117,  0.277785117,  0.490392640,  0.097545161, -0.415734806 },
    { 0.353553391, -0.353553391, -0.353553391,  0.353553391,  0.353553391, -0.353553391, -0.353553391,  0.353553391 },
    { 0.277785117, -0.490392640,  0.097545161,  0.415734806, -0.415734806, -0.097545161,  0.490392640, -0.277785117 },
    { 0.191341716, -0.461939766,  0.461939766, -0.191341716, -0.191341716,  0.461939766, -0.461939766,  0.191341716 },
    { 0.097545161, -0.277785117,  0.415734806, -0.490392640,  0.490392640, -0.415734806,  0.277785117, -0.097545161 }
};

// Retained from the original software compressor so hardware output matches it.
static const std::uint8_t ZIGZAG_ORDER[BLOCK_ELEMENTS] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

// Reciprocal quantisation values in the same zigzag order.
static const reciprocal_t INVERSE_QUANTISATION_ZIGZAG[BLOCK_ELEMENTS] = {
    0.062500000000, 0.090909090909, 0.025000000000, 0.019607843137,
    0.016666666667, 0.018181818182, 0.034482758621, 0.019607843137,
    0.100000000000, 0.041666666667, 0.016393442623, 0.017241379310,
    0.071428571429, 0.045454545455, 0.011494252874, 0.018181818182,
    0.062500000000, 0.083333333333, 0.038461538462, 0.076923076923,
    0.058823529412, 0.012500000000, 0.028571428571, 0.015625000000,
    0.083333333333, 0.052631578947, 0.062500000000, 0.071428571429,
    0.016129032258, 0.041666666667, 0.012345679012, 0.008264462810,
    0.071428571429, 0.041666666667, 0.017857142857, 0.055555555556,
    0.012987012987, 0.009615384615, 0.009708737864, 0.008333333333,
    0.025000000000, 0.014492753623, 0.045454545455, 0.009708737864,
    0.008849557522, 0.011494252874, 0.009900990099, 0.008928571429,
    0.017543859649, 0.027027027027, 0.009174311927, 0.010869565217,
    0.012820512821, 0.013888888889, 0.010204081633, 0.010000000000,
    0.017857142857, 0.014705882353, 0.020408163265, 0.015625000000,
    0.010869565217, 0.010526315789, 0.009708737864, 0.010101010101
};

static first_stage_t dot_product_pixels(
    int r, pixel_t x0, pixel_t x1, pixel_t x2, pixel_t x3,
    pixel_t x4, pixel_t x5, pixel_t x6, pixel_t x7
) {
#pragma HLS INLINE
    const first_stage_t p0 = DCT_MATRIX[r][0] * x0;
    const first_stage_t p1 = DCT_MATRIX[r][1] * x1;
    const first_stage_t p2 = DCT_MATRIX[r][2] * x2;
    const first_stage_t p3 = DCT_MATRIX[r][3] * x3;
    const first_stage_t p4 = DCT_MATRIX[r][4] * x4;
    const first_stage_t p5 = DCT_MATRIX[r][5] * x5;
    const first_stage_t p6 = DCT_MATRIX[r][6] * x6;
    const first_stage_t p7 = DCT_MATRIX[r][7] * x7;
    return ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
}

static dct_value_t dot_product_first_stage(
    int r, first_stage_t x0, first_stage_t x1, first_stage_t x2, first_stage_t x3,
    first_stage_t x4, first_stage_t x5, first_stage_t x6, first_stage_t x7
) {
#pragma HLS INLINE
    const second_stage_accumulator_t p0 = DCT_MATRIX[r][0] * x0;
    const second_stage_accumulator_t p1 = DCT_MATRIX[r][1] * x1;
    const second_stage_accumulator_t p2 = DCT_MATRIX[r][2] * x2;
    const second_stage_accumulator_t p3 = DCT_MATRIX[r][3] * x3;
    const second_stage_accumulator_t p4 = DCT_MATRIX[r][4] * x4;
    const second_stage_accumulator_t p5 = DCT_MATRIX[r][5] * x5;
    const second_stage_accumulator_t p6 = DCT_MATRIX[r][6] * x6;
    const second_stage_accumulator_t p7 = DCT_MATRIX[r][7] * x7;
    return static_cast<dct_value_t>(
        ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
    );
}

static void perform_dct_block(
    const pixel_t input[BLOCK_SIZE][BLOCK_SIZE],
    dct_value_t output[BLOCK_SIZE][BLOCK_SIZE]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=DCT_MATRIX complete dim=0
#pragma HLS ARRAY_PARTITION variable=input complete dim=0
#pragma HLS ARRAY_PARTITION variable=output complete dim=0

    first_stage_t temporary[BLOCK_SIZE][BLOCK_SIZE];
#pragma HLS ARRAY_PARTITION variable=temporary complete dim=0

DCT_FIRST_ROWS:
    for (int fr = 0; fr < BLOCK_SIZE; ++fr) {
DCT_FIRST_COLUMNS:
        for (int c = 0; c < BLOCK_SIZE; ++c) {
#pragma HLS PIPELINE II=1
            temporary[fr][c] = dot_product_pixels(
                fr,
                input[0][c], input[1][c], input[2][c], input[3][c],
                input[4][c], input[5][c], input[6][c], input[7][c]
            );
        }
    }

DCT_SECOND_ROWS:
    for (int fr = 0; fr < BLOCK_SIZE; ++fr) {
DCT_SECOND_COLUMNS:
        for (int fc = 0; fc < BLOCK_SIZE; ++fc) {
#pragma HLS PIPELINE II=1
            output[fr][fc] = dot_product_first_stage(
                fc,
                temporary[fr][0], temporary[fr][1], temporary[fr][2], temporary[fr][3],
                temporary[fr][4], temporary[fr][5], temporary[fr][6], temporary[fr][7]
            );
        }
    }
}

static std::int16_t quantise_and_round(dct_value_t value, reciprocal_t reciprocal) {
#pragma HLS INLINE
    const quantisation_product_t scaled = value * reciprocal;

    ap_int<QUANTISATION_PRODUCT_WIDTH> raw_bits =
        scaled.range(QUANTISATION_PRODUCT_WIDTH - 1, 0);

    ap_int<QUANTISATION_PRODUCT_WIDTH> half = 1;
    half <<= QUANTISATION_PRODUCT_FRACTION_BITS - 1;

    ap_int<QUANTISATION_PRODUCT_WIDTH> rounded;
    if (raw_bits >= 0) {
        rounded = (raw_bits + half) >> QUANTISATION_PRODUCT_FRACTION_BITS;
    } else {
        const ap_int<QUANTISATION_PRODUCT_WIDTH> magnitude = -raw_bits;
        rounded = -((magnitude + half) >> QUANTISATION_PRODUCT_FRACTION_BITS);
    }

    if (rounded > 32767) return 32767;
    if (rounded < -32768) return -32768;
    return static_cast<std::int16_t>(rounded);
}

// DDR -> 8-row BRAM strip -> stream in 8x8 block order.
static void load_and_shift_stage(
    const std::uint8_t* input_pixels,
    hls::stream<pixel_t>& pixel_stream
) {
#pragma HLS INLINE off
    pixel_t row_strip[BLOCK_SIZE][IMAGE_WIDTH];
#pragma HLS ARRAY_PARTITION variable=row_strip complete dim=1
#pragma HLS BIND_STORAGE variable=row_strip type=RAM_1P impl=BRAM

BLOCK_ROW_LOOP:
    for (int block_y = 0; block_y < BLOCKS_Y; ++block_y) {
LOAD_STRIP_ROWS:
        for (int row = 0; row < BLOCK_SIZE; ++row) {
            const int row_base = (block_y * BLOCK_SIZE + row) * IMAGE_WIDTH;
LOAD_STRIP_COLUMNS:
            for (int column = 0; column < IMAGE_WIDTH; ++column) {
#pragma HLS PIPELINE II=1
                row_strip[row][column] = static_cast<pixel_t>(
                    static_cast<int>(input_pixels[row_base + column]) - 128
                );
            }
        }

STREAM_BLOCK_COLUMNS:
        for (int block_x = 0; block_x < BLOCKS_X; ++block_x) {
            const int block_base = block_x * BLOCK_SIZE;
STREAM_BLOCK_ROWS:
            for (int row = 0; row < BLOCK_SIZE; ++row) {
STREAM_BLOCK_PIXELS:
                for (int column = 0; column < BLOCK_SIZE; ++column) {
#pragma HLS PIPELINE II=1
                    pixel_stream.write(row_strip[row][block_base + column]);
                }
            }
        }
    }
}

// Stream -> local 8x8 block -> DCT -> stream in zigzag order.
static void dct_stage(
    hls::stream<pixel_t>& pixel_stream,
    hls::stream<dct_value_t>& dct_stream
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ZIGZAG_ORDER complete dim=1

DCT_BLOCKS:
    for (int block = 0; block < NUM_BLOCKS; ++block) {
        pixel_t input_block[BLOCK_SIZE][BLOCK_SIZE];
        dct_value_t dct_block[BLOCK_SIZE][BLOCK_SIZE];
#pragma HLS ARRAY_PARTITION variable=input_block complete dim=0
#pragma HLS ARRAY_PARTITION variable=dct_block complete dim=0

READ_BLOCK:
        for (int index = 0; index < BLOCK_ELEMENTS; ++index) {
#pragma HLS PIPELINE II=1
            input_block[index >> 3][index & 7] = pixel_stream.read();
        }

        perform_dct_block(input_block, dct_block);

STREAM_DCT_ZIGZAG:
        for (int i = 0; i < BLOCK_ELEMENTS; ++i) {
#pragma HLS PIPELINE II=1
            const std::uint8_t matrix_index = ZIGZAG_ORDER[i];
            dct_stream.write(dct_block[matrix_index >> 3][matrix_index & 7]);
        }
    }
}

// DCT stream -> quantisation -> simple {zero_run,value} RLE streams.
static void quantise_and_rle_stage(
    hls::stream<dct_value_t>& dct_stream,
    hls::stream<std::int16_t>& dc_stream,
    hls::stream<std::uint8_t>& pair_count_stream,
    hls::stream<RLEPair>& pair_stream
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=INVERSE_QUANTISATION_ZIGZAG complete dim=1

RLE_BLOCKS:
    for (int block = 0; block < NUM_BLOCKS; ++block) {
        std::uint8_t local_zero_runs[MAX_RLE_PAIRS_PER_BLOCK];
        std::int16_t local_pair_values[MAX_RLE_PAIRS_PER_BLOCK];
#pragma HLS BIND_STORAGE variable=local_zero_runs type=RAM_2P impl=LUTRAM
#pragma HLS BIND_STORAGE variable=local_pair_values type=RAM_2P impl=LUTRAM

        std::uint8_t zero_run = 0;
        std::uint8_t pair_count = 0;

QUANTISE_RLE:
        for (int i = 0; i < BLOCK_ELEMENTS; ++i) {
#pragma HLS PIPELINE II=1
            const std::int16_t q = quantise_and_round(
                dct_stream.read(),
                INVERSE_QUANTISATION_ZIGZAG[i]
            );

            if (i == 0) {
                dc_stream.write(q);
            } else if (q == 0) {
                ++zero_run;
            } else {
                local_zero_runs[pair_count] = zero_run;
                local_pair_values[pair_count] = q;
                ++pair_count;
                zero_run = 0;
            }
        }

        pair_count_stream.write(pair_count);

STREAM_VALID_PAIRS:
        for (int p = 0; p < pair_count; ++p) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=63 avg=10
            RLEPair pair;
            pair.zero_run = local_zero_runs[p];
            pair.value = local_pair_values[p];
            pair_stream.write(pair);
        }
    }
}

// Final streams -> output DDR buffers. Each block has 63 reserved pair slots.
static void write_results_stage(
    hls::stream<std::int16_t>& dc_stream,
    hls::stream<std::uint8_t>& pair_count_stream,
    hls::stream<RLEPair>& pair_stream,
    std::int16_t* dc_values,
    std::uint8_t* pair_counts,
    std::uint8_t* zero_runs,
    std::int16_t* pair_values
) {
#pragma HLS INLINE off

WRITE_BLOCKS:
    for (int block = 0; block < NUM_BLOCKS; ++block) {
        const int base = block * MAX_RLE_PAIRS_PER_BLOCK;
        const std::uint8_t count = pair_count_stream.read();

        dc_values[block] = dc_stream.read();
        pair_counts[block] = count;

WRITE_PAIRS:
        for (int p = 0; p < count; ++p) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=63 avg=10
            const RLEPair pair = pair_stream.read();
            zero_runs[base + p] = pair.zero_run;
            pair_values[base + p] = pair.value;
        }
    }
}

} // namespace

extern "C" void compression_stream_kernel(
    const std::uint8_t* __restrict input_pixels,
    std::int16_t* __restrict dc_values,
    std::uint8_t* __restrict pair_counts,
    std::uint8_t* __restrict zero_runs,
    std::int16_t* __restrict pair_values
) {
#pragma HLS INTERFACE mode=m_axi port=input_pixels offset=slave bundle=gmem0 depth=IMAGE_PIXELS max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE mode=m_axi port=dc_values offset=slave bundle=gmem1 depth=NUM_BLOCKS max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE mode=m_axi port=pair_counts offset=slave bundle=gmem2 depth=NUM_BLOCKS max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE mode=m_axi port=zero_runs offset=slave bundle=gmem3 depth=TOTAL_RLE_PAIR_SLOTS max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE mode=m_axi port=pair_values offset=slave bundle=gmem4 depth=TOTAL_RLE_PAIR_SLOTS max_write_burst_length=64 num_write_outstanding=16

#pragma HLS INTERFACE mode=s_axilite port=input_pixels bundle=control
#pragma HLS INTERFACE mode=s_axilite port=dc_values bundle=control
#pragma HLS INTERFACE mode=s_axilite port=pair_counts bundle=control
#pragma HLS INTERFACE mode=s_axilite port=zero_runs bundle=control
#pragma HLS INTERFACE mode=s_axilite port=pair_values bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    hls::stream<pixel_t> pixel_stream;
    hls::stream<dct_value_t> dct_stream;
    hls::stream<std::int16_t> dc_stream;
    hls::stream<std::uint8_t> pair_count_stream;
    hls::stream<RLEPair> pair_stream;

#pragma HLS STREAM variable=pixel_stream depth=128
#pragma HLS STREAM variable=dct_stream depth=128
#pragma HLS STREAM variable=dc_stream depth=16
#pragma HLS STREAM variable=pair_count_stream depth=16
#pragma HLS STREAM variable=pair_stream depth=128

#pragma HLS DATAFLOW

    load_and_shift_stage(input_pixels, pixel_stream);
    dct_stage(pixel_stream, dct_stream);
    quantise_and_rle_stage(dct_stream, dc_stream, pair_count_stream, pair_stream);
    write_results_stage(
        dc_stream, pair_count_stream, pair_stream,
        dc_values, pair_counts, zero_runs, pair_values
    );
}


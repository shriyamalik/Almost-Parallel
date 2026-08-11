#include "compression_hls_common.hpp"

#include <ap_int.h>
#include <cstdint>

using namespace compression_detail;

extern "C" void dct_kernel(
    const shifted_storage_t* shifted_pixels,
    dct_storage_t* dct_values
) {
#pragma HLS INTERFACE m_axi port=shifted_pixels offset=slave bundle=gmem0 depth=TOTAL_COEFFICIENTS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=dct_values offset=slave bundle=gmem1 depth=TOTAL_COEFFICIENTS max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=shifted_pixels bundle=control
#pragma HLS INTERFACE s_axilite port=dct_values bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

DCT_BLOCKS:
    for (int block = 0; block < NUM_BLOCKS; ++block) {
        shifted_pixel_t input_block[8][8];
        dct_value_t output_block[8][8];

#pragma HLS ARRAY_PARTITION variable=input_block complete dim=0
#pragma HLS ARRAY_PARTITION variable=output_block complete dim=0

    DCT_LOAD_ROWS:
        for (int row = 0; row < BLOCK_SIZE; ++row) {
        DCT_LOAD_COLUMNS:
            for (int column = 0;
                 column < BLOCK_SIZE;
                 ++column) {
#pragma HLS PIPELINE II=1
                const int index =
                    block * BLOCK_ELEMENTS +
                    row * BLOCK_SIZE + column;

                input_block[row][column] =
                    shifted_pixels[index];
            }
        }

        perform_dct(input_block, output_block);

    DCT_STORE_ROWS:
        for (int row = 0; row < BLOCK_SIZE; ++row) {
        DCT_STORE_COLUMNS:
            for (int column = 0;
                 column < BLOCK_SIZE;
                 ++column) {
#pragma HLS PIPELINE II=1
                const int index =
                    block * BLOCK_ELEMENTS +
                    row * BLOCK_SIZE + column;

                const ap_int<DCT_STORAGE_WIDTH> raw_bits =
                    output_block[row][column].range(
                        DCT_STORAGE_WIDTH - 1,
                        0
                    );

                dct_values[index] =
                    static_cast<dct_storage_t>(
                        raw_bits.to_int64()
                    );
            }
        }
    }
}

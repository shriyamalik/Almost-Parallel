#include "compression_hls_common.hpp"

#include <ap_int.h>
#include <cstdint>

using namespace compression_detail;

extern "C" void quantise_kernel(
    const dct_storage_t* dct_values,
    quantised_storage_t* quantised_values
) {
#pragma HLS INTERFACE m_axi port=dct_values offset=slave bundle=gmem0 depth=TOTAL_COEFFICIENTS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=quantised_values offset=slave bundle=gmem1 depth=TOTAL_COEFFICIENTS max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=dct_values bundle=control
#pragma HLS INTERFACE s_axilite port=quantised_values bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS ARRAY_PARTITION variable=QUANTISATION_MATRIX complete dim=0

QUANTISE_BLOCKS:
    for (int block = 0; block < NUM_BLOCKS; ++block) {
    QUANTISE_ROWS:
        for (int row = 0; row < BLOCK_SIZE; ++row) {
        QUANTISE_COLUMNS:
            for (int column = 0;
                 column < BLOCK_SIZE;
                 ++column) {
#pragma HLS PIPELINE II=1
                const int index =
                    block * BLOCK_ELEMENTS +
                    row * BLOCK_SIZE + column;

                const ap_int<DCT_STORAGE_WIDTH> raw_bits =
                    static_cast<ap_int<DCT_STORAGE_WIDTH>>(
                        dct_values[index]
                    );

                dct_value_t value;
                value.range(DCT_STORAGE_WIDTH - 1, 0) =
                    raw_bits.range(DCT_STORAGE_WIDTH - 1, 0);

                quantised_values[index] =
                    static_cast<quantised_storage_t>(
                        quantise_and_round(
                            value,
                            QUANTISATION_MATRIX[row][column]
                        )
                    );
            }
        }
    }
}

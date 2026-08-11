#include "compression_hls_common.hpp"

#include <ap_int.h>
#include <cstdint>

using namespace compression_detail;

extern "C" void encode_kernel(
    const quantised_storage_t* quantised_values,
    std::int16_t* dc_values,
    std::uint16_t* ac_bit_counts,
    std::uint8_t* ac_data,
    std::uint32_t* compact_size_bytes,
    std::uint32_t* status
) {
#pragma HLS INTERFACE m_axi port=quantised_values offset=slave bundle=gmem0 depth=TOTAL_COEFFICIENTS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=dc_values offset=slave bundle=gmem1 depth=NUM_BLOCKS
#pragma HLS INTERFACE m_axi port=ac_bit_counts offset=slave bundle=gmem1 depth=NUM_BLOCKS
#pragma HLS INTERFACE m_axi port=ac_data offset=slave bundle=gmem1 depth=TOTAL_AC_STORAGE_BYTES max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=compact_size_bytes offset=slave bundle=gmem1 depth=1
#pragma HLS INTERFACE m_axi port=status offset=slave bundle=gmem1 depth=1
#pragma HLS INTERFACE s_axilite port=quantised_values bundle=control
#pragma HLS INTERFACE s_axilite port=dc_values bundle=control
#pragma HLS INTERFACE s_axilite port=ac_bit_counts bundle=control
#pragma HLS INTERFACE s_axilite port=ac_data bundle=control
#pragma HLS INTERFACE s_axilite port=compact_size_bytes bundle=control
#pragma HLS INTERFACE s_axilite port=status bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    ap_uint<32> compact_size = GSC2_HEADER_BYTES;
    ap_uint<32> status_flags = 0;

ENCODE_BLOCKS:
    for (int block = 0; block < NUM_BLOCKS; ++block) {
        quantised_t quantised[8][8];
        ap_uint<8> packed_bytes[MAX_AC_BYTES_PER_BLOCK];
        ap_uint<16> bit_count = 0;

#pragma HLS ARRAY_PARTITION variable=quantised complete dim=0
#pragma HLS ARRAY_PARTITION variable=packed_bytes cyclic factor=8 dim=1

    ENCODE_LOAD_ROWS:
        for (int row = 0; row < BLOCK_SIZE; ++row) {
        ENCODE_LOAD_COLUMNS:
            for (int column = 0;
                 column < BLOCK_SIZE;
                 ++column) {
#pragma HLS PIPELINE II=1
                const int index =
                    block * BLOCK_ELEMENTS +
                    row * BLOCK_SIZE + column;

                quantised[row][column] =
                    quantised_values[index];
            }
        }

        encode_ac_coefficients(
            quantised,
            bit_count,
            packed_bytes,
            status_flags
        );

        dc_values[block] =
            static_cast<std::int16_t>(quantised[0][0]);
        ac_bit_counts[block] =
            static_cast<std::uint16_t>(bit_count);

        const ap_uint<16> byte_count =
            (bit_count + 7) >> 3;
        const int output_base =
            block * MAX_AC_BYTES_PER_BLOCK;

    ENCODE_WRITE_BYTES:
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

    compact_size_bytes[0] =
        static_cast<std::uint32_t>(compact_size);
    status[0] =
        static_cast<std::uint32_t>(status_flags);
}

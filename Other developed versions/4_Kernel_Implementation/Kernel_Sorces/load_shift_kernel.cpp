#include "compression_config.hpp"

#include <cstdint>

extern "C" void load_shift_kernel(
    const std::uint8_t* input_pixels,
    shifted_storage_t* shifted_pixels
) {
#pragma HLS INTERFACE m_axi port=input_pixels offset=slave bundle=gmem0 depth=IMAGE_PIXELS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=shifted_pixels offset=slave bundle=gmem1 depth=TOTAL_COEFFICIENTS max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=input_pixels bundle=control
#pragma HLS INTERFACE s_axilite port=shifted_pixels bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

LOAD_BLOCK_ROWS:
    for (int block_y = 0; block_y < BLOCKS_Y; ++block_y) {
    LOAD_BLOCK_COLUMNS:
        for (int block_x = 0; block_x < BLOCKS_X; ++block_x) {
            const int block_index =
                block_y * BLOCKS_X + block_x;

        LOAD_ROWS:
            for (int row = 0; row < BLOCK_SIZE; ++row) {
            LOAD_COLUMNS:
                for (int column = 0;
                     column < BLOCK_SIZE;
                     ++column) {
#pragma HLS PIPELINE II=1
                    const int image_x =
                        block_x * BLOCK_SIZE + column;
                    const int image_y =
                        block_y * BLOCK_SIZE + row;
                    const int input_index =
                        image_y * IMAGE_WIDTH + image_x;
                    const int output_index =
                        block_index * BLOCK_ELEMENTS +
                        row * BLOCK_SIZE + column;

                    shifted_pixels[output_index] =
                        static_cast<shifted_storage_t>(
                            static_cast<int>(
                                input_pixels[input_index]
                            ) - 128
                        );
                }
            }
        }
    }
}

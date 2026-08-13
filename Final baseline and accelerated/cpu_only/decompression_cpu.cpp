#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

constexpr int BLOCK_SIZE = 8;
constexpr int BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE;
constexpr double PI = 3.14159265358979323846;

using DoubleBlock = array<array<double, BLOCK_SIZE>, BLOCK_SIZE>;
using IntegerBlock = array<array<int16_t, BLOCK_SIZE>, BLOCK_SIZE>;
using QuantisationMatrix = array<array<uint16_t, BLOCK_SIZE>, BLOCK_SIZE>;

struct GrayImage {
    uint32_t width = 0;
    uint32_t height = 0;
    vector<uint8_t> pixels;
};

struct CompressedHeader {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t padded_width = 0;
    uint32_t padded_height = 0;
    uint32_t block_count = 0;
    QuantisationMatrix quantisation_matrix{};
};

struct DecompressionResult {
    uintmax_t compressed_size = 0;
    uintmax_t reconstructed_size = 0;
    double elapsed_ms = 0.0;
};

constexpr int ZIGZAG_ORDER[BLOCK_ELEMENTS] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

string lowercaseExtension(const fs::path& path) {
    string extension = path.extension().string();

    transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char character) {
            return static_cast<char>(tolower(character));
        }
    );

    return extension;
}

bool isGscFile(const fs::path& path) {
    return lowercaseExtension(path) == ".gsc";
}

uint8_t readUInt8(istream& input) {
    char byte = 0;

    if (!input.get(byte)) {
        throw runtime_error("Unexpected end of compressed file.");
    }

    return static_cast<uint8_t>(
        static_cast<unsigned char>(byte)
    );
}

uint16_t readUInt16(istream& input) {
    const uint16_t byte0 = readUInt8(input);
    const uint16_t byte1 = readUInt8(input);

    return static_cast<uint16_t>(
        byte0 | (byte1 << 8)
    );
}

int16_t readInt16(istream& input) {
    return static_cast<int16_t>(readUInt16(input));
}

uint32_t readUInt32(istream& input) {
    uint32_t value = 0;

    for (int shift = 0; shift < 32; shift += 8) {
        value |= static_cast<uint32_t>(readUInt8(input)) << shift;
    }

    return value;
}

CompressedHeader readHeader(istream& input) {
    array<char, 4> magic{};

    input.read(magic.data(), static_cast<streamsize>(magic.size()));

    if (input.gcount() != static_cast<streamsize>(magic.size())) {
        throw runtime_error("Compressed file is too small.");
    }

    if (string(magic.data(), magic.size()) != "GSC1") {
        throw runtime_error("Invalid GSC file. Expected GSC1 header.");
    }

    CompressedHeader header;

    header.width = readUInt32(input);
    header.height = readUInt32(input);
    header.padded_width = readUInt32(input);
    header.padded_height = readUInt32(input);
    header.block_count = readUInt32(input);

    if (header.width == 0 || header.height == 0) {
        throw runtime_error("Invalid image dimensions in GSC header.");
    }

    if (
        header.padded_width < header.width ||
        header.padded_height < header.height ||
        header.padded_width % BLOCK_SIZE != 0 ||
        header.padded_height % BLOCK_SIZE != 0
    ) {
        throw runtime_error("Invalid padded dimensions in GSC header.");
    }

    const uint64_t expected_blocks =
        static_cast<uint64_t>(header.padded_width / BLOCK_SIZE) *
        static_cast<uint64_t>(header.padded_height / BLOCK_SIZE);

    if (expected_blocks != header.block_count) {
        throw runtime_error("Block count does not match image dimensions.");
    }

    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0; column < BLOCK_SIZE; ++column) {
            const uint16_t value = readUInt16(input);

            if (value == 0) {
                throw runtime_error("Quantisation values must be non-zero.");
            }

            header.quantisation_matrix[row][column] = value;
        }
    }

    return header;
}

DoubleBlock createDCTMatrix() {
    DoubleBlock matrix{};

    for (int frequency = 0; frequency < BLOCK_SIZE; ++frequency) {
        const double normalisation =
            frequency == 0
                ? sqrt(1.0 / BLOCK_SIZE)
                : sqrt(2.0 / BLOCK_SIZE);

        for (int position = 0; position < BLOCK_SIZE; ++position) {
            matrix[frequency][position] =
                normalisation *
                cos(
                    ((2.0 * position + 1.0) *
                     frequency *
                     PI) /
                    (2.0 * BLOCK_SIZE)
                );
        }
    }

    return matrix;
}

array<int16_t, BLOCK_ELEMENTS> readAndDecodeRleBlock(
    istream& input
) {
    array<int16_t, BLOCK_ELEMENTS> coefficients{};

    coefficients[0] = readInt16(input);

    const uint8_t pair_count = readUInt8(input);

    if (pair_count > BLOCK_ELEMENTS - 1) {
        throw runtime_error("Invalid RLE pair count.");
    }

    size_t position = 1;

    for (uint16_t pair_index = 0;
         pair_index < pair_count;
         ++pair_index) {

        const uint8_t zero_run = readUInt8(input);
        const int16_t value = readInt16(input);

        position += zero_run;

        if (position >= BLOCK_ELEMENTS) {
            throw runtime_error("RLE data extends beyond one 8 x 8 block.");
        }

        coefficients[position] = value;
        ++position;
    }

    return coefficients;
}

IntegerBlock inverseZigzag(
    const array<int16_t, BLOCK_ELEMENTS>& coefficients
) {
    IntegerBlock block{};

    for (int zigzag_index = 0;
         zigzag_index < BLOCK_ELEMENTS;
         ++zigzag_index) {

        const int matrix_index = ZIGZAG_ORDER[zigzag_index];
        const int row = matrix_index / BLOCK_SIZE;
        const int column = matrix_index % BLOCK_SIZE;

        block[row][column] = coefficients[zigzag_index];
    }

    return block;
}

DoubleBlock dequantise(
    const IntegerBlock& quantised,
    const QuantisationMatrix& quantisation_matrix
) {
    DoubleBlock dct{};

    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0; column < BLOCK_SIZE; ++column) {
            dct[row][column] =
                static_cast<double>(quantised[row][column]) *
                static_cast<double>(
                    quantisation_matrix[row][column]
                );
        }
    }

    return dct;
}

DoubleBlock performInverseDCT(
    const DoubleBlock& dct,
    const DoubleBlock& dct_matrix
) {
    DoubleBlock temporary{};
    DoubleBlock output{};

    // temporary = C^T * dct
    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int frequency_column = 0;
             frequency_column < BLOCK_SIZE;
             ++frequency_column) {

            double sum = 0.0;

            for (int frequency_row = 0;
                 frequency_row < BLOCK_SIZE;
                 ++frequency_row) {

                sum +=
                    dct_matrix[frequency_row][row] *
                    dct[frequency_row][frequency_column];
            }

            temporary[row][frequency_column] = sum;
        }
    }

    // output = temporary * C
    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0; column < BLOCK_SIZE; ++column) {
            double sum = 0.0;

            for (int frequency_column = 0;
                 frequency_column < BLOCK_SIZE;
                 ++frequency_column) {

                sum +=
                    temporary[row][frequency_column] *
                    dct_matrix[frequency_column][column];
            }

            output[row][column] = sum;
        }
    }

    return output;
}

void placeReconstructedBlock(
    GrayImage& image,
    const DoubleBlock& block,
    uint32_t start_x,
    uint32_t start_y
) {
    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0; column < BLOCK_SIZE; ++column) {
            const uint32_t image_x =
                start_x + static_cast<uint32_t>(column);

            const uint32_t image_y =
                start_y + static_cast<uint32_t>(row);

            if (image_x >= image.width || image_y >= image.height) {
                continue;
            }

            long pixel = lround(block[row][column] + 128.0);
            pixel = clamp(pixel, 0L, 255L);

            const size_t index =
                static_cast<size_t>(image_y) *
                static_cast<size_t>(image.width) +
                static_cast<size_t>(image_x);

            image.pixels[index] = static_cast<uint8_t>(pixel);
        }
    }
}

void savePgmP5(
    const fs::path& output_path,
    const GrayImage& image
) {
    ofstream output(output_path, ios::binary);

    if (!output) {
        throw runtime_error(
            "Could not create reconstructed PGM file: " +
            output_path.string()
        );
    }

    output
        << "P5\n"
        << image.width << ' ' << image.height << "\n"
        << "255\n";

    output.write(
        reinterpret_cast<const char*>(image.pixels.data()),
        static_cast<streamsize>(image.pixels.size())
    );

    if (!output) {
        throw runtime_error(
            "Error while writing reconstructed PGM file: " +
            output_path.string()
        );
    }
}

DecompressionResult decompressImage(
    const fs::path& input_path,
    const fs::path& output_path,
    const DoubleBlock& dct_matrix
) {
    const auto start_time = chrono::steady_clock::now();

    ifstream input(input_path, ios::binary);

    if (!input) {
        throw runtime_error(
            "Could not open compressed file: " +
            input_path.string()
        );
    }

    const CompressedHeader header = readHeader(input);

    GrayImage image;
    image.width = header.width;
    image.height = header.height;
    image.pixels.assign(
        static_cast<size_t>(image.width) *
        static_cast<size_t>(image.height),
        0
    );

    const uint32_t blocks_x =
        header.padded_width / BLOCK_SIZE;

    for (uint32_t block_index = 0;
         block_index < header.block_count;
         ++block_index) {

        const auto zigzag_coefficients =
            readAndDecodeRleBlock(input);

        const IntegerBlock quantised =
            inverseZigzag(zigzag_coefficients);

        const DoubleBlock dct =
            dequantise(
                quantised,
                header.quantisation_matrix
            );

        const DoubleBlock reconstructed_block =
            performInverseDCT(dct, dct_matrix);

        const uint32_t block_x = block_index % blocks_x;
        const uint32_t block_y = block_index / blocks_x;

        placeReconstructedBlock(
            image,
            reconstructed_block,
            block_x * BLOCK_SIZE,
            block_y * BLOCK_SIZE
        );
    }

    savePgmP5(output_path, image);

    const auto end_time = chrono::steady_clock::now();

    DecompressionResult result;
    result.compressed_size = fs::file_size(input_path);
    result.reconstructed_size = fs::file_size(output_path);
    result.elapsed_ms =
        chrono::duration<double, milli>(
            end_time - start_time
        ).count();

    return result;
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            cerr
                << "Usage:\n"
                << "  decompress_gsc <input_folder> <output_folder>\n\n"
                << "Example:\n"
                << "  decompress_gsc compressed_output reconstructed_pgm\n";

            return 1;
        }

        const fs::path input_directory = argv[1];
        const fs::path output_directory = argv[2];

        if (!fs::exists(input_directory)) {
            throw runtime_error(
                "Input folder does not exist: " +
                input_directory.string()
            );
        }

        if (!fs::is_directory(input_directory)) {
            throw runtime_error(
                "Input path is not a folder: " +
                input_directory.string()
            );
        }

        fs::create_directories(output_directory);

        vector<fs::path> input_files;

        for (const fs::directory_entry& entry :
             fs::directory_iterator(input_directory)) {

            if (
                entry.is_regular_file() &&
                isGscFile(entry.path())
            ) {
                input_files.push_back(entry.path());
            }
        }

        sort(input_files.begin(), input_files.end());

        if (input_files.empty()) {
            cout
                << "No .gsc files were found in: "
                << input_directory.string()
                << '\n';

            return 0;
        }

        const DoubleBlock dct_matrix = createDCTMatrix();

        size_t image_count = 0;
        uintmax_t total_compressed_size = 0;
        uintmax_t total_reconstructed_size = 0;
        double total_elapsed_ms = 0.0;

        cout << fixed << setprecision(3);

        for (const fs::path& input_path : input_files) {
            fs::path output_path =
                output_directory / input_path.stem();

            output_path += ".pgm";

            const DecompressionResult result =
                decompressImage(
                    input_path,
                    output_path,
                    dct_matrix
                );

            cout
                << input_path.filename().string()
                << " -> "
                << output_path.filename().string()
                << " | compressed: "
                << result.compressed_size
                << " bytes"
                << " | reconstructed PGM: "
                << result.reconstructed_size
                << " bytes"
                << " | time: "
                << result.elapsed_ms
                << " ms\n";

            total_compressed_size += result.compressed_size;
            total_reconstructed_size += result.reconstructed_size;
            total_elapsed_ms += result.elapsed_ms;
            ++image_count;
        }

        cout << "\nDecompression complete\n";
        cout << "Images decompressed: " << image_count << '\n';
        cout << "Output format: binary PGM P5\n";
        cout << "Total compressed size: "
             << total_compressed_size
             << " bytes\n";
        cout << "Total reconstructed PGM size: "
             << total_reconstructed_size
             << " bytes\n";
        cout << "Total decompression time: "
             << total_elapsed_ms
             << " ms\n";

        return 0;
    }
    catch (const exception& error) {
        cerr
            << "Decompression failed: "
            << error.what()
            << '\n';

        return 1;
    }
}
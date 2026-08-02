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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace fs = std::filesystem;

constexpr int BLOCK_SIZE = 8;
constexpr int BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE;
constexpr double PI = 3.14159265358979323846;

using DoubleArray = array<array<double, BLOCK_SIZE>, BLOCK_SIZE>;
using IntegerArray = array<array<int16_t, BLOCK_SIZE>, BLOCK_SIZE>;
using QuantisationMatrix =
    array<array<uint16_t, BLOCK_SIZE>, BLOCK_SIZE>;

struct GrayImage {
    int width = 0;
    int height = 0;
    vector<uint8_t> pixels;
};

struct CompressedImage {
    uint32_t original_width = 0;
    uint32_t original_height = 0;
    uint32_t processed_width = 0;
    uint32_t processed_height = 0;
    uint32_t block_count = 0;
    QuantisationMatrix quantisation_matrix{};
};

struct DecompressionResult {
    uintmax_t compressed_size = 0;
    uintmax_t decompressed_size = 0;
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


class BitReader {
private:
    const vector<uint8_t>& bytes;
    size_t bit_count = 0;
    size_t bit_position = 0;

public:
    BitReader(
        const vector<uint8_t>& data,
        size_t valid_bit_count
    )
        : bytes(data),
          bit_count(valid_bit_count) {
        if (bit_count > bytes.size() * 8ULL) {
            throw runtime_error(
                "AC bit count exceeds stored AC data"
            );
        }
    }

    size_t remainingBits() const {
        return bit_count - bit_position;
    }

    uint32_t readBits(uint8_t number_of_bits) {
        if (number_of_bits > 32) {
            throw runtime_error(
                "Cannot read more than 32 bits at once"
            );
        }

        if (remainingBits() < number_of_bits) {
            throw runtime_error(
                "Unexpected end of packed AC bitstream"
            );
        }

        uint32_t value = 0;

        for (uint8_t index = 0;
             index < number_of_bits;
             ++index) {
            const size_t byte_index = bit_position / 8;
            const int bit_index =
                7 - static_cast<int>(bit_position % 8);

            const uint8_t bit = static_cast<uint8_t>(
                (bytes[byte_index] >> bit_index) & 1U
            );

            value = (value << 1) | bit;
            ++bit_position;
        }

        return value;
    }
};

uint8_t readUInt8(istream& input) {
    const int value = input.get();

    if (value == EOF) {
        throw runtime_error(
            "Unexpected end of compressed file"
        );
    }

    return static_cast<uint8_t>(value);
}

uint16_t readUInt16(istream& input) {
    const uint16_t byte_0 = readUInt8(input);
    const uint16_t byte_1 = readUInt8(input);

    return static_cast<uint16_t>(
        byte_0 | (byte_1 << 8)
    );
}

int16_t readInt16(istream& input) {
    return static_cast<int16_t>(readUInt16(input));
}

uint32_t readUInt32(istream& input) {
    uint32_t value = 0;

    for (int shift = 0; shift < 32; shift += 8) {
        value |= static_cast<uint32_t>(
            readUInt8(input)
        ) << shift;
    }

    return value;
}

DoubleArray createDCTMatrix() {
    DoubleArray dct{};

    for (int frequency = 0;
         frequency < BLOCK_SIZE;
         ++frequency) {
        const double normalisation =
            frequency == 0
                ? sqrt(1.0 / BLOCK_SIZE)
                : sqrt(2.0 / BLOCK_SIZE);

        for (int position = 0;
             position < BLOCK_SIZE;
             ++position) {
            dct[frequency][position] =
                normalisation *
                cos(
                    ((2.0 * position + 1.0) *
                     frequency * PI) /
                    (2.0 * BLOCK_SIZE)
                );
        }
    }

    return dct;
}

int16_t decodeAmplitude(
    uint32_t amplitude_bits,
    uint8_t size
) {
    if (size == 0 || size > 15) {
        throw runtime_error(
            "Invalid AC coefficient size"
        );
    }

    const uint32_t positive_threshold =
        1U << (size - 1);

    int32_t value = 0;

    if (amplitude_bits >= positive_threshold) {
        value = static_cast<int32_t>(amplitude_bits);
    } else {
        const uint32_t mask = (1U << size) - 1U;

        value =
            static_cast<int32_t>(amplitude_bits) -
            static_cast<int32_t>(mask);
    }

    if (
        value < numeric_limits<int16_t>::min() ||
        value > numeric_limits<int16_t>::max()
    ) {
        throw runtime_error(
            "Decoded AC coefficient is outside int16_t range"
        );
    }

    return static_cast<int16_t>(value);
}

array<int16_t, BLOCK_ELEMENTS> decodeBlock(
    int16_t dc,
    const vector<uint8_t>& ac_data,
    uint16_t ac_bit_count
) {
    array<int16_t, BLOCK_ELEMENTS> coefficients{};
    coefficients[0] = dc;

    BitReader reader(ac_data, ac_bit_count);
    int coefficient_index = 1;
    bool found_eob = false;

    while (
        coefficient_index < BLOCK_ELEMENTS &&
        reader.remainingBits() > 0
    ) {
        if (reader.remainingBits() < 8) {
            throw runtime_error(
                "Incomplete RUN/SIZE symbol in AC bitstream"
            );
        }

        const uint8_t symbol =
            static_cast<uint8_t>(reader.readBits(8));

        if (symbol == 0x00) {
            found_eob = true;
            break;
        }

        if (symbol == 0xF0) {
            coefficient_index += 16;

            if (coefficient_index > BLOCK_ELEMENTS) {
                throw runtime_error(
                    "ZRL moves beyond the end of the block"
                );
            }

            continue;
        }

        const uint8_t zero_run = symbol >> 4;
        const uint8_t size = symbol & 0x0F;

        if (size == 0) {
            throw runtime_error(
                "Invalid RUN/SIZE symbol with SIZE equal to zero"
            );
        }

        coefficient_index += zero_run;

        if (coefficient_index >= BLOCK_ELEMENTS) {
            throw runtime_error(
                "RUN/SIZE symbol moves beyond the end of the block"
            );
        }

        const uint32_t amplitude_bits =
            reader.readBits(size);

        coefficients[coefficient_index] =
            decodeAmplitude(amplitude_bits, size);

        ++coefficient_index;
    }

    if (
        coefficient_index < BLOCK_ELEMENTS &&
        !found_eob &&
        reader.remainingBits() == 0
    ) {
        throw runtime_error(
            "AC bitstream ended before all coefficients or EOB"
        );
    }

    if (reader.remainingBits() != 0) {
        throw runtime_error(
            "Unused valid bits remain after decoding the AC block"
        );
    }

    return coefficients;
}

IntegerArray inverseZigzag(
    const array<int16_t, BLOCK_ELEMENTS>& coefficients
) {
    IntegerArray block{};

    for (int zigzag_index = 0;
         zigzag_index < BLOCK_ELEMENTS;
         ++zigzag_index) {
        const int matrix_index =
            ZIGZAG_ORDER[zigzag_index];

        const int row = matrix_index / BLOCK_SIZE;
        const int column = matrix_index % BLOCK_SIZE;

        block[row][column] =
            coefficients[zigzag_index];
    }

    return block;
}

DoubleArray dequantise(
    const IntegerArray& quantised,
    const QuantisationMatrix& quantisation_matrix
) {
    DoubleArray output{};

    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0;
             column < BLOCK_SIZE;
             ++column) {
            output[row][column] =
                static_cast<double>(quantised[row][column]) *
                static_cast<double>(
                    quantisation_matrix[row][column]
                );
        }
    }

    return output;
}

DoubleArray performInverseDCT(
    const DoubleArray& coefficients,
    const DoubleArray& dct_matrix
) {
    DoubleArray temporary{};
    DoubleArray output{};

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
                    coefficients[frequency_row]
                                [frequency_column];
            }

            temporary[row][frequency_column] = sum;
        }
    }

    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0;
             column < BLOCK_SIZE;
             ++column) {
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

void placeBlock(
    GrayImage& processed_image,
    const DoubleArray& level_shifted_block,
    int start_x,
    int start_y
) {
    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0;
             column < BLOCK_SIZE;
             ++column) {
            const long restored = lround(
                level_shifted_block[row][column] + 128.0
            );

            const uint8_t pixel = static_cast<uint8_t>(
                clamp(restored, 0L, 255L)
            );

            const size_t index =
                static_cast<size_t>(start_y + row) *
                static_cast<size_t>(processed_image.width) +
                static_cast<size_t>(start_x + column);

            processed_image.pixels[index] = pixel;
        }
    }
}

GrayImage cropImage(
    const GrayImage& processed_image,
    uint32_t original_width,
    uint32_t original_height
) {
    if (
        original_width >
            static_cast<uint32_t>(processed_image.width) ||
        original_height >
            static_cast<uint32_t>(processed_image.height)
    ) {
        throw runtime_error(
            "Original dimensions exceed processed dimensions"
        );
    }

    GrayImage output;
    output.width = static_cast<int>(original_width);
    output.height = static_cast<int>(original_height);
    output.pixels.resize(
        static_cast<size_t>(output.width) *
        static_cast<size_t>(output.height)
    );

    for (int row = 0; row < output.height; ++row) {
        const size_t source_offset =
            static_cast<size_t>(row) *
            static_cast<size_t>(processed_image.width);

        const size_t destination_offset =
            static_cast<size_t>(row) *
            static_cast<size_t>(output.width);

        copy_n(
            processed_image.pixels.begin() +
                static_cast<ptrdiff_t>(source_offset),
            output.width,
            output.pixels.begin() +
                static_cast<ptrdiff_t>(destination_offset)
        );
    }

    return output;
}

void savePGM(
    const fs::path& output_path,
    const GrayImage& image
) {
    ofstream output(output_path, ios::binary);

    if (!output) {
        throw runtime_error(
            "Could not create output image: " +
            output_path.string()
        );
    }

    output
        << "P5\n"
        << image.width
        << ' '
        << image.height
        << "\n255\n";

    output.write(
        reinterpret_cast<const char*>(
            image.pixels.data()
        ),
        static_cast<streamsize>(
            image.pixels.size()
        )
    );

    if (!output) {
        throw runtime_error(
            "Failed while writing output image: " +
            output_path.string()
        );
    }
}

DecompressionResult decompressImage(
    const fs::path& input_path,
    const fs::path& output_path,
    const DoubleArray& dct_matrix
) {
    const auto start_time =
        chrono::steady_clock::now();

    ifstream input(input_path, ios::binary);

    if (!input) {
        throw runtime_error(
            "Could not open compressed file: " +
            input_path.string()
        );
    }

    char magic[4]{};
    input.read(magic, 4);

    if (
        input.gcount() != 4 ||
        string(magic, 4) != "GSC2"
    ) {
        throw runtime_error(
            "File is not a GSC2 compressed image: " +
            input_path.string()
        );
    }

    CompressedImage header;
    header.original_width = readUInt32(input);
    header.original_height = readUInt32(input);
    header.processed_width = readUInt32(input);
    header.processed_height = readUInt32(input);
    header.block_count = readUInt32(input);

    if (
        header.original_width == 0 ||
        header.original_height == 0 ||
        header.processed_width == 0 ||
        header.processed_height == 0 ||
        header.processed_width % BLOCK_SIZE != 0 ||
        header.processed_height % BLOCK_SIZE != 0 ||
        header.original_width > header.processed_width ||
        header.original_height > header.processed_height ||
        header.processed_width >
            static_cast<uint32_t>(numeric_limits<int>::max()) ||
        header.processed_height >
            static_cast<uint32_t>(numeric_limits<int>::max())
    ) {
        throw runtime_error(
            "Invalid image dimensions in GSC2 header"
        );
    }

    const uint64_t expected_block_count =
        static_cast<uint64_t>(
            header.processed_width / BLOCK_SIZE
        ) *
        static_cast<uint64_t>(
            header.processed_height / BLOCK_SIZE
        );

    if (header.block_count != expected_block_count) {
        throw runtime_error(
            "GSC2 block count does not match image dimensions"
        );
    }

    for (auto& row : header.quantisation_matrix) {
        for (uint16_t& value : row) {
            value = readUInt16(input);

            if (value == 0) {
                throw runtime_error(
                    "Quantisation matrix contains zero"
                );
            }
        }
    }

    GrayImage processed_image;
    processed_image.width =
        static_cast<int>(header.processed_width);
    processed_image.height =
        static_cast<int>(header.processed_height);
    processed_image.pixels.resize(
        static_cast<size_t>(processed_image.width) *
        static_cast<size_t>(processed_image.height)
    );

    const int blocks_x =
        processed_image.width / BLOCK_SIZE;

    for (uint32_t block_index = 0;
         block_index < header.block_count;
         ++block_index) {
        const int16_t dc = readInt16(input);
        const uint16_t ac_bit_count =
            readUInt16(input);

        const size_t ac_byte_count =
            (static_cast<size_t>(ac_bit_count) + 7) / 8;

        vector<uint8_t> ac_data(ac_byte_count);

        if (ac_byte_count != 0) {
            input.read(
                reinterpret_cast<char*>(ac_data.data()),
                static_cast<streamsize>(ac_byte_count)
            );

            if (
                input.gcount() !=
                static_cast<streamsize>(ac_byte_count)
            ) {
                throw runtime_error(
                    "Compressed file ended inside an AC block"
                );
            }
        }

        const auto zigzag_coefficients =
            decodeBlock(
                dc,
                ac_data,
                ac_bit_count
            );

        const IntegerArray quantised =
            inverseZigzag(zigzag_coefficients);

        const DoubleArray dequantised =
            dequantise(
                quantised,
                header.quantisation_matrix
            );

        const DoubleArray spatial_block =
            performInverseDCT(
                dequantised,
                dct_matrix
            );

        const int block_x =
            static_cast<int>(block_index) % blocks_x;

        const int block_y =
            static_cast<int>(block_index) / blocks_x;

        placeBlock(
            processed_image,
            spatial_block,
            block_x * BLOCK_SIZE,
            block_y * BLOCK_SIZE
        );
    }

    if (input.peek() != EOF) {
        throw runtime_error(
            "Extra bytes found after the final GSC2 block"
        );
    }

    const GrayImage output_image =
        cropImage(
            processed_image,
            header.original_width,
            header.original_height
        );

    savePGM(output_path, output_image);

    const auto end_time =
        chrono::steady_clock::now();

    DecompressionResult result;
    result.compressed_size = fs::file_size(input_path);
    result.decompressed_size =
        static_cast<uintmax_t>(output_image.width) *
        static_cast<uintmax_t>(output_image.height);
    result.elapsed_ms =
        chrono::duration<double, milli>(
            end_time - start_time
        ).count();

    return result;
}

string lowercaseExtension(fs::path path) {
    string extension = path.extension().string();

    transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char character) {
            return static_cast<char>(
                tolower(character)
            );
        }
    );

    return extension;
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            cerr
                << "Usage:\n"
                << "  decompress <input_folder> "
                   "<output_folder>\n\n"
                << "Example:\n"
                << "  decompress output reconstructed\n";

            return 1;
        }

        const fs::path input_directory = argv[1];
        const fs::path output_directory = argv[2];

        if (
            !fs::exists(input_directory) ||
            !fs::is_directory(input_directory)
        ) {
            throw runtime_error(
                "Input folder does not exist: " +
                input_directory.string()
            );
        }

        fs::create_directories(output_directory);

        vector<fs::path> input_files;

        for (
            const fs::directory_entry& entry :
            fs::directory_iterator(input_directory)
        ) {
            if (
                entry.is_regular_file() &&
                lowercaseExtension(entry.path()) == ".gsc"
            ) {
                input_files.push_back(entry.path());
            }
        }

        sort(input_files.begin(), input_files.end());

        if (input_files.empty()) {
            throw runtime_error(
                "No .gsc files were found in: " +
                input_directory.string()
            );
        }

        const DoubleArray dct_matrix =
            createDCTMatrix();

        size_t image_count = 0;
        uintmax_t total_compressed_size = 0;
        uintmax_t total_decompressed_size = 0;
        double total_elapsed_ms = 0.0;

        cout << fixed << setprecision(3);

        for (const fs::path& input_path : input_files) {
            fs::path output_path =
                output_directory /
                input_path.stem();

            output_path += ".pgm";

            const DecompressionResult result =
                decompressImage(
                    input_path,
                    output_path,
                    dct_matrix
                );

            cout
                << input_path.filename().string()
                << ": "
                << result.compressed_size
                << " bytes -> "
                << result.decompressed_size
                << " pixels\n";

            total_compressed_size +=
                result.compressed_size;
            total_decompressed_size +=
                result.decompressed_size;
            total_elapsed_ms += result.elapsed_ms;
            ++image_count;
        }

        cout << "\nDecompression complete\n";
        cout
            << "Images decompressed: "
            << image_count
            << '\n';
        cout << "Output format: PGM P5\n";
        cout
            << "Total compressed size: "
            << total_compressed_size
            << " bytes\n";
        cout
            << "Total reconstructed pixel data: "
            << total_decompressed_size
            << " bytes\n";
        cout
            << "Total processing time: "
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

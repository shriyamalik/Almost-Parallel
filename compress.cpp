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

struct CompressedBlock {
    int16_t dc = 0;
    uint16_t ac_bit_count = 0;
    vector<uint8_t> ac_data;
};

struct CompressionResult {
    uintmax_t raw_size = 0;
    uintmax_t compressed_size = 0;
    double elapsed_ms = 0.0;
};

constexpr QuantisationMatrix QUANTISATION_MATRIX = {{
    {{16, 11, 10, 16, 24, 40, 51, 61}},
    {{12, 12, 14, 19, 26, 58, 60, 55}},
    {{14, 13, 16, 24, 40, 57, 69, 56}},
    {{14, 17, 22, 29, 51, 87, 80, 62}},
    {{18, 22, 37, 56, 68, 109, 103, 77}},
    {{24, 35, 55, 64, 81, 104, 113, 92}},
    {{49, 64, 78, 87, 103, 121, 120, 101}},
    {{72, 92, 95, 98, 112, 100, 103, 99}}
}};


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

class BitWriter {
private:
    vector<uint8_t> bytes;
    uint8_t current_byte = 0;
    uint8_t bits_in_current_byte = 0;
    uint32_t total_bits = 0;

public:
    void writeBits(uint32_t value, uint8_t bit_count) {
        if (bit_count > 32) {
            throw runtime_error("Cannot write more than 32 bits at once");
        }

        for (int bit = static_cast<int>(bit_count) - 1;
             bit >= 0;
             --bit) {
            const uint8_t next_bit =
                static_cast<uint8_t>((value >> bit) & 1U);

            current_byte = static_cast<uint8_t>(
                (current_byte << 1) | next_bit
            );

            ++bits_in_current_byte;
            ++total_bits;

            if (bits_in_current_byte == 8) {
                bytes.push_back(current_byte);
                current_byte = 0;
                bits_in_current_byte = 0;
            }
        }
    }

    void flush() {
        if (bits_in_current_byte != 0) {
            current_byte <<= static_cast<uint8_t>(
                8 - bits_in_current_byte
            );

            bytes.push_back(current_byte);
            current_byte = 0;
            bits_in_current_byte = 0;
        }
    }

    uint32_t bitCount() const {
        return total_bits;
    }

    vector<uint8_t> takeBytes() {
        return move(bytes);
    }
};

string readToken(istream& input) {
    while (true) {
        const int next = input.peek();

        if (next == EOF) {
            return {};
        }

        if (isspace(static_cast<unsigned char>(next))) {
            input.get();
            continue;
        }

        if (next == '#') {
            input.ignore(
                numeric_limits<streamsize>::max(),
                '\n'
            );
            continue;
        }

        break;
    }

    string token;

    while (true) {
        const int next = input.peek();

        if (
            next == EOF ||
            isspace(static_cast<unsigned char>(next)) ||
            next == '#'
        ) {
            break;
        }

        token.push_back(static_cast<char>(input.get()));
    }

    return token;
}

GrayImage loadImage(const fs::path& image_path) {
    ifstream input(image_path, ios::binary);

    if (!input) {
        throw runtime_error(
            "Could not open image: " + image_path.string()
        );
    }

    const string magic = readToken(input);
    const string width_text = readToken(input);
    const string height_text = readToken(input);
    const string maximum_text = readToken(input);

    if (magic != "P5") {
        throw runtime_error(
            "Not a binary PGM P5 image: " +
            image_path.string()
        );
    }

    if (
        width_text.empty() ||
        height_text.empty() ||
        maximum_text.empty()
    ) {
        throw runtime_error(
            "Incomplete PGM header: " + image_path.string()
        );
    }

    GrayImage image;
    image.width = stoi(width_text);
    image.height = stoi(height_text);

    const int maximum_value = stoi(maximum_text);

    if (
        image.width <= 0 ||
        image.height <= 0 ||
        image.width % BLOCK_SIZE != 0 ||
        image.height % BLOCK_SIZE != 0
    ) {
        throw runtime_error(
            "Image dimensions must be positive multiples of 8: " +
            image_path.string()
        );
    }

    if (maximum_value != 255) {
        throw runtime_error(
            "Only 8-bit PGM images with maximum value 255 are supported: " +
            image_path.string()
        );
    }

    char separator = '\0';

    if (
        !input.get(separator) ||
        !isspace(static_cast<unsigned char>(separator))
    ) {
        throw runtime_error(
            "Invalid PGM header separator: " +
            image_path.string()
        );
    }

    if (separator == '\r' && input.peek() == '\n') {
        input.get();
    }

    const size_t pixel_count =
        static_cast<size_t>(image.width) *
        static_cast<size_t>(image.height);

    image.pixels.resize(pixel_count);

    input.read(
        reinterpret_cast<char*>(image.pixels.data()),
        static_cast<streamsize>(pixel_count)
    );

    if (
        input.gcount() !=
        static_cast<streamsize>(pixel_count)
    ) {
        throw runtime_error(
            "PGM file does not contain enough pixel data: " +
            image_path.string()
        );
    }

    return image;
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

DoubleArray levelShift(
    const GrayImage& image,
    int start_x,
    int start_y
) {
    DoubleArray block{};

    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0;
             column < BLOCK_SIZE;
             ++column) {
            const int image_x = start_x + column;
            const int image_y = start_y + row;

            const size_t index =
                static_cast<size_t>(image_y) *
                static_cast<size_t>(image.width) +
                static_cast<size_t>(image_x);

            block[row][column] =
                static_cast<double>(image.pixels[index]) -
                128.0;
        }
    }

    return block;
}

DoubleArray performDCT(
    const DoubleArray& input,
    const DoubleArray& dct_matrix
) {
    DoubleArray temporary{};
    DoubleArray output{};

    for (int frequency_row = 0;
         frequency_row < BLOCK_SIZE;
         ++frequency_row) {
        for (int column = 0;
             column < BLOCK_SIZE;
             ++column) {
            double sum = 0.0;

            for (int row = 0;
                 row < BLOCK_SIZE;
                 ++row) {
                sum +=
                    dct_matrix[frequency_row][row] *
                    input[row][column];
            }

            temporary[frequency_row][column] = sum;
        }
    }

    for (int frequency_row = 0;
         frequency_row < BLOCK_SIZE;
         ++frequency_row) {
        for (int frequency_column = 0;
             frequency_column < BLOCK_SIZE;
             ++frequency_column) {
            double sum = 0.0;

            for (int column = 0;
                 column < BLOCK_SIZE;
                 ++column) {
                sum +=
                    temporary[frequency_row][column] *
                    dct_matrix[frequency_column][column];
            }

            output[frequency_row][frequency_column] = sum;
        }
    }

    return output;
}

IntegerArray quantise(
    const DoubleArray& dct,
    const QuantisationMatrix& quantisation_matrix
) {
    IntegerArray output{};

    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int column = 0;
             column < BLOCK_SIZE;
             ++column) {
            const double divided =
                dct[row][column] /
                static_cast<double>(
                    quantisation_matrix[row][column]
                );

            long rounded = lround(divided);

            rounded = clamp(
                rounded,
                static_cast<long>(
                    numeric_limits<int16_t>::min()
                ),
                static_cast<long>(
                    numeric_limits<int16_t>::max()
                )
            );

            output[row][column] =
                static_cast<int16_t>(rounded);
        }
    }

    return output;
}

array<int16_t, BLOCK_ELEMENTS> zigzagScan(
    const IntegerArray& block
) {
    array<int16_t, BLOCK_ELEMENTS> output{};

    for (int output_index = 0;
         output_index < BLOCK_ELEMENTS;
         ++output_index) {
        const int matrix_index =
            ZIGZAG_ORDER[output_index];

        const int row = matrix_index / BLOCK_SIZE;
        const int column = matrix_index % BLOCK_SIZE;

        output[output_index] = block[row][column];
    }

    return output;
}

uint8_t coefficientSize(int16_t value) {
    if (value == 0) {
        return 0;
    }

    uint32_t magnitude =
        value < 0
            ? static_cast<uint32_t>(
                -static_cast<int32_t>(value)
            )
            : static_cast<uint32_t>(value);

    uint8_t size = 0;

    while (magnitude != 0) {
        ++size;
        magnitude >>= 1;
    }

    return size;
}

uint32_t encodeAmplitude(
    int16_t value,
    uint8_t size
) {
    if (value >= 0) {
        return static_cast<uint32_t>(value);
    }

    const uint32_t mask =
        (1U << size) - 1U;

    return static_cast<uint32_t>(
        static_cast<int32_t>(value) +
        static_cast<int32_t>(mask)
    );
}

CompressedBlock encodeBlock(
    const array<int16_t, BLOCK_ELEMENTS>& coefficients
) {
    CompressedBlock block;
    block.dc = coefficients[0];

    BitWriter writer;
    uint8_t zero_run = 0;

    for (int index = 1;
         index < BLOCK_ELEMENTS;
         ++index) {
        const int16_t value = coefficients[index];

        if (value == 0) {
            ++zero_run;
            continue;
        }

        while (zero_run >= 16) {
            writer.writeBits(0xF0, 8);
            zero_run -= 16;
        }

        const uint8_t size = coefficientSize(value);

        if (size > 15) {
            throw runtime_error(
                "Quantised AC coefficient requires more than 15 bits"
            );
        }

        const uint8_t run_size =
            static_cast<uint8_t>(
                (zero_run << 4) | size
            );

        writer.writeBits(run_size, 8);
        writer.writeBits(
            encodeAmplitude(value, size),
            size
        );

        zero_run = 0;
    }

    if (zero_run > 0) {
        writer.writeBits(0x00, 8);
    }

    writer.flush();

    if (
        writer.bitCount() >
        numeric_limits<uint16_t>::max()
    ) {
        throw runtime_error(
            "Encoded AC block is too large"
        );
    }

    block.ac_bit_count =
        static_cast<uint16_t>(writer.bitCount());

    block.ac_data = writer.takeBytes();

    return block;
}

void writeUInt8(ostream& output, uint8_t value) {
    output.put(static_cast<char>(value));
}

void writeUInt16(ostream& output, uint16_t value) {
    writeUInt8(
        output,
        static_cast<uint8_t>(value & 0xFF)
    );

    writeUInt8(
        output,
        static_cast<uint8_t>((value >> 8) & 0xFF)
    );
}

void writeInt16(ostream& output, int16_t value) {
    writeUInt16(
        output,
        static_cast<uint16_t>(value)
    );
}

void writeUInt32(ostream& output, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        writeUInt8(
            output,
            static_cast<uint8_t>(
                (value >> shift) & 0xFF
            )
        );
    }
}

void saveCompressedImage(
    const fs::path& output_path,
    const GrayImage& image,
    const QuantisationMatrix& quantisation_matrix,
    const vector<CompressedBlock>& blocks
) {
    ofstream output(output_path, ios::binary);

    if (!output) {
        throw runtime_error(
            "Could not create output file: " +
            output_path.string()
        );
    }

    /*
    GSC2 block layout:
      int16_t  DC
      uint16_t number of valid AC bits
      uint8_t  packed AC bytes[ceil(ac_bit_count / 8)]
    */
    output.write("GSC2", 4);

    writeUInt32(
        output,
        static_cast<uint32_t>(image.width)
    );

    writeUInt32(
        output,
        static_cast<uint32_t>(image.height)
    );

    writeUInt32(
        output,
        static_cast<uint32_t>(image.width)
    );

    writeUInt32(
        output,
        static_cast<uint32_t>(image.height)
    );

    writeUInt32(
        output,
        static_cast<uint32_t>(blocks.size())
    );

    for (const auto& row : quantisation_matrix) {
        for (uint16_t value : row) {
            writeUInt16(output, value);
        }
    }

    for (const CompressedBlock& block : blocks) {
        writeInt16(output, block.dc);
        writeUInt16(output, block.ac_bit_count);

        output.write(
            reinterpret_cast<const char*>(
                block.ac_data.data()
            ),
            static_cast<streamsize>(
                block.ac_data.size()
            )
        );
    }

    if (!output) {
        throw runtime_error(
            "Failed while writing output file: " +
            output_path.string()
        );
    }
}

CompressionResult compressImage(
    const fs::path& input_path,
    const fs::path& output_path,
    const DoubleArray& dct_matrix
) {
    const auto start_time =
        chrono::steady_clock::now();

    const GrayImage image = loadImage(input_path);

    const int blocks_x =
        image.width / BLOCK_SIZE;

    const int blocks_y =
        image.height / BLOCK_SIZE;

    vector<CompressedBlock> compressed_blocks;

    compressed_blocks.reserve(
        static_cast<size_t>(blocks_x) *
        static_cast<size_t>(blocks_y)
    );

    for (int block_y = 0;
         block_y < blocks_y;
         ++block_y) {
        for (int block_x = 0;
             block_x < blocks_x;
             ++block_x) {
            const DoubleArray pixels =
                levelShift(
                    image,
                    block_x * BLOCK_SIZE,
                    block_y * BLOCK_SIZE
                );

            const DoubleArray dct =
                performDCT(pixels, dct_matrix);

            const IntegerArray quantised =
                quantise(
                    dct,
                    QUANTISATION_MATRIX
                );

            const auto zigzag =
                zigzagScan(quantised);

            compressed_blocks.push_back(
                encodeBlock(zigzag)
            );
        }
    }

    saveCompressedImage(
        output_path,
        image,
        QUANTISATION_MATRIX,
        compressed_blocks
    );

    const auto end_time =
        chrono::steady_clock::now();

    CompressionResult result;

    result.raw_size =
        static_cast<uintmax_t>(image.width) *
        static_cast<uintmax_t>(image.height);

    result.compressed_size =
        fs::file_size(output_path);

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
                << "  compress <input_folder> "
                   "<output_folder>\n\n"
                << "Example:\n"
                << "  compress img output\n";

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

        vector<fs::path> input_images;

        for (
            const fs::directory_entry& entry :
            fs::directory_iterator(input_directory)
        ) {
            if (
                entry.is_regular_file() &&
                lowercaseExtension(entry.path()) == ".pgm"
            ) {
                input_images.push_back(entry.path());
            }
        }

        sort(
            input_images.begin(),
            input_images.end()
        );

        if (input_images.empty()) {
            throw runtime_error(
                "No .pgm images were found in: " +
                input_directory.string()
            );
        }

        const DoubleArray dct_matrix =
            createDCTMatrix();

        size_t image_count = 0;
        uintmax_t total_raw_size = 0;
        uintmax_t total_compressed_size = 0;
        double total_elapsed_ms = 0.0;

        cout << fixed << setprecision(3);

        for (
            const fs::path& input_path :
            input_images
        ) {
            fs::path output_path =
                output_directory /
                input_path.stem();

            output_path += ".gsc";

            const CompressionResult result =
                compressImage(
                    input_path,
                    output_path,
                    dct_matrix
                );

            cout
                << input_path.filename().string()
                << ": "
                << result.raw_size
                << " -> "
                << result.compressed_size
                << " bytes, ratio "
                << (
                    result.compressed_size == 0
                        ? 0.0
                        : static_cast<double>(
                            result.raw_size
                        ) /
                        static_cast<double>(
                            result.compressed_size
                        )
                )
                << ":1\n";

            total_raw_size += result.raw_size;
            total_compressed_size +=
                result.compressed_size;

            total_elapsed_ms += result.elapsed_ms;
            ++image_count;
        }

        const double overall_ratio =
            total_compressed_size == 0
                ? 0.0
                : static_cast<double>(
                    total_raw_size
                ) /
                static_cast<double>(
                    total_compressed_size
                );

        cout << "\nCompression complete\n";
        cout
            << "Images compressed: "
            << image_count
            << '\n';

        cout << "Input format: PGM P5\n";

        cout
            << "Total raw image size: "
            << total_raw_size
            << " bytes\n";

        cout
            << "Total compressed size: "
            << total_compressed_size
            << " bytes\n";

        cout
            << "Overall compression ratio: "
            << overall_ratio
            << ":1\n";

        cout
            << "Total processing time: "
            << total_elapsed_ms
            << " ms\n";

        return 0;
    }
    catch (const exception& error) {
        cerr
            << "Compression failed: "
            << error.what()
            << '\n';

        return 1;
    }
}

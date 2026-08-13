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

using ProfileClock = chrono::steady_clock;

double toMilliseconds(ProfileClock::duration duration) {
    return chrono::duration<double, milli>(duration).count();
}

constexpr int IMAGE_WIDTH = 512;
constexpr int IMAGE_HEIGHT = 512;
constexpr int BLOCK_SIZE = 8;
constexpr int BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE;
constexpr double PI = 3.14159265358979323846;

using DoubleArray = array<array<double, BLOCK_SIZE>, BLOCK_SIZE>;
using IntergerArray = array<array<int16_t, BLOCK_SIZE>, BLOCK_SIZE>;
using qMatrix = array<array<uint16_t, BLOCK_SIZE>, BLOCK_SIZE>;

struct Grayimage {
    int width = 0;
    int height = 0;
    vector<uint8_t> pixels;
};

struct RLEPair {
    uint8_t zero_run = 0;
    int16_t value = 0;
};

struct CompressedBlock {
    int16_t dc = 0;
    vector<RLEPair> pairs;
};

struct StageProfile {
    double load_image_ms = 0.0;
    double level_shift_ms = 0.0;
    double dct_ms = 0.0;
    double quantisation_ms = 0.0;
    double zigzag_ms = 0.0;
    double encoding_ms = 0.0;
    double save_file_ms = 0.0;

    size_t block_count = 0;

    double profiledTotalMs() const {
        return
            load_image_ms +
            level_shift_ms +
            dct_ms +
            quantisation_ms +
            zigzag_ms +
            encoding_ms +
            save_file_ms;
    }
};

void addStageProfile(
    StageProfile& total,
    const StageProfile& current
) {
    total.load_image_ms += current.load_image_ms;
    total.level_shift_ms += current.level_shift_ms;
    total.dct_ms += current.dct_ms;
    total.quantisation_ms += current.quantisation_ms;
    total.zigzag_ms += current.zigzag_ms;
    total.encoding_ms += current.encoding_ms;
    total.save_file_ms += current.save_file_ms;
    total.block_count += current.block_count;
}

struct CompressionResult {
    uintmax_t raw_size = 0;
    uintmax_t compressed_size = 0;
    double elapsed_ms = 0.0;
    StageProfile profile;
};

constexpr qMatrix QUANTISATION_MATRIX = {{
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

string readImage(istream& input) {
    while (true) {
        const int next = input.peek();

        if (next == EOF) {
            break;
        }

        if (isspace((unsigned char)(next))) {
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

    string bits;

    while (true) {
        const int next = input.peek();

        if (next == EOF || isspace((unsigned char)(next)) || next == '#') break;
        bits.push_back((char)(input.get()));
    }
    return bits;
}

Grayimage loadImg(const fs::path& img_path) {
    ifstream input(img_path, ios::binary);

    Grayimage img;
    readImage(input);
    img.width = stoi(readImage(input));
    img.height = stoi(readImage(input));

    const int maximum_value = stoi(readImage(input));

    char separator = '\0';

    if (
        !input.get(separator) ||
        !isspace(static_cast<unsigned char>(separator))
    ) {
        throw runtime_error(
            "Invalid PGM header separator: " +
            img_path.string()
        );
    }

    if (separator == '\r' && input.peek() == '\n') {
        input.get();
    }

    const size_t pixel_count = (size_t)(img.width) * (size_t)(img.height);

    img.pixels.resize(pixel_count);

    input.read(
        reinterpret_cast<char*>(img.pixels.data()),
        (streamsize)(pixel_count)
    );

    return img;
}

DoubleArray createDCT() {
    DoubleArray dtc{};

    for (int f = 0; f < BLOCK_SIZE; ++f) {
        const double normalisation =
            f == 0
                ? sqrt(1.0 / BLOCK_SIZE)
                : sqrt(2.0 / BLOCK_SIZE);

        for (int position = 0; position < BLOCK_SIZE; ++position) {
            dtc[f][position] = normalisation *
                cos(((2.0 * position + 1.0) *
                    f * PI) / (2.0 * BLOCK_SIZE));
        }
    }
    return dtc;
}

DoubleArray LevelShift(
    const Grayimage& img,
    int start_x,
    int start_y
) {
    DoubleArray block{};

    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int col = 0; col < BLOCK_SIZE; ++col) {
            const int img_x = start_x + col;
            const int img_y = start_y + row;

            const size_t index =
                (size_t)(img_y) *
                (size_t)(img.width) +
                (size_t)(img_x);

            block[row][col] =
                (double)(img.pixels[index]) - 128.0;
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

    for (int f_row = 0; f_row < BLOCK_SIZE; ++f_row) {
        for (int col = 0; col < BLOCK_SIZE; ++col) {
            double sum = 0.0;

            for (int row = 0; row < BLOCK_SIZE; ++row) {
                sum += dct_matrix[f_row][row] * input[row][col];
            }
            temporary[f_row][col] = sum;
        }
    }

    for (int f_row = 0;
         f_row < BLOCK_SIZE;
         ++f_row) {

        for (int f_col = 0;
             f_col < BLOCK_SIZE;
             ++f_col) {

            double sum = 0.0;

            for (int col = 0; col < BLOCK_SIZE; ++col) {
                sum +=
                    temporary[f_row][col] *
                    dct_matrix[f_col][col];
            }

            output[f_row][f_col] = sum;
        }
    }

    return output;
}

IntergerArray quantise(
    const DoubleArray& dct,
    const qMatrix& quantisation_matrix
) {
    IntergerArray output{};

    for (int row = 0; row < BLOCK_SIZE; ++row) {
        for (int col = 0; col < BLOCK_SIZE; ++col) {
            const double divided =
                dct[row][col] / (double)(quantisation_matrix[row][col]);

            long rounded = lround(divided);

            rounded = clamp(
                rounded,
                (long)(
                    numeric_limits<int16_t>::min()
                ),
                (long)(
                    numeric_limits<int16_t>::max()
                )
            );

            output[row][col] =
                (int16_t)(rounded);
        }
    }

    return output;
}

array<int16_t, BLOCK_ELEMENTS> zigzagScan(
    const IntergerArray& block
) {
    array<int16_t, BLOCK_ELEMENTS> output{};

    for (int output_index = 0;
         output_index < BLOCK_ELEMENTS;
         ++output_index) {

        const int matrix_index = ZIGZAG_ORDER[output_index];
        const int row = matrix_index / BLOCK_SIZE;
        const int col = matrix_index % BLOCK_SIZE;

        output[output_index] = block[row][col];
    }

    return output;
}

CompressedBlock runLengthEncode(const array<int16_t, BLOCK_ELEMENTS>& coefficients) {
    CompressedBlock block;
    block.dc = coefficients[0];
    block.pairs.reserve(BLOCK_ELEMENTS - 1);

    uint8_t zero_run = 0;

    for (int index = 1; index < BLOCK_ELEMENTS; ++index) {
        const int16_t value = coefficients[index];

        if (value == 0) {
            ++zero_run;
            continue;
        }

        block.pairs.push_back({zero_run, value});
        zero_run = 0;
    }

    return block;
}

void writeUInt8(ostream& output, uint8_t value) {
    output.put((char)(value));
}

void writeUInt16(ostream& output, uint16_t value) {
    writeUInt8(output, (uint8_t)(value & 0xFF));

    writeUInt8(output, (uint8_t)((value >> 8) & 0xFF));
}

void writeInt16(ostream& output, int16_t value) {
    writeUInt16(output, (uint16_t)(value));
}

void writeUInt32(ostream& output, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        writeUInt8(output, (uint8_t)((value >> shift) & 0xFF));
    }
}

void saveCompressedimage(
    const fs::path& output_path,
    const Grayimage& img,
    const qMatrix& quantisation_matrix,
    const vector<CompressedBlock>& blocks
) {
    ofstream output(output_path, ios::binary);
    output.write("GSC1", 4);

    writeUInt32(output, (uint32_t)(img.width));
    writeUInt32(output, (uint32_t)(img.height));
    writeUInt32(output, (uint32_t)(img.width));
    writeUInt32(output, (uint32_t)(img.height));
    writeUInt32(output, (uint32_t)(blocks.size()));

    for (const auto& row : quantisation_matrix) {
        for (uint16_t value : row) {
            writeUInt16(output, value);
        }
    }

    for (const CompressedBlock& block : blocks) {
        writeInt16(output, block.dc);
        writeUInt8(output, (uint8_t)(block.pairs.size()));

        for (const RLEPair& pair : block.pairs) {
            writeUInt8(output, pair.zero_run);
            writeInt16(output, pair.value);
        }
    }
}

CompressionResult compressimg(
    const fs::path& input_path,
    const fs::path& output_path,
    const DoubleArray& dct_matrix
) {
    CompressionResult result;

    const auto total_start = ProfileClock::now();

    auto stage_start = ProfileClock::now();
    const Grayimage img = loadImg(input_path);
    result.profile.load_image_ms += toMilliseconds(
        ProfileClock::now() - stage_start
    );

    const int blocks_x = img.width / BLOCK_SIZE;
    const int blocks_y = img.height / BLOCK_SIZE;

    vector<CompressedBlock> compressed_blocks;
    compressed_blocks.reserve((size_t)(blocks_x) * (size_t)(blocks_y));

    for (int block_y = 0; block_y < blocks_y; ++block_y) {
        for (int block_x = 0; block_x < blocks_x; ++block_x) {
            stage_start = ProfileClock::now();
            const DoubleArray pixels = LevelShift(
                img,
                block_x * BLOCK_SIZE,
                block_y * BLOCK_SIZE
            );
            result.profile.level_shift_ms += toMilliseconds(
                ProfileClock::now() - stage_start
            );

            stage_start = ProfileClock::now();
            const DoubleArray dct = performDCT(pixels, dct_matrix);
            result.profile.dct_ms += toMilliseconds(
                ProfileClock::now() - stage_start
            );

            stage_start = ProfileClock::now();
            const IntergerArray quantised = quantise(
                dct,
                QUANTISATION_MATRIX
            );
            result.profile.quantisation_ms += toMilliseconds(
                ProfileClock::now() - stage_start
            );

            stage_start = ProfileClock::now();
            const auto zigzag = zigzagScan(quantised);
            result.profile.zigzag_ms += toMilliseconds(
                ProfileClock::now() - stage_start
            );

            stage_start = ProfileClock::now();
            compressed_blocks.push_back(runLengthEncode(zigzag));
            result.profile.encoding_ms += toMilliseconds(
                ProfileClock::now() - stage_start
            );

            ++result.profile.block_count;
        }
    }

    stage_start = ProfileClock::now();
    saveCompressedimage(
        output_path,
        img,
        QUANTISATION_MATRIX,
        compressed_blocks
    );
    result.profile.save_file_ms += toMilliseconds(
        ProfileClock::now() - stage_start
    );

    result.raw_size = (uintmax_t)(img.width) *
        (uintmax_t)(img.height);

    result.compressed_size = fs::file_size(output_path);

    const auto total_end = ProfileClock::now();
    result.elapsed_ms = toMilliseconds(total_end - total_start);

    return result;
}

void printAverageStageProfile(
    const StageProfile& total_profile,
    double total_elapsed_ms,
    size_t image_count
) {
    if (image_count == 0) {
        return;
    }

    const double images = static_cast<double>(image_count);

    const double average_load_image_ms =
        total_profile.load_image_ms / images;
    const double average_level_shift_ms =
        total_profile.level_shift_ms / images;
    const double average_dct_ms =
        total_profile.dct_ms / images;
    const double average_quantisation_ms =
        total_profile.quantisation_ms / images;
    const double average_zigzag_ms =
        total_profile.zigzag_ms / images;
    const double average_encoding_ms =
        total_profile.encoding_ms / images;
    const double average_save_file_ms =
        total_profile.save_file_ms / images;
    const double average_total_ms =
        total_elapsed_ms / images;

    const double profiled_total_ms =
        total_profile.profiledTotalMs();

    const double total_other_ms = max(
        0.0,
        total_elapsed_ms - profiled_total_ms
    );

    const double average_other_ms =
        total_other_ms / images;

    auto percentage =
        [total_elapsed_ms](double stage_total_ms) {
            if (total_elapsed_ms == 0.0) {
                return 0.0;
            }

            return stage_total_ms / total_elapsed_ms * 100.0;
        };

    auto microsecondsPerBlock =
        [&total_profile](double stage_total_ms) {
            if (total_profile.block_count == 0) {
                return 0.0;
            }

            return stage_total_ms * 1000.0 /
                static_cast<double>(total_profile.block_count);
        };

    cout << "\nAverage profiling results\n";
    cout << "  Images processed:       "
         << image_count << '\n';
    cout << "  Total blocks processed: "
         << total_profile.block_count << '\n';

    cout << "  Image loading:          "
         << average_load_image_ms
         << " ms/image ("
         << percentage(total_profile.load_image_ms)
         << "%)\n";

    cout << "  Level shifting:         "
         << average_level_shift_ms
         << " ms/image, "
         << microsecondsPerBlock(total_profile.level_shift_ms)
         << " us/block ("
         << percentage(total_profile.level_shift_ms)
         << "%)\n";

    cout << "  2D DCT:                 "
         << average_dct_ms
         << " ms/image, "
         << microsecondsPerBlock(total_profile.dct_ms)
         << " us/block ("
         << percentage(total_profile.dct_ms)
         << "%)\n";

    cout << "  Quantisation:           "
         << average_quantisation_ms
         << " ms/image, "
         << microsecondsPerBlock(total_profile.quantisation_ms)
         << " us/block ("
         << percentage(total_profile.quantisation_ms)
         << "%)\n";

    cout << "  Zigzag scan:            "
         << average_zigzag_ms
         << " ms/image, "
         << microsecondsPerBlock(total_profile.zigzag_ms)
         << " us/block ("
         << percentage(total_profile.zigzag_ms)
         << "%)\n";

    cout << "  RLE encoding:           "
         << average_encoding_ms
         << " ms/image, "
         << microsecondsPerBlock(total_profile.encoding_ms)
         << " us/block ("
         << percentage(total_profile.encoding_ms)
         << "%)\n";

    cout << "  File writing:           "
         << average_save_file_ms
         << " ms/image ("
         << percentage(total_profile.save_file_ms)
         << "%)\n";

    cout << "  Other overhead:         "
         << average_other_ms
         << " ms/image ("
         << percentage(total_other_ms)
         << "%)\n";

    cout << "  Average total time:     "
         << average_total_ms
         << " ms/image\n";

    cout << "  Batch processing time:  "
         << total_elapsed_ms
         << " ms\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 3 || argc > 4) {
            cerr
                << "Usage:\n"
                << "  compress_imgs <input_folder> "
                   "<output_folder> [quantisation_scale]\n\n"
                << "Example:\n"
                << "  compress_imgs dataset_pgm "
                   "compressed_output 1.0\n";

            return 1;
        }

        const fs::path input_directory = argv[1];
        const fs::path output_directory = argv[2];


        fs::create_directories(output_directory);

        vector<fs::path> input_imgs;

        for (const fs::directory_entry& entry : fs::directory_iterator(input_directory)) {
            if (entry.is_regular_file()) {
                input_imgs.push_back(entry.path());
            }
        }

        sort(input_imgs.begin(), input_imgs.end());

        const DoubleArray dct_matrix = createDCT();

        size_t img_count = 0;
        uintmax_t total_raw_size = 0;
        uintmax_t total_compressed_size = 0;
        double total_elapsed_ms = 0.0;

        StageProfile total_profile;

        cout << fixed << setprecision(3);

        for (const fs::path& input_path : input_imgs) {
            fs::path output_path = output_directory / input_path.stem();

            output_path += ".gsc";

            const CompressionResult result =
                compressimg(input_path, output_path, dct_matrix);

            const double compression_ratio =
                result.compressed_size == 0
                    ? 0.0
                    : (double)(result.raw_size) /
                      (double)(
                          result.compressed_size
                      );


            total_raw_size += result.raw_size;
            total_compressed_size += result.compressed_size;
            total_elapsed_ms += result.elapsed_ms;
            addStageProfile(total_profile, result.profile);
            ++img_count;
        }

        const double overall_ratio = total_compressed_size == 0
                ? 0.0
                : (double)(total_raw_size) /
                  (double)(
                      total_compressed_size
                  );

        cout << "\nCompression complete\n";
        cout << "imgs compressed: " << img_count << '\n';
        cout << "Input format: PGM P5\n";
        cout << "Processed dimensions: 512 x 512 grayscale\n";
        cout << "Blocks per img: 4096\n";
        cout << "Total raw img size: "
             << total_raw_size << " bytes\n";
        cout << "Total compressed size: "
             << total_compressed_size << " bytes\n";
        cout << "Overall compression ratio: "
             << overall_ratio << ":1\n";

        printAverageStageProfile(
            total_profile,
            total_elapsed_ms,
            img_count
        );

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
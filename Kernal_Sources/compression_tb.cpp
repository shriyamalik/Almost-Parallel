#include "compression_kernel.hpp"

#include <algorithm>
#include <cctype>
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

namespace {

constexpr uint16_t QUANTISATION_MATRIX[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

constexpr const char* IMAGE_NAMES[] = {
    "t005.pgm", "t007.pgm", "t012.pgm",
    "t019.pgm", "t022.pgm", "t028.pgm",
    "t030.pgm", "t042.pgm", "t045.pgm",
    "t058.pgm", "t067.pgm", "t080.pgm",
    "t085.pgm", "t094.pgm", "t096.pgm",
    "t102.pgm", "t106.pgm", "t111.pgm",
    "t117.pgm", "t120.pgm", "t126.pgm",
    "t129.pgm", "t132.pgm", "t135.pgm",
    "t140.pgm", "t146.pgm", "t149.pgm",
    "t158.pgm", "t164.pgm", "t168.pgm"
};

struct GrayImage {
    int width = 0;
    int height = 0;
    vector<uint8_t> pixels;
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

GrayImage loadPgm(const fs::path& path) {
    ifstream input(path, ios::binary);

    if (!input) {
        throw runtime_error(
            "Could not open " + path.string()
        );
    }

    const string magic = readToken(input);
    const string width_text = readToken(input);
    const string height_text = readToken(input);
    const string maximum_text = readToken(input);

    if (
        magic.empty() ||
        width_text.empty() ||
        height_text.empty() ||
        maximum_text.empty()
    ) {
        throw runtime_error(
            "Incomplete PGM header: " + path.string()
        );
    }

    if (magic != "P5") {
        throw runtime_error(
            "Expected PGM P5: " + path.string()
        );
    }

    GrayImage image;
    image.width = stoi(width_text);
    image.height = stoi(height_text);

    const int maximum_value = stoi(maximum_text);

    if (
        image.width != IMAGE_WIDTH ||
        image.height != IMAGE_HEIGHT ||
        maximum_value != 255
    ) {
        throw runtime_error(
            "Kernel requires 512x512 8-bit PGM: " +
            path.string()
        );
    }

    char separator = '\0';

    if (
        !input.get(separator) ||
        !isspace(static_cast<unsigned char>(separator))
    ) {
        throw runtime_error(
            "Invalid PGM separator: " + path.string()
        );
    }

    if (separator == '\r' && input.peek() == '\n') {
        input.get();
    }

    image.pixels.resize(IMAGE_PIXELS);

    input.read(
        reinterpret_cast<char*>(image.pixels.data()),
        static_cast<streamsize>(IMAGE_PIXELS)
    );

    if (
        input.gcount() !=
        static_cast<streamsize>(IMAGE_PIXELS)
    ) {
        throw runtime_error(
            "Incomplete pixel data: " + path.string()
        );
    }

    return image;
}

void writeUInt8(ostream& output, uint8_t value) {
    output.put(static_cast<char>(value));
}

void writeUInt16(ostream& output, uint16_t value) {
    writeUInt8(
        output,
        static_cast<uint8_t>(value & 0xFFU)
    );

    writeUInt8(
        output,
        static_cast<uint8_t>((value >> 8) & 0xFFU)
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
                (value >> shift) & 0xFFU
            )
        );
    }
}

void saveGsc2(
    const fs::path& path,
    const vector<int16_t>& dc_values,
    const vector<uint16_t>& ac_bit_counts,
    const vector<uint8_t>& ac_data
) {
    ofstream output(path, ios::binary);

    if (!output) {
        throw runtime_error(
            "Could not create " + path.string()
        );
    }

    output.write("GSC2", 4);

    writeUInt32(output, IMAGE_WIDTH);
    writeUInt32(output, IMAGE_HEIGHT);
    writeUInt32(output, IMAGE_WIDTH);
    writeUInt32(output, IMAGE_HEIGHT);
    writeUInt32(output, NUM_BLOCKS);

    for (const auto& row : QUANTISATION_MATRIX) {
        for (uint16_t value : row) {
            writeUInt16(output, value);
        }
    }

    for (int block = 0; block < NUM_BLOCKS; ++block) {
        writeInt16(output, dc_values[block]);
        writeUInt16(output, ac_bit_counts[block]);

        const int byte_count =
            (ac_bit_counts[block] + 7) / 8;

        const int base =
            block * MAX_AC_BYTES_PER_BLOCK;

        output.write(
            reinterpret_cast<const char*>(
                ac_data.data() + base
            ),
            static_cast<streamsize>(byte_count)
        );
    }

    if (!output) {
        throw runtime_error(
            "Failed while writing " + path.string()
        );
    }
}

} // namespace

int main() {
    try {
        const fs::path input_directory =
            "/home/krishna/Projects/hls_compression_package/images";

        const fs::path output_directory =
            "/home/krishna/Projects/hls_compression_package/tb_output";

        cout
            << "Input directory: "
            << input_directory
            << '\n';

        cout
            << "Output directory: "
            << output_directory
            << '\n';

        fs::create_directories(output_directory);

        ofstream csv(output_directory / "results.csv");

        if (!csv) {
            throw runtime_error(
                "Could not create results.csv"
            );
        }

        csv
            << "image,raw_bytes,compressed_bytes,ratio,status\n";

        vector<int16_t> dc_values(NUM_BLOCKS);
        vector<uint16_t> ac_bit_counts(NUM_BLOCKS);
        vector<uint8_t> ac_data(TOTAL_AC_STORAGE_BYTES);

        uint64_t total_raw_bytes = 0;
        uint64_t total_compressed_bytes = 0;
        int failures = 0;

        cout << fixed << setprecision(3);

        for (const char* image_name : IMAGE_NAMES) {
            try {
                const fs::path input_path =
                    input_directory / image_name;

                const GrayImage image =
                    loadPgm(input_path);

                fill(
                    dc_values.begin(),
                    dc_values.end(),
                    0
                );

                fill(
                    ac_bit_counts.begin(),
                    ac_bit_counts.end(),
                    0
                );

                fill(
                    ac_data.begin(),
                    ac_data.end(),
                    0
                );

                uint32_t compressed_size = 0;
                uint32_t kernel_status = 0;

                compression_kernel(
                    image.pixels.data(),
                    dc_values.data(),
                    ac_bit_counts.data(),
                    ac_data.data(),
                    &compressed_size,
                    &kernel_status
                );

                if (kernel_status != 0) {
                    cerr
                        << image_name
                        << ": kernel status = "
                        << kernel_status
                        << '\n';

                    csv
                        << image_name
                        << ','
                        << IMAGE_PIXELS
                        << ",0,0,"
                        << kernel_status
                        << '\n';

                    ++failures;
                    continue;
                }

                const fs::path output_path =
                    output_directory /
                    (
                        fs::path(image_name)
                            .stem()
                            .string() +
                        "_kernel.gsc"
                    );

                saveGsc2(
                    output_path,
                    dc_values,
                    ac_bit_counts,
                    ac_data
                );

                const uintmax_t written_size =
                    fs::file_size(output_path);

                const double ratio =
                    written_size == 0
                        ? 0.0
                        : static_cast<double>(
                            IMAGE_PIXELS
                        ) /
                        static_cast<double>(
                            written_size
                        );

                cout
                    << setw(10)
                    << image_name
                    << " | compressed "
                    << setw(7)
                    << written_size
                    << " bytes | ratio "
                    << setw(7)
                    << ratio
                    << ":1 | status "
                    << kernel_status
                    << '\n';

                csv
                    << image_name
                    << ','
                    << IMAGE_PIXELS
                    << ','
                    << written_size
                    << ','
                    << ratio
                    << ','
                    << kernel_status
                    << '\n';

                total_raw_bytes += IMAGE_PIXELS;
                total_compressed_bytes +=
                    written_size;
            }
            catch (const exception& error) {
                cerr
                    << image_name
                    << ": "
                    << error.what()
                    << '\n';

                csv
                    << image_name
                    << ','
                    << IMAGE_PIXELS
                    << ",0,0,error\n";

                ++failures;
            }
        }

        const size_t image_count =
            sizeof(IMAGE_NAMES) /
            sizeof(IMAGE_NAMES[0]);

        cout << "\nCompression testbench summary\n";

        cout
            << "Images processed: "
            << image_count
            << '\n';

        cout
            << "Successful images: "
            << image_count - failures
            << '\n';

        cout
            << "Failures: "
            << failures
            << '\n';

        if (total_compressed_bytes != 0) {
            cout
                << "Overall compression ratio: "
                << static_cast<double>(
                    total_raw_bytes
                ) /
                static_cast<double>(
                    total_compressed_bytes
                )
                << ":1\n";
        }

        return failures == 0 ? 0 : 1;
    }
    catch (const exception& error) {
        cerr
            << "Testbench failed: "
            << error.what()
            << '\n';

        return 1;
    }
}
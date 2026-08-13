#include "compression_config.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::uint16_t QUANTISATION_MATRIX[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

struct Statistics {
    double average_ms = 0.0;
    double minimum_ms = 0.0;
    double maximum_ms = 0.0;
};

struct PipelineProfile {
    Statistics load_shift;
    Statistics dct;
    Statistics quantise;
    Statistics encode;
    Statistics total_kernel;
};

std::string readToken(std::istream& input) {
    while (true) {
        const int next = input.peek();

        if (next == EOF) {
            return {};
        }

        if (std::isspace(static_cast<unsigned char>(next))) {
            input.get();
            continue;
        }

        if (next == '#') {
            input.ignore(
                std::numeric_limits<std::streamsize>::max(),
                '\n'
            );
            continue;
        }

        break;
    }

    std::string token;

    while (true) {
        const int next = input.peek();

        if (
            next == EOF ||
            std::isspace(static_cast<unsigned char>(next)) ||
            next == '#'
        ) {
            break;
        }

        token.push_back(static_cast<char>(input.get()));
    }

    return token;
}

GrayImage loadPgm(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);

    if (!input) {
        throw std::runtime_error(
            "Could not open " + path.string()
        );
    }

    const std::string magic = readToken(input);
    const std::string width_text = readToken(input);
    const std::string height_text = readToken(input);
    const std::string maximum_text = readToken(input);

    if (
        magic.empty() ||
        width_text.empty() ||
        height_text.empty() ||
        maximum_text.empty()
    ) {
        throw std::runtime_error(
            "Incomplete PGM header: " + path.string()
        );
    }

    if (magic != "P5") {
        throw std::runtime_error(
            "Expected an 8-bit binary PGM (P5): " +
            path.string()
        );
    }

    GrayImage image;
    image.width = std::stoi(width_text);
    image.height = std::stoi(height_text);
    const int maximum_value = std::stoi(maximum_text);

    if (
        image.width != IMAGE_WIDTH ||
        image.height != IMAGE_HEIGHT ||
        maximum_value != 255
    ) {
        throw std::runtime_error(
            "The hardware requires a 512x512 8-bit PGM: " +
            path.string()
        );
    }

    char separator = '\0';

    if (
        !input.get(separator) ||
        !std::isspace(static_cast<unsigned char>(separator))
    ) {
        throw std::runtime_error(
            "Invalid PGM separator: " + path.string()
        );
    }

    if (separator == '\r' && input.peek() == '\n') {
        input.get();
    }

    image.pixels.resize(IMAGE_PIXELS);
    input.read(
        reinterpret_cast<char*>(image.pixels.data()),
        static_cast<std::streamsize>(IMAGE_PIXELS)
    );

    if (
        input.gcount() !=
        static_cast<std::streamsize>(IMAGE_PIXELS)
    ) {
        throw std::runtime_error(
            "Incomplete pixel data: " + path.string()
        );
    }

    return image;
}

void writeUInt8(std::ostream& output, std::uint8_t value) {
    output.put(static_cast<char>(value));
}

void writeUInt16(std::ostream& output, std::uint16_t value) {
    writeUInt8(
        output,
        static_cast<std::uint8_t>(value & 0xFFU)
    );
    writeUInt8(
        output,
        static_cast<std::uint8_t>((value >> 8) & 0xFFU)
    );
}

void writeInt16(std::ostream& output, std::int16_t value) {
    writeUInt16(output, static_cast<std::uint16_t>(value));
}

void writeUInt32(std::ostream& output, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        writeUInt8(
            output,
            static_cast<std::uint8_t>(
                (value >> shift) & 0xFFU
            )
        );
    }
}

void saveGsc2(
    const fs::path& path,
    const std::int16_t* dc_values,
    const std::uint16_t* ac_bit_counts,
    const std::uint8_t* ac_data
) {
    std::ofstream output(path, std::ios::binary);

    if (!output) {
        throw std::runtime_error(
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
        for (std::uint16_t value : row) {
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
            reinterpret_cast<const char*>(ac_data + base),
            static_cast<std::streamsize>(byte_count)
        );
    }

    if (!output) {
        throw std::runtime_error(
            "Failed while writing " + path.string()
        );
    }
}

std::vector<fs::path> collectImages(const fs::path& input_path) {
    std::vector<fs::path> images;

    if (fs::is_regular_file(input_path)) {
        images.push_back(input_path);
    }
    else if (fs::is_directory(input_path)) {
        for (const auto& entry :
             fs::directory_iterator(input_path)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            std::string extension =
                entry.path().extension().string();
            std::transform(
                extension.begin(),
                extension.end(),
                extension.begin(),
                [](unsigned char character) {
                    return static_cast<char>(
                        std::tolower(character)
                    );
                }
            );

            if (extension == ".pgm") {
                images.push_back(entry.path());
            }
        }
    }
    else {
        throw std::runtime_error(
            "Input path does not exist: " +
            input_path.string()
        );
    }

    std::sort(images.begin(), images.end());

    if (images.empty()) {
        throw std::runtime_error(
            "No PGM files found in " + input_path.string()
        );
    }

    return images;
}

double elapsedMilliseconds(
    Clock::time_point start,
    Clock::time_point finish
) {
    return std::chrono::duration<double, std::milli>(
        finish - start
    ).count();
}

Statistics calculateStatistics(
    const std::vector<double>& samples
) {
    if (samples.empty()) {
        throw std::runtime_error(
            "Cannot calculate statistics without samples"
        );
    }

    const auto minimum =
        std::min_element(samples.begin(), samples.end());
    const auto maximum =
        std::max_element(samples.begin(), samples.end());
    const double total =
        std::accumulate(samples.begin(), samples.end(), 0.0);

    Statistics result;
    result.average_ms =
        total / static_cast<double>(samples.size());
    result.minimum_ms = *minimum;
    result.maximum_ms = *maximum;
    return result;
}

template <typename LaunchFunction>
double timeKernel(LaunchFunction&& launch) {
    const auto start = Clock::now();
    auto run = launch();
    run.wait();
    const auto finish = Clock::now();
    return elapsedMilliseconds(start, finish);
}

void runPipelineOnce(
    xrt::kernel& load_shift,
    xrt::kernel& dct,
    xrt::kernel& quantise,
    xrt::kernel& encode,
    xrt::bo& input_bo,
    xrt::bo& shifted_bo,
    xrt::bo& dct_bo,
    xrt::bo& quantised_bo,
    xrt::bo& dc_bo,
    xrt::bo& bit_count_bo,
    xrt::bo& ac_data_bo,
    xrt::bo& compact_size_bo,
    xrt::bo& status_bo
) {
    auto load_run = load_shift(input_bo, shifted_bo);
    load_run.wait();

    auto dct_run = dct(shifted_bo, dct_bo);
    dct_run.wait();

    auto quantise_run = quantise(dct_bo, quantised_bo);
    quantise_run.wait();

    auto encode_run = encode(
        quantised_bo,
        dc_bo,
        bit_count_bo,
        ac_data_bo,
        compact_size_bo,
        status_bo
    );
    encode_run.wait();
}

PipelineProfile profilePipeline(
    int timed_runs,
    xrt::kernel& load_shift,
    xrt::kernel& dct,
    xrt::kernel& quantise,
    xrt::kernel& encode,
    xrt::bo& input_bo,
    xrt::bo& shifted_bo,
    xrt::bo& dct_bo,
    xrt::bo& quantised_bo,
    xrt::bo& dc_bo,
    xrt::bo& bit_count_bo,
    xrt::bo& ac_data_bo,
    xrt::bo& compact_size_bo,
    xrt::bo& status_bo
) {
    // One untimed run removes first-launch and cache effects from the samples.
    runPipelineOnce(
        load_shift,
        dct,
        quantise,
        encode,
        input_bo,
        shifted_bo,
        dct_bo,
        quantised_bo,
        dc_bo,
        bit_count_bo,
        ac_data_bo,
        compact_size_bo,
        status_bo
    );

    std::vector<double> load_samples;
    std::vector<double> dct_samples;
    std::vector<double> quantise_samples;
    std::vector<double> encode_samples;
    std::vector<double> total_samples;

    load_samples.reserve(timed_runs);
    dct_samples.reserve(timed_runs);
    quantise_samples.reserve(timed_runs);
    encode_samples.reserve(timed_runs);
    total_samples.reserve(timed_runs);

    for (int run_index = 0;
         run_index < timed_runs;
         ++run_index) {
        const auto total_start = Clock::now();

        load_samples.push_back(
            timeKernel([&]() {
                return load_shift(input_bo, shifted_bo);
            })
        );

        dct_samples.push_back(
            timeKernel([&]() {
                return dct(shifted_bo, dct_bo);
            })
        );

        quantise_samples.push_back(
            timeKernel([&]() {
                return quantise(dct_bo, quantised_bo);
            })
        );

        encode_samples.push_back(
            timeKernel([&]() {
                return encode(
                    quantised_bo,
                    dc_bo,
                    bit_count_bo,
                    ac_data_bo,
                    compact_size_bo,
                    status_bo
                );
            })
        );

        const auto total_finish = Clock::now();
        total_samples.push_back(
            elapsedMilliseconds(total_start, total_finish)
        );
    }

    PipelineProfile profile;
    profile.load_shift = calculateStatistics(load_samples);
    profile.dct = calculateStatistics(dct_samples);
    profile.quantise = calculateStatistics(quantise_samples);
    profile.encode = calculateStatistics(encode_samples);
    profile.total_kernel = calculateStatistics(total_samples);
    return profile;
}

void printUsage(const char* program_name) {
    std::cerr
        << "Usage:\n  "
        << program_name
        << " <compression_pipeline.xclbin>"
        << " <image.pgm|image_directory>"
        << " <output_directory> [timed_runs]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const fs::path xclbin_path = argv[1];
        const fs::path input_path = argv[2];
        const fs::path output_directory = argv[3];
        const int timed_runs =
            argc == 5 ? std::stoi(argv[4]) : 10;

        if (timed_runs <= 0) {
            throw std::runtime_error(
                "timed_runs must be greater than zero"
            );
        }

        fs::create_directories(output_directory);
        const std::vector<fs::path> images =
            collectImages(input_path);

        std::cout << "Opening XRT device 0\n";
        xrt::device device(0);

        std::cout
            << "Loading "
            << xclbin_path
            << '\n';
        const auto uuid =
            device.load_xclbin(xclbin_path.string());

        xrt::kernel load_shift(
            device,
            uuid,
            "load_shift_kernel"
        );
        xrt::kernel dct(
            device,
            uuid,
            "dct_kernel"
        );
        xrt::kernel quantise(
            device,
            uuid,
            "quantise_kernel"
        );
        xrt::kernel encode(
            device,
            uuid,
            "encode_kernel"
        );

        constexpr std::size_t input_bytes =
            IMAGE_PIXELS * sizeof(std::uint8_t);
        constexpr std::size_t shifted_bytes =
            TOTAL_COEFFICIENTS * sizeof(shifted_storage_t);
        constexpr std::size_t dct_bytes =
            TOTAL_COEFFICIENTS * sizeof(dct_storage_t);
        constexpr std::size_t quantised_bytes =
            TOTAL_COEFFICIENTS *
            sizeof(quantised_storage_t);
        constexpr std::size_t dc_bytes =
            NUM_BLOCKS * sizeof(std::int16_t);
        constexpr std::size_t bit_count_bytes =
            NUM_BLOCKS * sizeof(std::uint16_t);
        constexpr std::size_t ac_data_bytes =
            TOTAL_AC_STORAGE_BYTES * sizeof(std::uint8_t);

        xrt::bo input_bo(
            device,
            input_bytes,
            load_shift.group_id(0)
        );
        xrt::bo shifted_bo(
            device,
            shifted_bytes,
            load_shift.group_id(1)
        );
        xrt::bo dct_bo(
            device,
            dct_bytes,
            dct.group_id(1)
        );
        xrt::bo quantised_bo(
            device,
            quantised_bytes,
            quantise.group_id(1)
        );
        xrt::bo dc_bo(
            device,
            dc_bytes,
            encode.group_id(1)
        );
        xrt::bo bit_count_bo(
            device,
            bit_count_bytes,
            encode.group_id(2)
        );
        xrt::bo ac_data_bo(
            device,
            ac_data_bytes,
            encode.group_id(3)
        );
        xrt::bo compact_size_bo(
            device,
            sizeof(std::uint32_t),
            encode.group_id(4)
        );
        xrt::bo status_bo(
            device,
            sizeof(std::uint32_t),
            encode.group_id(5)
        );

        auto* input_map = input_bo.map<std::uint8_t*>();
        auto* dc_map = dc_bo.map<std::int16_t*>();
        auto* bit_count_map =
            bit_count_bo.map<std::uint16_t*>();
        auto* ac_data_map =
            ac_data_bo.map<std::uint8_t*>();
        auto* compact_size_map =
            compact_size_bo.map<std::uint32_t*>();
        auto* status_map =
            status_bo.map<std::uint32_t*>();

        std::ofstream csv(
            output_directory / "hardware_stage_profile.csv"
        );

        if (!csv) {
            throw std::runtime_error(
                "Could not create hardware_stage_profile.csv"
            );
        }

        csv
            << "image,raw_bytes,compressed_bytes,compression_ratio,status,"
            << "timed_runs,pgm_load_ms,host_copy_ms,h2d_ms,"
            << "load_shift_avg_ms,load_shift_min_ms,load_shift_max_ms,"
            << "dct_avg_ms,dct_min_ms,dct_max_ms,"
            << "quantise_avg_ms,quantise_min_ms,quantise_max_ms,"
            << "encode_avg_ms,encode_min_ms,encode_max_ms,"
            << "total_kernel_avg_ms,total_kernel_min_ms,total_kernel_max_ms,"
            << "d2h_ms,file_write_ms,end_to_end_ms,kernel_mpix_per_s\n";

        std::cout << std::fixed << std::setprecision(3);
        int failures = 0;

        for (const fs::path& image_path : images) {
            const auto end_to_end_start = Clock::now();

            try {
                const auto load_start = Clock::now();
                const GrayImage image = loadPgm(image_path);
                const auto load_finish = Clock::now();
                const double pgm_load_ms =
                    elapsedMilliseconds(load_start, load_finish);

                const auto copy_start = Clock::now();
                std::copy(
                    image.pixels.begin(),
                    image.pixels.end(),
                    input_map
                );
                const auto copy_finish = Clock::now();
                const double host_copy_ms =
                    elapsedMilliseconds(copy_start, copy_finish);

                const auto h2d_start = Clock::now();
                input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                const auto h2d_finish = Clock::now();
                const double h2d_ms =
                    elapsedMilliseconds(h2d_start, h2d_finish);

                const PipelineProfile profile =
                    profilePipeline(
                        timed_runs,
                        load_shift,
                        dct,
                        quantise,
                        encode,
                        input_bo,
                        shifted_bo,
                        dct_bo,
                        quantised_bo,
                        dc_bo,
                        bit_count_bo,
                        ac_data_bo,
                        compact_size_bo,
                        status_bo
                    );

                const auto d2h_start = Clock::now();
                dc_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                bit_count_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                ac_data_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                compact_size_bo.sync(
                    XCL_BO_SYNC_BO_FROM_DEVICE
                );
                status_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const auto d2h_finish = Clock::now();
                const double d2h_ms =
                    elapsedMilliseconds(d2h_start, d2h_finish);

                const std::uint32_t kernel_status = status_map[0];
                const std::uint32_t reported_size =
                    compact_size_map[0];

                if (kernel_status != 0) {
                    throw std::runtime_error(
                        "Kernel status = " +
                        std::to_string(kernel_status)
                    );
                }

                const fs::path output_path =
                    output_directory /
                    (image_path.stem().string() +
                     "_four_kernel.gsc");

                const auto write_start = Clock::now();
                saveGsc2(
                    output_path,
                    dc_map,
                    bit_count_map,
                    ac_data_map
                );
                const auto write_finish = Clock::now();
                const double file_write_ms =
                    elapsedMilliseconds(write_start, write_finish);

                const std::uintmax_t written_size =
                    fs::file_size(output_path);

                if (written_size != reported_size) {
                    throw std::runtime_error(
                        "Kernel reported " +
                        std::to_string(reported_size) +
                        " bytes, but the GSC file contains " +
                        std::to_string(written_size) +
                        " bytes"
                    );
                }

                const double compression_ratio =
                    written_size == 0
                        ? 0.0
                        : static_cast<double>(IMAGE_PIXELS) /
                          static_cast<double>(written_size);

                const auto end_to_end_finish = Clock::now();
                const double end_to_end_ms =
                    elapsedMilliseconds(
                        end_to_end_start,
                        end_to_end_finish
                    );

                const double kernel_mpix_per_s =
                    profile.total_kernel.average_ms <= 0.0
                        ? 0.0
                        : (static_cast<double>(IMAGE_PIXELS) /
                           1'000'000.0) /
                          (profile.total_kernel.average_ms /
                           1000.0);

                std::cout
                    << std::setw(12)
                    << image_path.filename().string()
                    << " | load "
                    << profile.load_shift.average_ms
                    << " ms | DCT "
                    << profile.dct.average_ms
                    << " ms | quantise "
                    << profile.quantise.average_ms
                    << " ms | encode "
                    << profile.encode.average_ms
                    << " ms | total "
                    << profile.total_kernel.average_ms
                    << " ms | ratio "
                    << compression_ratio
                    << ":1\n";

                csv
                    << image_path.filename().string() << ','
                    << IMAGE_PIXELS << ','
                    << written_size << ','
                    << compression_ratio << ','
                    << kernel_status << ','
                    << timed_runs << ','
                    << pgm_load_ms << ','
                    << host_copy_ms << ','
                    << h2d_ms << ','
                    << profile.load_shift.average_ms << ','
                    << profile.load_shift.minimum_ms << ','
                    << profile.load_shift.maximum_ms << ','
                    << profile.dct.average_ms << ','
                    << profile.dct.minimum_ms << ','
                    << profile.dct.maximum_ms << ','
                    << profile.quantise.average_ms << ','
                    << profile.quantise.minimum_ms << ','
                    << profile.quantise.maximum_ms << ','
                    << profile.encode.average_ms << ','
                    << profile.encode.minimum_ms << ','
                    << profile.encode.maximum_ms << ','
                    << profile.total_kernel.average_ms << ','
                    << profile.total_kernel.minimum_ms << ','
                    << profile.total_kernel.maximum_ms << ','
                    << d2h_ms << ','
                    << file_write_ms << ','
                    << end_to_end_ms << ','
                    << kernel_mpix_per_s
                    << '\n';
            }
            catch (const std::exception& error) {
                ++failures;
                std::cerr
                    << image_path.filename().string()
                    << ": "
                    << error.what()
                    << '\n';
            }
        }

        std::cout
            << "\nProfile CSV: "
            << output_directory / "hardware_stage_profile.csv"
            << '\n';
        std::cout
            << "Images processed: "
            << images.size()
            << ", failures: "
            << failures
            << '\n';

        return failures == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Host failed: " << error.what() << '\n';
        return 1;
    }
}

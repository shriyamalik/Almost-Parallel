#include "compression_header.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

struct BufferSet {
    xrt::bo input_bo;
    xrt::bo dc_bo;
    xrt::bo pair_count_bo;
    xrt::bo zero_run_bo;
    xrt::bo pair_value_bo;

    std::uint8_t* input_map = nullptr;
    std::int16_t* dc_map = nullptr;
    std::uint8_t* pair_count_map = nullptr;
    std::uint8_t* zero_run_map = nullptr;
    std::int16_t* pair_value_map = nullptr;

    BufferSet(
        xrt::device& device,
        xrt::kernel& compression,
        std::size_t input_bytes,
        std::size_t dc_bytes,
        std::size_t pair_count_bytes,
        std::size_t zero_run_bytes,
        std::size_t pair_value_bytes
    )
        : input_bo(device, input_bytes, compression.group_id(0)),
          dc_bo(device, dc_bytes, compression.group_id(1)),
          pair_count_bo(device, pair_count_bytes, compression.group_id(2)),
          zero_run_bo(device, zero_run_bytes, compression.group_id(3)),
          pair_value_bo(device, pair_value_bytes, compression.group_id(4))
    {
        input_map = input_bo.map<std::uint8_t*>();
        dc_map = dc_bo.map<std::int16_t*>();
        pair_count_map = pair_count_bo.map<std::uint8_t*>();
        zero_run_map = zero_run_bo.map<std::uint8_t*>();
        pair_value_map = pair_value_bo.map<std::int16_t*>();
    }
};

struct PreparedTiming {
    double pgm_load_ms = 0.0;
    double host_copy_ms = 0.0;
    double h2d_ms = 0.0;
};

double elapsedMilliseconds(Clock::time_point start, Clock::time_point finish) {
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

std::string readToken(std::istream& input) {
    while (true) {
        const int next = input.peek();
        if (next == EOF) return {};
        if (std::isspace(static_cast<unsigned char>(next))) {
            input.get();
            continue;
        }
        if (next == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        break;
    }

    std::string token;

    while (true) {
        const int next = input.peek();
        if (next == EOF || std::isspace(static_cast<unsigned char>(next)) || next == '#') break;
        token.push_back(static_cast<char>(input.get()));
    }

    return token;
}

GrayImage loadPgm(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Could not open " + path.string());

    const std::string magic = readToken(input);
    const std::string width_text = readToken(input);
    const std::string height_text = readToken(input);
    const std::string maximum_text = readToken(input);

    if (magic != "P5" || width_text.empty() || height_text.empty() || maximum_text.empty())
        throw std::runtime_error("Invalid PGM header: " + path.string());

    GrayImage image;
    image.width = std::stoi(width_text);
    image.height = std::stoi(height_text);
    const int maximum_value = std::stoi(maximum_text);

    if (image.width != IMAGE_WIDTH || image.height != IMAGE_HEIGHT || maximum_value != 255)
        throw std::runtime_error("Hardware requires a 512x512 8-bit PGM: " + path.string());

    char separator = '\0';

    if (!input.get(separator) || !std::isspace(static_cast<unsigned char>(separator)))
        throw std::runtime_error("Invalid PGM separator: " + path.string());

    if (separator == '\r' && input.peek() == '\n') input.get();

    image.pixels.resize(IMAGE_PIXELS);
    input.read(reinterpret_cast<char*>(image.pixels.data()), IMAGE_PIXELS);

    if (input.gcount() != static_cast<std::streamsize>(IMAGE_PIXELS))
        throw std::runtime_error("Incomplete pixel data: " + path.string());

    return image;
}

void saveGsc1(
    const fs::path& path,
    const std::int16_t* dc_values,
    const std::uint8_t* pair_counts,
    const std::uint8_t* zero_runs,
    const std::int16_t* pair_values
) {
    std::size_t total_pairs = 0;

    for (int block = 0; block < NUM_BLOCKS; ++block) {
        const std::uint8_t count = pair_counts[block];

        if (count > MAX_RLE_PAIRS_PER_BLOCK)
            throw std::runtime_error("Invalid RLE pair count in block " + std::to_string(block));

        total_pairs += count;
    }

    constexpr std::size_t HEADER_SIZE = 4 + (5 * 4) + (BLOCK_ELEMENTS * 2);
    const std::size_t total_size = HEADER_SIZE + static_cast<std::size_t>(NUM_BLOCKS) * 3 + total_pairs * 3;

    std::vector<std::uint8_t> buffer(total_size);
    std::size_t position = 0;

    auto putUInt8 = [&](std::uint8_t value) {
        buffer[position++] = value;
    };

    auto putUInt16 = [&](std::uint16_t value) {
        buffer[position++] = static_cast<std::uint8_t>(value & 0xFFU);
        buffer[position++] = static_cast<std::uint8_t>((value >> 8) & 0xFFU);
    };

    auto putInt16 = [&](std::int16_t value) {
        const std::uint16_t u = static_cast<std::uint16_t>(value);
        buffer[position++] = static_cast<std::uint8_t>(u & 0xFFU);
        buffer[position++] = static_cast<std::uint8_t>((u >> 8) & 0xFFU);
    };

    auto putUInt32 = [&](std::uint32_t value) {
        buffer[position++] = static_cast<std::uint8_t>(value & 0xFFU);
        buffer[position++] = static_cast<std::uint8_t>((value >> 8) & 0xFFU);
        buffer[position++] = static_cast<std::uint8_t>((value >> 16) & 0xFFU);
        buffer[position++] = static_cast<std::uint8_t>((value >> 24) & 0xFFU);
    };

    putUInt8('G');
    putUInt8('S');
    putUInt8('C');
    putUInt8('1');

    putUInt32(IMAGE_WIDTH);
    putUInt32(IMAGE_HEIGHT);
    putUInt32(IMAGE_WIDTH);
    putUInt32(IMAGE_HEIGHT);
    putUInt32(NUM_BLOCKS);

    for (const auto& row : QUANTISATION_MATRIX)
        for (std::uint16_t value : row)
            putUInt16(value);

    for (int block = 0; block < NUM_BLOCKS; ++block) {
        putInt16(dc_values[block]);

        const std::uint8_t count = pair_counts[block];
        putUInt8(count);

        const int base = block * MAX_RLE_PAIRS_PER_BLOCK;

        for (int pair = 0; pair < count; ++pair) {
            putUInt8(zero_runs[base + pair]);
            putInt16(pair_values[base + pair]);
        }
    }

    if (position != total_size) throw std::runtime_error("Internal GSC1 size mismatch");

    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Could not create " + path.string());

    output.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));

    if (!output) throw std::runtime_error("Failed while writing " + path.string());
}

std::vector<fs::path> collectImages(const fs::path& input_path) {
    std::vector<fs::path> images;

    if (fs::is_regular_file(input_path)) {
        images.push_back(input_path);
    } else if (fs::is_directory(input_path)) {
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (!entry.is_regular_file()) continue;

            std::string extension = entry.path().extension().string();

            std::transform(
                extension.begin(),
                extension.end(),
                extension.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
            );

            if (extension == ".pgm") images.push_back(entry.path());
        }
    } else {
        throw std::runtime_error("Input path does not exist: " + input_path.string());
    }

    std::sort(images.begin(), images.end());

    if (images.empty()) throw std::runtime_error("No PGM files found in " + input_path.string());

    return images;
}

void printUsage(const char* program_name) {
    std::cerr << "Usage:\n  " << program_name
              << " <compression_stream.xclbin>"
              << " <image.pgm|image_directory>"
              << " <output_directory>\n";
}

}

int main(int argc, char** argv) {
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const fs::path xclbin_path = argv[1];
        const fs::path input_path = argv[2];
        const fs::path output_directory = argv[3];

        fs::create_directories(output_directory);
        const auto images = collectImages(input_path);

        std::cout << "Opening XRT device 0\n";
        xrt::device device(0);

        std::cout << "Loading " << xclbin_path << '\n';
        const auto uuid = device.load_xclbin(xclbin_path.string());

        xrt::kernel compression(device, uuid, "compression_stream_kernel");

        constexpr std::size_t input_bytes = IMAGE_PIXELS * sizeof(std::uint8_t);
        constexpr std::size_t dc_bytes = NUM_BLOCKS * sizeof(std::int16_t);
        constexpr std::size_t pair_count_bytes = NUM_BLOCKS * sizeof(std::uint8_t);
        constexpr std::size_t zero_run_bytes = TOTAL_RLE_PAIR_SLOTS * sizeof(std::uint8_t);
        constexpr std::size_t pair_value_bytes = TOTAL_RLE_PAIR_SLOTS * sizeof(std::int16_t);

        BufferSet slot0(device, compression, input_bytes, dc_bytes, pair_count_bytes, zero_run_bytes, pair_value_bytes);
        BufferSet slot1(device, compression, input_bytes, dc_bytes, pair_count_bytes, zero_run_bytes, pair_value_bytes);

        BufferSet* slots[2] = {&slot0, &slot1};

        std::ofstream csv(output_directory / "stream_profile.csv");
        if (!csv) throw std::runtime_error("Could not create stream_profile.csv");

        csv << "image,raw_bytes,compressed_bytes,compression_ratio,timed_runs,"
            << "pgm_load_ms,host_copy_ms,h2d_ms,kernel_avg_ms,kernel_min_ms,kernel_max_ms,"
            << "d2h_ms,file_write_ms,pipeline_with_transfer_avg_ms,estimated_end_to_end_avg_ms,"
            << "kernel_mpix_per_s\n";

        std::cout << std::fixed << std::setprecision(3);

        PreparedTiming timing[2];
        std::future<double> kernel_future[2];

        int failures = 0;
        double total_kernel_ms = 0.0;
        double total_estimated_end_to_end_ms = 0.0;

        auto prepareImage = [&](std::size_t image_index, int slot_index) {
            BufferSet& slot = *slots[slot_index];
            const fs::path& image_path = images[image_index];

            const auto load_start = Clock::now();
            const GrayImage image = loadPgm(image_path);
            const auto load_finish = Clock::now();
            timing[slot_index].pgm_load_ms = elapsedMilliseconds(load_start, load_finish);

            const auto copy_start = Clock::now();
            std::copy(image.pixels.begin(), image.pixels.end(), slot.input_map);
            const auto copy_finish = Clock::now();
            timing[slot_index].host_copy_ms = elapsedMilliseconds(copy_start, copy_finish);

            const auto h2d_start = Clock::now();
            slot.input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const auto h2d_finish = Clock::now();
            timing[slot_index].h2d_ms = elapsedMilliseconds(h2d_start, h2d_finish);
        };

        auto launchKernel = [&](int slot_index) {
            BufferSet& slot = *slots[slot_index];
            const auto kernel_start = Clock::now();

            xrt::run run = compression(
                slot.input_bo,
                slot.dc_bo,
                slot.pair_count_bo,
                slot.zero_run_bo,
                slot.pair_value_bo
            );

            kernel_future[slot_index] = std::async(
                std::launch::async,
                [run = std::move(run), kernel_start]() mutable {
                    run.wait();
                    return elapsedMilliseconds(kernel_start, Clock::now());
                }
            );
        };

        const auto batch_start = Clock::now();

        prepareImage(0, 0);
        launchKernel(0);

        for (std::size_t i = 0; i < images.size(); ++i) {
            const int current_slot = static_cast<int>(i % 2);
            const int next_slot = 1 - current_slot;
            const bool has_next = (i + 1 < images.size());

            BufferSet& current = *slots[current_slot];
            const fs::path& image_path = images[i];

            try {
                if (has_next) prepareImage(i + 1, next_slot);

                const double kernel_ms = kernel_future[current_slot].get();

                if (has_next) launchKernel(next_slot);

                const auto d2h_start = Clock::now();

                current.dc_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                current.pair_count_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                current.zero_run_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                current.pair_value_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                const auto d2h_finish = Clock::now();
                const double d2h_ms = elapsedMilliseconds(d2h_start, d2h_finish);

                const fs::path output_path =
                    output_directory /
                    (image_path.stem().string() + "_stream.gsc");

                const auto write_start = Clock::now();

                saveGsc1(
                    output_path,
                    current.dc_map,
                    current.pair_count_map,
                    current.zero_run_map,
                    current.pair_value_map
                );

                const auto write_finish = Clock::now();
                const double file_write_ms = elapsedMilliseconds(write_start, write_finish);

                const std::uintmax_t compressed_bytes = fs::file_size(output_path);

                const double compression_ratio =
                    compressed_bytes == 0
                    ? 0.0
                    : static_cast<double>(IMAGE_PIXELS) /
                      static_cast<double>(compressed_bytes);

                const double pgm_load_ms = timing[current_slot].pgm_load_ms;
                const double host_copy_ms = timing[current_slot].host_copy_ms;
                const double h2d_ms = timing[current_slot].h2d_ms;

                const double pipeline_with_transfer_avg_ms =
                    h2d_ms + kernel_ms + d2h_ms;

                const double estimated_end_to_end_avg_ms =
                    pgm_load_ms +
                    host_copy_ms +
                    pipeline_with_transfer_avg_ms +
                    file_write_ms;

                const double kernel_mpix_per_s =
                    kernel_ms <= 0.0
                    ? 0.0
                    : (static_cast<double>(IMAGE_PIXELS) / 1'000'000.0) /
                      (kernel_ms / 1000.0);

                total_kernel_ms += kernel_ms;
                total_estimated_end_to_end_ms += estimated_end_to_end_avg_ms;

                std::cout
                    << image_path.filename().string()
                    << " | Total: "
                    << estimated_end_to_end_avg_ms
                    << " ms"
                    << " | Kernel: "
                    << kernel_ms
                    << " ms"
                    << " | Write: "
                    << file_write_ms
                    << " ms\n";

                csv
                    << image_path.filename().string() << ','
                    << IMAGE_PIXELS << ','
                    << compressed_bytes << ','
                    << compression_ratio << ','
                    << 1 << ','
                    << pgm_load_ms << ','
                    << host_copy_ms << ','
                    << h2d_ms << ','
                    << kernel_ms << ','
                    << kernel_ms << ','
                    << kernel_ms << ','
                    << d2h_ms << ','
                    << file_write_ms << ','
                    << pipeline_with_transfer_avg_ms << ','
                    << estimated_end_to_end_avg_ms << ','
                    << kernel_mpix_per_s << '\n';

            } catch (const std::exception& error) {
                ++failures;

                std::cerr
                    << image_path.filename().string()
                    << ": "
                    << error.what()
                    << '\n';
            }
        }

        const auto batch_finish = Clock::now();
        const double actual_batch_ms = elapsedMilliseconds(batch_start, batch_finish);

        const std::size_t successful_images =
            images.size() -
            static_cast<std::size_t>(failures);

        if (successful_images > 0) {
            const double average_kernel_ms =
                total_kernel_ms /
                static_cast<double>(successful_images);

            const double average_estimated_total_ms =
                total_estimated_end_to_end_ms /
                static_cast<double>(successful_images);

            const double actual_average_ms =
                actual_batch_ms /
                static_cast<double>(successful_images);

            const double images_per_second =
                static_cast<double>(successful_images) /
                (actual_batch_ms / 1000.0);

            std::cout
                << "\nTotal kernel time for "
                << successful_images
                << " images: "
                << total_kernel_ms
                << " ms\n";

            std::cout
                << "Average kernel time per image: "
                << average_kernel_ms
                << " ms\n";

            std::cout
                << "Total estimated stage time for "
                << successful_images
                << " images: "
                << total_estimated_end_to_end_ms
                << " ms\n";

            std::cout
                << "Average estimated stage time per image: "
                << average_estimated_total_ms
                << " ms\n";

            std::cout
                << "Actual pipelined batch time: "
                << actual_batch_ms
                << " ms\n";

            std::cout
                << "Actual pipelined average time per image: "
                << actual_average_ms
                << " ms\n";

            std::cout
                << "Actual pipelined throughput: "
                << images_per_second
                << " images/s\n";
        }

        return failures == 0 ? 0 : 1;

    } catch (const std::exception& error) {
        std::cerr << "Host failed: " << error.what() << '\n';
        return 1;
    }
}
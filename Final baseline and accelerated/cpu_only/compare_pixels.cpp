#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int SSIM_WINDOW_SIZE = 8;

struct GrayImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixels;
};

struct ComparisonMetrics {
    std::uint64_t pixel_count = 0;
    std::uint64_t exact_pixels = 0;
    std::uint64_t within_one_pixels = 0;
    std::uint64_t within_two_pixels = 0;
    std::uint64_t sum_absolute_error = 0;
    std::uint64_t sum_squared_error = 0;
    std::uint64_t original_signal_energy = 0;
    std::int64_t sum_signed_error = 0;
    std::uint16_t maximum_absolute_error = 0;

    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double mean_error = 0.0;
    double psnr_db = std::numeric_limits<double>::infinity();
    double snr_db = std::numeric_limits<double>::infinity();
    double mean_ssim_8x8 = 1.0;
    double exact_percent = 100.0;
    double within_one_percent = 100.0;
    double within_two_percent = 100.0;
    double metric_time_ms = 0.0;
};

std::string readPgmToken(std::istream& input) {
    while (true) {
        const int next = input.peek();

        if (next == std::char_traits<char>::eof()) {
            throw std::runtime_error("Unexpected end of PGM header");
        }

        if (std::isspace(static_cast<unsigned char>(next)) != 0) {
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

        if (
            next == std::char_traits<char>::eof() ||
            std::isspace(static_cast<unsigned char>(next)) != 0 ||
            next == '#'
        ) {
            break;
        }

        token.push_back(static_cast<char>(input.get()));
    }

    if (token.empty()) {
        throw std::runtime_error("Empty token in PGM header");
    }

    return token;
}

std::uint32_t parseU32(
    const std::string& token,
    const std::string& field_name
) {
    std::size_t consumed = 0;
    const unsigned long value = std::stoul(token, &consumed);

    if (
        consumed != token.size() ||
        value > std::numeric_limits<std::uint32_t>::max()
    ) {
        throw std::runtime_error(
            "Invalid PGM " + field_name + ": " + token
        );
    }

    return static_cast<std::uint32_t>(value);
}

GrayImage loadPgmP5(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);

    if (!input) {
        throw std::runtime_error("Could not open PGM file: " + path.string());
    }

    if (readPgmToken(input) != "P5") {
        throw std::runtime_error(
            "Only binary PGM P5 images are supported: " + path.string()
        );
    }

    GrayImage image;
    image.width = parseU32(readPgmToken(input), "width");
    image.height = parseU32(readPgmToken(input), "height");
    const std::uint32_t maximum_value =
        parseU32(readPgmToken(input), "maximum value");

    if (image.width == 0 || image.height == 0) {
        throw std::runtime_error("PGM dimensions must be non-zero");
    }

    if (maximum_value != 255U) {
        throw std::runtime_error(
            "Only 8-bit PGM images with maximum value 255 are supported: " +
            path.string()
        );
    }

    // Consume exactly the delimiter between the header and raster. Do not skip
    // arbitrary whitespace because the first pixel may itself be whitespace.
    char separator = '\0';

    if (
        !input.get(separator) ||
        std::isspace(static_cast<unsigned char>(separator)) == 0
    ) {
        throw std::runtime_error(
            "Missing whitespace before PGM raster: " + path.string()
        );
    }

    if (separator == '\r' && input.peek() == '\n') {
        input.get();
    }

    const std::uint64_t pixel_count_64 =
        static_cast<std::uint64_t>(image.width) * image.height;

    if (pixel_count_64 > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("PGM image is too large: " + path.string());
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(pixel_count_64);

    image.pixels.resize(pixel_count);
    input.read(
        reinterpret_cast<char*>(image.pixels.data()),
        static_cast<std::streamsize>(pixel_count)
    );

    if (input.gcount() != static_cast<std::streamsize>(pixel_count)) {
        throw std::runtime_error(
            "PGM raster is shorter than width x height bytes: " +
            path.string()
        );
    }

    return image;
}


fs::path decompressedPathFor(
    const fs::path& decompressed_directory,
    const char* original_name
) {
    const fs::path name(original_name);
    return decompressed_directory / (name.stem().string() + "_stream.pgm");
}

// Mean SSIM over non-overlapping 8x8 windows. This is a dependency-free,
// block-oriented SSIM measure suitable for the project's 8x8 JPEG-style codec.
double calculateMeanSsim8x8(
    const GrayImage& original,
    const GrayImage& reconstructed
) {
    constexpr long double dynamic_range = 255.0L;
    constexpr long double k1 = 0.01L;
    constexpr long double k2 = 0.03L;
    constexpr long double c1 =
        (k1 * dynamic_range) * (k1 * dynamic_range);
    constexpr long double c2 =
        (k2 * dynamic_range) * (k2 * dynamic_range);

    long double ssim_sum = 0.0L;
    std::uint64_t window_count = 0;

    for (
        std::uint32_t start_y = 0;
        start_y < original.height;
        start_y += SSIM_WINDOW_SIZE
    ) {
        for (
            std::uint32_t start_x = 0;
            start_x < original.width;
            start_x += SSIM_WINDOW_SIZE
        ) {
            const std::uint32_t end_y = std::min<std::uint32_t>(
                start_y + SSIM_WINDOW_SIZE,
                original.height
            );

            const std::uint32_t end_x = std::min<std::uint32_t>(
                start_x + SSIM_WINDOW_SIZE,
                original.width
            );

            const std::uint64_t n =
                static_cast<std::uint64_t>(end_y - start_y) *
                static_cast<std::uint64_t>(end_x - start_x);

            long double mean_x = 0.0L;
            long double mean_y = 0.0L;

            for (std::uint32_t y = start_y; y < end_y; ++y) {
                for (std::uint32_t x = start_x; x < end_x; ++x) {
                    const std::size_t index =
                        static_cast<std::size_t>(y) * original.width + x;

                    mean_x += original.pixels[index];
                    mean_y += reconstructed.pixels[index];
                }
            }

            mean_x /= static_cast<long double>(n);
            mean_y /= static_cast<long double>(n);

            long double variance_x = 0.0L;
            long double variance_y = 0.0L;
            long double covariance = 0.0L;

            for (std::uint32_t y = start_y; y < end_y; ++y) {
                for (std::uint32_t x = start_x; x < end_x; ++x) {
                    const std::size_t index =
                        static_cast<std::size_t>(y) * original.width + x;

                    const long double dx =
                        static_cast<long double>(original.pixels[index]) - mean_x;
                    const long double dy =
                        static_cast<long double>(reconstructed.pixels[index]) - mean_y;

                    variance_x += dx * dx;
                    variance_y += dy * dy;
                    covariance += dx * dy;
                }
            }

            variance_x /= static_cast<long double>(n);
            variance_y /= static_cast<long double>(n);
            covariance /= static_cast<long double>(n);

            const long double numerator =
                (2.0L * mean_x * mean_y + c1) *
                (2.0L * covariance + c2);

            const long double denominator =
                (mean_x * mean_x + mean_y * mean_y + c1) *
                (variance_x + variance_y + c2);

            const long double ssim =
                denominator == 0.0L ? 1.0L : numerator / denominator;

            ssim_sum += ssim;
            ++window_count;
        }
    }

    return window_count == 0
        ? 1.0
        : static_cast<double>(
            ssim_sum / static_cast<long double>(window_count)
        );
}

ComparisonMetrics compareImages(
    const GrayImage& original,
    const GrayImage& reconstructed
) {
    if (
        original.width != reconstructed.width ||
        original.height != reconstructed.height
    ) {
        throw std::runtime_error(
            "Image dimensions do not match: original=" +
            std::to_string(original.width) + "x" +
            std::to_string(original.height) +
            ", reconstructed=" +
            std::to_string(reconstructed.width) + "x" +
            std::to_string(reconstructed.height)
        );
    }

    if (original.pixels.size() != reconstructed.pixels.size()) {
        throw std::runtime_error("Image pixel counts do not match");
    }


    const auto start = Clock::now();

    ComparisonMetrics metrics;
    metrics.pixel_count = original.pixels.size();

    for (std::size_t index = 0; index < original.pixels.size(); ++index) {
        const int original_value = original.pixels[index];
        const int reconstructed_value = reconstructed.pixels[index];
        const int signed_error = reconstructed_value - original_value;
        const int absolute_error = std::abs(signed_error);

        metrics.sum_signed_error += signed_error;
        metrics.sum_absolute_error +=
            static_cast<std::uint64_t>(absolute_error);
        metrics.sum_squared_error +=
            static_cast<std::uint64_t>(absolute_error * absolute_error);
        metrics.original_signal_energy +=
            static_cast<std::uint64_t>(original_value * original_value);
        metrics.maximum_absolute_error = std::max<std::uint16_t>(
            metrics.maximum_absolute_error,
            static_cast<std::uint16_t>(absolute_error)
        );

        if (absolute_error == 0) {
            ++metrics.exact_pixels;
        }

        if (absolute_error <= 1) {
            ++metrics.within_one_pixels;
        }

        if (absolute_error <= 2) {
            ++metrics.within_two_pixels;
        }

    }

    const long double count =
        static_cast<long double>(metrics.pixel_count);

    metrics.mse = static_cast<double>(
        static_cast<long double>(metrics.sum_squared_error) / count
    );

    metrics.rmse = std::sqrt(metrics.mse);

    metrics.mae = static_cast<double>(
        static_cast<long double>(metrics.sum_absolute_error) / count
    );

    metrics.mean_error = static_cast<double>(
        static_cast<long double>(metrics.sum_signed_error) / count
    );

    metrics.exact_percent =
        100.0 * static_cast<double>(metrics.exact_pixels) /
        static_cast<double>(metrics.pixel_count);

    metrics.within_one_percent =
        100.0 * static_cast<double>(metrics.within_one_pixels) /
        static_cast<double>(metrics.pixel_count);

    metrics.within_two_percent =
        100.0 * static_cast<double>(metrics.within_two_pixels) /
        static_cast<double>(metrics.pixel_count);

    if (metrics.sum_squared_error == 0) {
        metrics.psnr_db = std::numeric_limits<double>::infinity();
        metrics.snr_db = std::numeric_limits<double>::infinity();
    } else {
        metrics.psnr_db = 10.0 * std::log10(
            (255.0 * 255.0) / metrics.mse
        );

        if (metrics.original_signal_energy == 0) {
            metrics.snr_db = -std::numeric_limits<double>::infinity();
        } else {
            metrics.snr_db = 10.0 * std::log10(
                static_cast<double>(metrics.original_signal_energy) /
                static_cast<double>(metrics.sum_squared_error)
            );
        }
    }

    metrics.mean_ssim_8x8 =
        calculateMeanSsim8x8(original, reconstructed);

    metrics.metric_time_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    return metrics;
}

std::string formatFloating(double value, int precision = 10) {
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }

    if (std::isnan(value)) {
        return "nan";
    }

    std::ostringstream output;
    output << std::fixed << std::setprecision(precision) << value;
    return output.str();
}

std::string csvEscape(const std::string& text) {
    if (
        text.find_first_of(",\"\n\r") == std::string::npos
    ) {
        return text;
    }

    std::string escaped = "\"";

    for (const char character : text) {
        if (character == '\"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }

    escaped += '\"';
    return escaped;
}

std::vector<std::string> findOriginalImages(const fs::path& directory) {
    std::vector<std::string> image_names;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path path = entry.path();
        if (path.extension() == ".pgm") {
            image_names.push_back(path.filename().string());
        }
    }

    std::sort(image_names.begin(), image_names.end());
    return image_names;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <original_image_directory> <decompressed_image_directory>\n";
        return 1;
    }

    const fs::path original_directory = argv[1];
    const fs::path decompressed_directory = argv[2];
    const fs::path source_directory =
        fs::absolute(fs::path(__FILE__)).parent_path();
    const fs::path results_path =
        source_directory / "image_comparison_metrics.csv";

    try {
        if (!fs::is_directory(original_directory)) {
            throw std::runtime_error(
                "Original-image directory does not exist: " +
                original_directory.string()
            );
        }

        if (!fs::is_directory(decompressed_directory)) {
            throw std::runtime_error(
                "Decompressed-image directory does not exist: " +
                decompressed_directory.string()
            );
        }

        const std::vector<std::string> image_names =
            findOriginalImages(original_directory);

        if (image_names.empty()) {
            throw std::runtime_error(
                "No .pgm images found in original-image directory: " +
                original_directory.string()
            );
        }

        std::ofstream csv(results_path);

        if (!csv) {
            throw std::runtime_error(
                "Could not create results CSV: " + results_path.string()
            );
        }

        csv
            << "image,width,height,pixels,mse,rmse,mae,mean_error,"
            << "max_absolute_error,psnr_db,snr_db,mean_ssim_8x8,"
            << "exact_pixels,exact_percent,within_1_percent,within_2_percent,"
            << "metric_time_ms,status,error\n";

        std::uint64_t total_pixels = 0;
        std::uint64_t total_exact_pixels = 0;
        std::uint64_t total_within_one_pixels = 0;
        std::uint64_t total_within_two_pixels = 0;
        std::uint64_t total_absolute_error = 0;
        std::uint64_t total_squared_error = 0;
        std::uint64_t total_signal_energy = 0;
        std::int64_t total_signed_error = 0;
        std::uint16_t overall_maximum_error = 0;
        long double total_ssim = 0.0L;
        long double total_finite_psnr = 0.0L;
        std::size_t finite_psnr_count = 0;
        std::size_t successful_images = 0;
        std::size_t failed_images = 0;
        double total_metric_time_ms = 0.0;

        double lowest_psnr = std::numeric_limits<double>::infinity();
        double highest_finite_psnr = -std::numeric_limits<double>::infinity();
        std::string lowest_psnr_image;
        std::string highest_psnr_image;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Original directory: " << original_directory << '\n';
        std::cout << "Decompressed directory: " << decompressed_directory << '\n';
        std::cout << "CSV output: " << results_path << "\n\n";

        for (const std::string& image_name : image_names) {
            const fs::path original_path = original_directory / image_name;
            const fs::path reconstructed_path =
                decompressedPathFor(decompressed_directory, image_name.c_str());

            try {
                const GrayImage original = loadPgmP5(original_path);
                const GrayImage reconstructed = loadPgmP5(reconstructed_path);
                const ComparisonMetrics metrics =
                    compareImages(original, reconstructed);

                csv
                    << csvEscape(image_name) << ','
                    << original.width << ','
                    << original.height << ','
                    << metrics.pixel_count << ','
                    << formatFloating(metrics.mse) << ','
                    << formatFloating(metrics.rmse) << ','
                    << formatFloating(metrics.mae) << ','
                    << formatFloating(metrics.mean_error) << ','
                    << metrics.maximum_absolute_error << ','
                    << formatFloating(metrics.psnr_db) << ','
                    << formatFloating(metrics.snr_db) << ','
                    << formatFloating(metrics.mean_ssim_8x8) << ','
                    << metrics.exact_pixels << ','
                    << formatFloating(metrics.exact_percent) << ','
                    << formatFloating(metrics.within_one_percent) << ','
                    << formatFloating(metrics.within_two_percent) << ','
                    << formatFloating(metrics.metric_time_ms, 6) << ','
                    << "ok," << '\n';

                std::cout
                    << std::left << std::setw(28) << image_name
                    << " PSNR=" << std::setw(12)
                    << formatFloating(metrics.psnr_db, 4)
                    << " dB  SSIM=" << std::setw(10)
                    << formatFloating(metrics.mean_ssim_8x8, 6)
                    << " MSE=" << std::setw(12)
                    << formatFloating(metrics.mse, 6)
                    << " MAE=" << std::setw(10)
                    << formatFloating(metrics.mae, 6)
                    << " Max=" << metrics.maximum_absolute_error
                    << '\n';

                total_pixels += metrics.pixel_count;
                total_exact_pixels += metrics.exact_pixels;
                total_within_one_pixels += metrics.within_one_pixels;
                total_within_two_pixels += metrics.within_two_pixels;
                total_absolute_error += metrics.sum_absolute_error;
                total_squared_error += metrics.sum_squared_error;
                total_signal_energy += metrics.original_signal_energy;
                total_signed_error += metrics.sum_signed_error;
                overall_maximum_error = std::max(
                    overall_maximum_error,
                    metrics.maximum_absolute_error
                );
                total_ssim += metrics.mean_ssim_8x8;
                total_metric_time_ms += metrics.metric_time_ms;

                if (std::isfinite(metrics.psnr_db)) {
                    total_finite_psnr += metrics.psnr_db;
                    ++finite_psnr_count;

                    if (metrics.psnr_db < lowest_psnr) {
                        lowest_psnr = metrics.psnr_db;
                        lowest_psnr_image = image_name;
                    }

                    if (metrics.psnr_db > highest_finite_psnr) {
                        highest_finite_psnr = metrics.psnr_db;
                        highest_psnr_image = image_name;
                    }
                }

                ++successful_images;
            }
            catch (const std::exception& error) {
                ++failed_images;

                csv
                    << csvEscape(image_name)
                    << ",,,,,,,,,,,,,,,,,error,"
                    << csvEscape(error.what()) << '\n';

                std::cerr
                    << image_name
                    << ": comparison failed: "
                    << error.what()
                    << '\n';
            }
        }

        if (successful_images != 0 && total_pixels != 0) {
            const double pooled_mse =
                static_cast<double>(total_squared_error) /
                static_cast<double>(total_pixels);

            const double pooled_rmse = std::sqrt(pooled_mse);
            const double pooled_mae =
                static_cast<double>(total_absolute_error) /
                static_cast<double>(total_pixels);

            const double pooled_mean_error =
                static_cast<double>(total_signed_error) /
                static_cast<double>(total_pixels);

            const double pooled_psnr = total_squared_error == 0
                ? std::numeric_limits<double>::infinity()
                : 10.0 * std::log10((255.0 * 255.0) / pooled_mse);

            const double pooled_snr = total_squared_error == 0
                ? std::numeric_limits<double>::infinity()
                : (total_signal_energy == 0
                    ? -std::numeric_limits<double>::infinity()
                    : 10.0 * std::log10(
                        static_cast<double>(total_signal_energy) /
                        static_cast<double>(total_squared_error)
                    ));

            const double mean_ssim = static_cast<double>(
                total_ssim / static_cast<long double>(successful_images)
            );

            const double exact_percent =
                100.0 * static_cast<double>(total_exact_pixels) /
                static_cast<double>(total_pixels);

            const double within_one_percent =
                100.0 * static_cast<double>(total_within_one_pixels) /
                static_cast<double>(total_pixels);

            const double within_two_percent =
                100.0 * static_cast<double>(total_within_two_pixels) /
                static_cast<double>(total_pixels);

            const double mean_finite_psnr = finite_psnr_count == 0
                ? std::numeric_limits<double>::infinity()
                : static_cast<double>(
                    total_finite_psnr /
                    static_cast<long double>(finite_psnr_count)
                );

            csv
                << "AGGREGATE,,," << total_pixels << ','
                << formatFloating(pooled_mse) << ','
                << formatFloating(pooled_rmse) << ','
                << formatFloating(pooled_mae) << ','
                << formatFloating(pooled_mean_error) << ','
                << overall_maximum_error << ','
                << formatFloating(pooled_psnr) << ','
                << formatFloating(pooled_snr) << ','
                << formatFloating(mean_ssim) << ','
                << total_exact_pixels << ','
                << formatFloating(exact_percent) << ','
                << formatFloating(within_one_percent) << ','
                << formatFloating(within_two_percent) << ','
                << formatFloating(total_metric_time_ms, 6)
                << ",ok,\n";

            std::cout << "\nComparison summary\n";
            std::cout << "Images found: " << image_names.size() << '\n';
            std::cout << "Successful images: " << successful_images << '\n';
            std::cout << "Failures: " << failed_images << '\n';
            std::cout << "Total pixels compared: " << total_pixels << '\n';
            std::cout << "Pooled MSE: " << formatFloating(pooled_mse, 8) << '\n';
            std::cout << "Pooled RMSE: " << formatFloating(pooled_rmse, 8) << '\n';
            std::cout << "Pooled MAE: " << formatFloating(pooled_mae, 8) << '\n';
            std::cout << "Pooled mean error: "
                      << formatFloating(pooled_mean_error, 8) << '\n';
            std::cout << "Pooled PSNR: "
                      << formatFloating(pooled_psnr, 6) << " dB\n";
            std::cout << "Mean finite per-image PSNR: "
                      << formatFloating(mean_finite_psnr, 6) << " dB\n";
            std::cout << "Mean 8x8-window SSIM: "
                      << formatFloating(mean_ssim, 8) << '\n';
            std::cout << "Maximum absolute error: "
                      << overall_maximum_error << '\n';
            std::cout << "Exact pixels: "
                      << formatFloating(exact_percent, 6) << "%\n";
            std::cout << "Pixels within +/-1: "
                      << formatFloating(within_one_percent, 6) << "%\n";
            std::cout << "Pixels within +/-2: "
                      << formatFloating(within_two_percent, 6) << "%\n";

            if (!lowest_psnr_image.empty()) {
                std::cout << "Lowest finite PSNR: "
                          << lowest_psnr_image << " ("
                          << formatFloating(lowest_psnr, 6) << " dB)\n";
            }

            if (!highest_psnr_image.empty()) {
                std::cout << "Highest finite PSNR: "
                          << highest_psnr_image << " ("
                          << formatFloating(highest_finite_psnr, 6)
                          << " dB)\n";
            }

            std::cout << "CSV written to: " << results_path << '\n';
        }

        return failed_images == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Image comparison failed: " << error.what() << '\n';
        return 1;
    }
}

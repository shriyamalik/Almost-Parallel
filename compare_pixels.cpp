// =============================================================================
// gsc2_compare - decompress .gsc2 files and compare them against the originals
// =============================================================================
//
// WHAT THIS PROGRAM IS FOR
// ------------------------
// Pipeline turns a .pgm image into a .gsc2 file. This program does the
// reverse, then measures the difference between what went in and what came out.
//
//
//
// THE FIVE STEPS, IN ORDER
// ------------------------
// Compression did this:
//
//     pixels -> subtract 128 -> DCT -> quantise -> zigzag -> RLE -> pack bits
//
// So decompression does the exact reverse:
//
//     unpack bits -> undo RLE -> undo zigzag -> dequantise -> inverse DCT
//                 -> add 128 back -> pixels
//
//
// FOUR FAMILIES OF MEASUREMENT
// ----------------------------
// (A) PIXEL METRICS - MSE, RMSE, PSNR, MAE, SSIM and friends. How different do
//     the two images look overall.
//
// (B) COEFFICIENT METRICS - the most important section for a hardware project.
//     Pixel metrics blend two completely different things: the loss
//     that quantisation introduces ON PURPOSE, and any numerical error your
//     FPGA DCT introduces BY MISTAKE. This section separates them. We run an
//     exact double-precision DCT and quantiser on the original image, then
//     compare that reference against the quantised coefficients your hardware
//     actually produced (which we can recover, because they are exactly what
//     the .gsc2 file stores). A handful of differences of +/-1 is normal
//     rounding. Thousands, or any difference larger than 1, means your
//     fixed-point widths are too narrow.
//
// (C) BLOCKING ARTIFACTS - the characteristic failure of any 8x8 DCT codec is
//     visible seams at the block edges. PSNR barely notices them because they
//     affect few pixels; the eye notices immediately. We measure the average
//     brightness step across block boundaries against the step inside blocks.
//
// (D) PER-BLOCK PSNR - one catastrophically broken block barely moves an
//     average taken over 4096 blocks. We report the worst block and where it
//     is, which catches sporadic bugs that image-wide metrics hide.
//
//
// AN IMPORTANT EXPECTATION
// ------------------------
// The decoded image will NOT be identical to the original, and that is correct.
// Quantisation throws information away by design. The coefficient metrics in
// section (B) are the ones that should come out near-perfect.
//
//
// HOW TO BUILD AND RUN
// --------------------
//     g++ -std=c++17 -O2 -o gsc2_compare gsc2_compare.cpp
//
//     ./gsc2_compare images/ images_output/ decoded/       <- whole folders
//     ./gsc2_compare images/ images_output/                <- no decoded output
//     ./gsc2_compare a.pgm a.gsc2 a_decoded.pgm            <- one image
//
// Two CSVs are written: quality_metrics.csv and coefficient_metrics.csv.
// =============================================================================

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int BLOCK_SIZE = 8;
constexpr int BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE;   // 64
constexpr int LEVEL_SHIFT = 128;

// -----------------------------------------------------------------------------
// THE ZIGZAG TABLE
//
// Position (row, column) inside a block has natural index row*8 + column. The
// DCT concentrates important information in the top-left corner, so reading the
// block diagonally from there tends to produce long runs of trailing zeros,
// which the run-length encoder compresses well.
//
// Read it as: "zigzag slot k holds the coefficient whose natural index is
// ZIGZAG_ORDER[k]". This MUST match the encoder's table exactly.
// -----------------------------------------------------------------------------
constexpr std::array<std::uint8_t, 64> ZIGZAG_ORDER = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// =============================================================================
// SECTION 1: A PLACE TO KEEP AN IMAGE
// =============================================================================

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    std::uint8_t at(int x, int y) const {
        return pixels[static_cast<std::size_t>(y) * width + x];
    }
    std::uint8_t& at(int x, int y) {
        return pixels[static_cast<std::size_t>(y) * width + x];
    }
};

// =============================================================================
// SECTION 2: READING AND WRITING PGM FILES
//
// A binary P5 PGM is a tiny text header then raw pixel bytes:
//     P5 / 512 512 / 255 / <262144 bytes>
// Whitespace is flexible and '#' starts a comment, hence the fussy parser.
// =============================================================================

std::string read_pgm_token(std::istream& input) {
    while (true) {
        const int next = input.peek();
        if (next == EOF) {
            throw std::runtime_error("Unexpected end of PGM header.");
        }
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
    char character = '\0';
    while (input.get(character)) {
        if (std::isspace(static_cast<unsigned char>(character))) {
            if (character == '\r' && input.peek() == '\n') {
                input.get();
            }
            break;
        }
        token.push_back(character);
    }

    if (token.empty()) {
        throw std::runtime_error("Invalid or empty PGM header token.");
    }
    return token;
}

int parse_positive_int(const std::string& text, const std::string& field) {
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid PGM " + field + ": " + text);
    }
    if (consumed != text.size() || value <= 0) {
        throw std::runtime_error("Invalid PGM " + field + ": " + text);
    }
    return value;
}

GrayImage load_pgm(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open PGM: " + path.string());
    }

    if (read_pgm_token(input) != "P5") {
        throw std::runtime_error(
            "Only binary P5 PGM is supported: " + path.string()
        );
    }

    GrayImage image;
    image.width  = parse_positive_int(read_pgm_token(input), "width");
    image.height = parse_positive_int(read_pgm_token(input), "height");
    const int maximum = parse_positive_int(read_pgm_token(input), "maximum");

    if (maximum != 255) {
        throw std::runtime_error(
            "PGM maximum value must be 255: " + path.string()
        );
    }

    image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
    input.read(
        reinterpret_cast<char*>(image.pixels.data()),
        static_cast<std::streamsize>(image.pixels.size())
    );

    if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        throw std::runtime_error("Incomplete PGM pixel data: " + path.string());
    }
    return image;
}

void save_pgm(const fs::path& path, const GrayImage& image) {
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not create PGM: " + path.string());
    }

    output << "P5\n" << image.width << ' ' << image.height << "\n255\n";
    output.write(
        reinterpret_cast<const char*>(image.pixels.data()),
        static_cast<std::streamsize>(image.pixels.size())
    );

    if (!output) {
        throw std::runtime_error("Failed writing PGM: " + path.string());
    }
}

// =============================================================================
// SECTION 3: THE .gsc2 FILE LAYOUT
//
//   [ header, header_bytes long ] [ block 0 ] [ block 1 ] ... [ block N-1 ]
//
// Header, by byte offset:
//     0   "GSC2"                     4 bytes, magic marker
//     4   version                    2
//     6   header_bytes               2, total size of this header
//     8   width                      4
//     12  height                     4
//     16  block_size                 2, always 8
//     18  max_ac_bytes_per_block     2, the encoder's per-block budget
//     20  num_blocks                 4
//     24  compact_size_bytes         4, total file size
//     28  status                     4, 0 means the encoder succeeded
//     32  quantisation matrix       64
//     96  reserved padding up to header_bytes
//
// Each block:
//     dc_value    2 bytes, signed, the already-quantised DC coefficient
//     bit_count   2 bytes, how many AC bits follow
//     ac_data     ceil(bit_count / 8) bytes
//
//
// All multi-byte numbers are little-endian.
// =============================================================================

struct Gsc2Header {
    std::uint16_t version = 0;
    std::uint16_t header_bytes = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t block_size = 0;
    std::uint16_t max_ac_bytes_per_block = 0;
    std::uint32_t num_blocks = 0;
    std::uint32_t compact_size_bytes = 0;
    std::uint32_t status = 0;
    std::array<std::uint8_t, 64> quantisation_matrix{};
};

// Little-endian: p[0] is the LOW byte.
std::uint16_t read_u16_le(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t read_u32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])         |
           (static_cast<std::uint32_t>(p[1]) << 8)  |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::vector<std::uint8_t> read_entire_file(const fs::path& path) {
    // ios::ate starts at the end, so tellg() immediately gives the file size.
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    const std::streamsize size = input.tellg();
    input.seekg(0);

    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(data.data()), size);

    if (input.gcount() != size) {
        throw std::runtime_error("Short read on: " + path.string());
    }
    return data;
}

Gsc2Header parse_header(const std::vector<std::uint8_t>& data,
                        const fs::path& path) {
    if (data.size() < 36) {
        throw std::runtime_error("File too small to be GSC2: " + path.string());
    }
    if (std::memcmp(data.data(), "GSC2", 4) != 0) {
        throw std::runtime_error("Bad GSC2 magic in: " + path.string());
    }

    Gsc2Header header;
    header.version                = read_u16_le(&data[4]);
    header.header_bytes           = read_u16_le(&data[6]);
    header.width                  = read_u32_le(&data[8]);
    header.height                 = read_u32_le(&data[12]);
    header.block_size             = read_u16_le(&data[16]);
    header.max_ac_bytes_per_block = read_u16_le(&data[18]);
    header.num_blocks             = read_u32_le(&data[20]);
    header.compact_size_bytes     = read_u32_le(&data[24]);
    header.status                 = read_u32_le(&data[28]);

    if (header.header_bytes < 96 || header.header_bytes > data.size()) {
        throw std::runtime_error(
            "Implausible GSC2 header length in: " + path.string()
        );
    }
    if (header.block_size != BLOCK_SIZE) {
        throw std::runtime_error(
            "Only 8x8 blocks are supported, file says " +
            std::to_string(header.block_size)
        );
    }

    std::memcpy(header.quantisation_matrix.data(), &data[32], 64);

    // Dequantisation multiplies by these. A zero means the encoder divided by
    // zero, so the file is corrupt.
    for (const auto q : header.quantisation_matrix) {
        if (q == 0) {
            throw std::runtime_error(
                "Quantisation matrix contains a zero in: " + path.string()
            );
        }
    }

    const std::uint32_t expected_blocks =
        (header.width / BLOCK_SIZE) * (header.height / BLOCK_SIZE);
    if (header.num_blocks != expected_blocks) {
        throw std::runtime_error(
            "Block count " + std::to_string(header.num_blocks) +
            " does not match " + std::to_string(header.width) + "x" +
            std::to_string(header.height) + " (expected " +
            std::to_string(expected_blocks) + ")."
        );
    }
    if (header.status != 0) {
        throw std::runtime_error(
            "GSC2 status field reports encoder failure: " +
            std::to_string(header.status)
        );
    }

    return header;
}

// =============================================================================
// SECTION 4: READING INDIVIDUAL BITS
//
// AC data is a bit stream, not a byte stream: a symbol is 8 bits but the
// amplitude after it is 1 to 15 bits, so nothing lines up with byte boundaries.
// The encoder wrote most-significant-bit first, so stream bit 0 is the TOP bit
// (0x80) of byte 0. This reader must match or everything decodes as noise.
// =============================================================================
class BitReader {
public:
    BitReader(const std::uint8_t* data, std::uint32_t bit_limit)
        : data_(data), bit_position_(0), bit_limit_(bit_limit) {}

    bool read(unsigned int bits, std::uint16_t& value) {
        if (bit_position_ + bits > bit_limit_) {
            return false;
        }

        std::uint32_t accumulator = 0;
        for (unsigned int i = 0; i < bits; ++i) {
            const std::uint32_t absolute = bit_position_ + i;
            const std::uint8_t byte = data_[absolute >> 3U];   // / 8, which byte
            const unsigned int shift = 7U - (absolute & 7U);   // % 8, which bit
            accumulator = (accumulator << 1U) | ((byte >> shift) & 1U);
        }

        bit_position_ += bits;
        value = static_cast<std::uint16_t>(accumulator);
        return true;
    }

    bool exhausted() const { return bit_position_ >= bit_limit_; }

private:
    const std::uint8_t* data_;
    std::uint32_t bit_position_;
    std::uint32_t bit_limit_;
};

// -----------------------------------------------------------------------------
// PACKED BITS BACK TO A SIGNED NUMBER
//
// A value is stored in the smallest number of bits that holds its magnitude;
// `size` is that count. Positives are stored as-is. Negatives are stored as
// (value + 2^size - 1). Telling them apart: a positive always has its top bit
// set, which is what made `size` bits necessary; shifted negatives never do.
//
//   size = 3 covers -7..-4 and 4..7
//     stored 100..111 (4..7) -> positive, use as-is
//     stored 000..011 (0..3) -> negative, subtract 7 to get -7..-4
// -----------------------------------------------------------------------------
std::int16_t decode_amplitude(std::uint16_t encoded, unsigned int size) {
    const std::uint32_t sign_threshold = 1U << (size - 1U);

    if (encoded >= sign_threshold) {
        return static_cast<std::int16_t>(encoded);
    }

    const std::uint32_t mask = (1U << size) - 1U;
    return static_cast<std::int16_t>(
        static_cast<std::int32_t>(encoded) - static_cast<std::int32_t>(mask)
    );
}

// =============================================================================
// SECTION 5: THE DCT, FORWARD AND INVERSE
//
// The forward DCT describes an 8x8 block as a recipe: how much of each of 64
// standard wave patterns to mix. The inverse does the mixing and gets pixels.
//
// The textbook 2D form is a quadruple loop, 4096 multiplies per block. It
// separates into two 1D passes for the same answer in 1024 multiplies, which is
// what we do. (This separation is exactly the optimisation your CPU-only DCT
// baseline is missing, and why it costs 161 ms where it should cost ~10.)
//
// The shared building block is
//
//     basis[x][u] = C(u)/2 * cos((2x+1) * u * pi / 16),  C(0)=1/sqrt(2) else 1
//
// which is the orthonormal DCT-II matrix. Being orthogonal, the SAME table
// serves both directions, just contracted over the other index. We build it
// once at startup rather than calling cos() millions of times.
// =============================================================================
struct DctBasis {
    std::array<std::array<double, BLOCK_SIZE>, BLOCK_SIZE> table{};

    DctBasis() {
        const double pi = 3.14159265358979323846;
        for (int x = 0; x < BLOCK_SIZE; ++x) {
            for (int u = 0; u < BLOCK_SIZE; ++u) {
                const double scale = (u == 0) ? (1.0 / std::sqrt(2.0)) : 1.0;
                table[x][u] =
                    0.5 * scale * std::cos((2.0 * x + 1.0) * u * pi / 16.0);
            }
        }
    }
};

const DctBasis& dct_basis() {
    static const DctBasis basis;   // built on first use, then reused
    return basis;
}

// INVERSE: coefficients (natural order, index v*8+u) -> samples (index y*8+x)
void inverse_dct_8x8(const double* coefficients, double* output) {
    const auto& basis = dct_basis().table;
    double intermediate[BLOCK_ELEMENTS];

    // Pass 1 (horizontal): for each frequency row v, turn frequency u into x.
    for (int v = 0; v < BLOCK_SIZE; ++v) {
        for (int x = 0; x < BLOCK_SIZE; ++x) {
            double sum = 0.0;
            for (int u = 0; u < BLOCK_SIZE; ++u) {
                sum += basis[x][u] * coefficients[v * BLOCK_SIZE + u];
            }
            intermediate[v * BLOCK_SIZE + x] = sum;
        }
    }

    // Pass 2 (vertical): for each column x, turn frequency v into y.
    for (int x = 0; x < BLOCK_SIZE; ++x) {
        for (int y = 0; y < BLOCK_SIZE; ++y) {
            double sum = 0.0;
            for (int v = 0; v < BLOCK_SIZE; ++v) {
                sum += basis[y][v] * intermediate[v * BLOCK_SIZE + x];
            }
            output[y * BLOCK_SIZE + x] = sum;
        }
    }
}

// FORWARD: samples (index y*8+x) -> coefficients (natural order, index v*8+u)
// This is the exact double-precision reference your FPGA is measured against.
void forward_dct_8x8(const double* samples, double* coefficients) {
    const auto& basis = dct_basis().table;
    double intermediate[BLOCK_ELEMENTS];

    // Pass 1 (horizontal): for each row y, turn x into frequency u.
    for (int y = 0; y < BLOCK_SIZE; ++y) {
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            double sum = 0.0;
            for (int x = 0; x < BLOCK_SIZE; ++x) {
                sum += basis[x][u] * samples[y * BLOCK_SIZE + x];
            }
            intermediate[y * BLOCK_SIZE + u] = sum;
        }
    }

    // Pass 2 (vertical): for each frequency column u, turn y into frequency v.
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        for (int v = 0; v < BLOCK_SIZE; ++v) {
            double sum = 0.0;
            for (int y = 0; y < BLOCK_SIZE; ++y) {
                sum += basis[y][v] * intermediate[y * BLOCK_SIZE + u];
            }
            coefficients[v * BLOCK_SIZE + u] = sum;
        }
    }
}

// The IDCT produces fractional values that can land just outside 0..255,
// because quantisation error pushes near-black below zero or near-white above.
std::uint8_t clamp_to_byte(double value) {
    const long rounded = std::lround(value);
    if (rounded < 0)   return 0;
    if (rounded > 255) return 255;
    return static_cast<std::uint8_t>(rounded);
}

// =============================================================================
// SECTION 6: FULL DECOMPRESSION
//
// BLOCK ORDERING ASSUMPTION: block b sits at row b / (width/8), column
// b % (width/8) - raster order, matching an encoder scanning left to right,
// top to bottom. 
// =============================================================================

struct DecodeStats {
    std::uint32_t blocks_with_eob = 0;
    std::uint32_t total_ac_bits = 0;
    std::uint32_t nonzero_ac_coefficients = 0;
};

// Also fills `quantised_natural` with every block's 64 quantised coefficients
// in natural order. These are the values your FPGA actually produced, and
// section 7B compares them against an exact CPU reference.
GrayImage decompress_gsc2(
    const std::vector<std::uint8_t>& data,
    const Gsc2Header& header,
    const fs::path& path,
    DecodeStats& stats,
    std::vector<std::int16_t>& quantised_natural
) {
    GrayImage image;
    image.width  = static_cast<int>(header.width);
    image.height = static_cast<int>(header.height);
    image.pixels.assign(
        static_cast<std::size_t>(image.width) * image.height, 0
    );

    quantised_natural.assign(
        static_cast<std::size_t>(header.num_blocks) * BLOCK_ELEMENTS, 0
    );

    const int blocks_per_row = image.width / BLOCK_SIZE;
    std::size_t offset = header.header_bytes;

    // Declared outside the loop so we are not reallocating 4096 times.
    std::array<std::int16_t, BLOCK_ELEMENTS> zigzag{};
    std::array<double, BLOCK_ELEMENTS> dequantised{};
    std::array<double, BLOCK_ELEMENTS> spatial{};

    for (std::uint32_t block = 0; block < header.num_blocks; ++block) {

        // ---- this block's small header -------------------------------------
        if (offset + 4 > data.size()) {
            throw std::runtime_error(
                "Truncated GSC2 at block header " + std::to_string(block) +
                " in " + path.string()
            );
        }

        // DC is stored raw and signed, not differenced against the previous
        // block. That is a simplification in your encoder, and it is also why
        // blocks can be decoded independently.
        const std::int16_t dc_value =
            static_cast<std::int16_t>(read_u16_le(&data[offset]));
        const std::uint16_t bit_count = read_u16_le(&data[offset + 2]);
        offset += 4;

        const std::size_t byte_count =
            (static_cast<std::size_t>(bit_count) + 7U) / 8U;

        if (bit_count > header.max_ac_bytes_per_block * 8U) {
            throw std::runtime_error(
                "AC bit count exceeds the per-block slot at block " +
                std::to_string(block)
            );
        }
        if (offset + byte_count > data.size()) {
            throw std::runtime_error(
                "Truncated GSC2 AC payload at block " + std::to_string(block) +
                " in " + path.string()
            );
        }

        stats.total_ac_bits += bit_count;

        // ---- STEP A: unpack bits, undo run-length encoding ------------------
        //
        // Start with all 64 slots zero and fill only the non-zero ones, which
        // is exactly what run-length encoding assumes.
        //
        // Three kinds of 8-bit symbol:
        //   0x00   EOB  - everything to the end of the block is zero, stop
        //   0xF0   ZRL  - sixteen zeros, no value follows
        //   else        - top 4 bits = zeros to skip, bottom 4 = value bit width
        //
        zigzag.fill(0);
        zigzag[0] = dc_value;

        BitReader reader(&data[offset], bit_count);
        offset += byte_count;

        int index = 1;                             // slots 1..63 are AC
        while (index < BLOCK_ELEMENTS && !reader.exhausted()) {
            std::uint16_t symbol = 0;
            if (!reader.read(8, symbol)) {
                throw std::runtime_error(
                    "Truncated symbol in block " + std::to_string(block)
                );
            }

            if (symbol == 0x00U) {                 // EOB
                ++stats.blocks_with_eob;
                break;
            }
            if (symbol == 0xF0U) {                 // ZRL, sixteen zeros
                index += 16;
                continue;
            }

            const unsigned int zero_run = (symbol >> 4U) & 0x0FU;
            const unsigned int size     = symbol & 0x0FU;

            if (size == 0) {   // only meaningful inside EOB/ZRL, handled above
                throw std::runtime_error(
                    "Zero amplitude size in a non-EOB symbol, block " +
                    std::to_string(block)
                );
            }

            index += static_cast<int>(zero_run);
            if (index >= BLOCK_ELEMENTS) {
                throw std::runtime_error(
                    "Run length overran block " + std::to_string(block)
                );
            }

            std::uint16_t amplitude = 0;
            if (!reader.read(size, amplitude)) {
                throw std::runtime_error(
                    "Truncated amplitude in block " + std::to_string(block)
                );
            }

            zigzag[index] = decode_amplitude(amplitude, size);
            ++stats.nonzero_ac_coefficients;
            ++index;
        }

        // ---- STEP B: undo zigzag, dequantise, and keep a copy ---------------
        //
        // Zigzag slot k belongs at natural position ZIGZAG_ORDER[k]. While
        // moving it we multiply by that position's quantisation value, undoing
        // the encoder's divide. This is where the loss lives: the encoder
        // divided and rounded, and multiplying back cannot recover the rounding.
        //
        // We also stash the quantised value itself, pre-multiplication, because
        // that is what section 7B needs.
        const std::size_t block_base =
            static_cast<std::size_t>(block) * BLOCK_ELEMENTS;

        for (int k = 0; k < BLOCK_ELEMENTS; ++k) {
            const int natural = ZIGZAG_ORDER[k];
            quantised_natural[block_base + natural] = zigzag[k];
            dequantised[natural] =
                static_cast<double>(zigzag[k]) *
                static_cast<double>(header.quantisation_matrix[natural]);
        }

        // ---- STEP C: inverse DCT, frequencies back to pixels ----------------
        inverse_dct_8x8(dequantised.data(), spatial.data());

        // ---- STEP D: undo the level shift, write into the image -------------
        const int block_row    = static_cast<int>(block) / blocks_per_row;
        const int block_column = static_cast<int>(block) % blocks_per_row;
        const int origin_x     = block_column * BLOCK_SIZE;
        const int origin_y     = block_row * BLOCK_SIZE;

        for (int y = 0; y < BLOCK_SIZE; ++y) {
            for (int x = 0; x < BLOCK_SIZE; ++x) {
                image.at(origin_x + x, origin_y + y) =
                    clamp_to_byte(
                        spatial[y * BLOCK_SIZE + x] +
                        static_cast<double>(LEVEL_SHIFT)
                    );
            }
        }
    }

    return image;
}

// =============================================================================
// SECTION 7A: PIXEL METRICS
//
// Several numbers, because each hides something the others reveal:
//
//   MSE   Average of (difference)^2. Squaring means a few large errors count
//         for far more than many tiny ones. Lower is better.
//   RMSE  Square root of MSE, back in units of pixel levels.
//   PSNR  A log-scale restatement of MSE in decibels, the number everyone
//         quotes. Higher is better; +3 dB is roughly half the error.
//   MAE   Like RMSE but without squaring, so outliers do not dominate. Small
//         MAE with large RMSE means damage concentrated in a few bad pixels.
//   Mean signed error - positives and negatives cancel, so this should sit near
//         zero. If it does not, the whole image drifted brighter or darker,
//         which points at the level shift.
//   SSIM  Compares small patches for brightness, contrast and pattern rather
//         than pixels one at a time. 0 to 1, higher is better. Tracks what a
//         person would notice far better than PSNR.
// =============================================================================

struct QualityMetrics {
    std::string image_name;

    double mse = 0.0;
    double rmse = 0.0;
    double psnr_db = 0.0;
    double mae = 0.0;
    int max_absolute_error = 0;
    double mean_signed_error = 0.0;
    double ssim = 0.0;

    std::size_t total_pixels = 0;
    std::size_t exact_matches = 0;
    std::size_t within_1 = 0;
    std::size_t within_2 = 0;
    std::size_t within_5 = 0;
    std::size_t within_10 = 0;

    int worst_x = 0;
    int worst_y = 0;

    std::uint64_t original_bytes = 0;
    std::uint64_t compressed_bytes = 0;
    double compression_ratio = 0.0;
    double bits_per_pixel = 0.0;

    double percent(std::size_t count) const {
        if (total_pixels == 0) return 0.0;
        return 100.0 * static_cast<double>(count) /
               static_cast<double>(total_pixels);
    }
};

// SSIM over a sliding 8x8 window, averaged across every position. C1 and C2
// only exist to stop the fractions blowing up in flat regions where means and
// variances approach zero. Published SSIM usually uses an 11x11 Gaussian
// window, so do not compare this figure directly against a paper; it is for
// comparing your own runs against each other.
double compute_ssim(const GrayImage& a, const GrayImage& b) {
    constexpr int window = 8;
    const double c1 = (0.01 * 255.0) * (0.01 * 255.0);
    const double c2 = (0.03 * 255.0) * (0.03 * 255.0);
    const double n = window * window;

    if (a.width < window || a.height < window) return 0.0;

    double total = 0.0;
    std::size_t windows = 0;

    for (int y = 0; y + window <= a.height; ++y) {
        for (int x = 0; x + window <= a.width; ++x) {
            // Five running sums give both means, both variances, the covariance.
            double sum_a = 0.0, sum_b = 0.0;
            double sum_aa = 0.0, sum_bb = 0.0, sum_ab = 0.0;

            for (int dy = 0; dy < window; ++dy) {
                for (int dx = 0; dx < window; ++dx) {
                    const double va = a.at(x + dx, y + dy);
                    const double vb = b.at(x + dx, y + dy);
                    sum_a  += va;   sum_b  += vb;
                    sum_aa += va * va;
                    sum_bb += vb * vb;
                    sum_ab += va * vb;
                }
            }

            const double mean_a = sum_a / n;
            const double mean_b = sum_b / n;
            const double var_a  = sum_aa / n - mean_a * mean_a;   // E[X^2]-E[X]^2
            const double var_b  = sum_bb / n - mean_b * mean_b;
            const double cov_ab = sum_ab / n - mean_a * mean_b;

            const double numerator =
                (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2);
            const double denominator =
                (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);

            total += numerator / denominator;
            ++windows;
        }
    }

    return windows == 0 ? 0.0 : total / static_cast<double>(windows);
}

QualityMetrics compare_images(
    const GrayImage& original,
    const GrayImage& decoded,
    std::uint64_t compressed_bytes
) {
    if (original.width != decoded.width || original.height != decoded.height) {
        throw std::runtime_error(
            "Dimension mismatch: original is " +
            std::to_string(original.width) + "x" +
            std::to_string(original.height) + ", decoded is " +
            std::to_string(decoded.width) + "x" +
            std::to_string(decoded.height)
        );
    }

    QualityMetrics m;
    m.total_pixels = original.pixels.size();

    // Accumulate in double: squared errors over 262144 pixels add up fast.
    double squared_error_total = 0.0;
    double absolute_error_total = 0.0;
    double signed_error_total = 0.0;

    for (int y = 0; y < original.height; ++y) {
        for (int x = 0; x < original.width; ++x) {
            const int a = original.at(x, y);
            const int b = decoded.at(x, y);
            const int difference = b - a;         // signed: the sign matters
            const int magnitude = std::abs(difference);

            squared_error_total  += static_cast<double>(difference) * difference;
            absolute_error_total += magnitude;
            signed_error_total   += difference;

            // Cumulative buckets: within 1 also counts as within 2.
            if (magnitude == 0)  ++m.exact_matches;
            if (magnitude <= 1)  ++m.within_1;
            if (magnitude <= 2)  ++m.within_2;
            if (magnitude <= 5)  ++m.within_5;
            if (magnitude <= 10) ++m.within_10;

            if (magnitude > m.max_absolute_error) {
                m.max_absolute_error = magnitude;
                m.worst_x = x;
                m.worst_y = y;
            }
        }
    }

    const double pixel_count = static_cast<double>(m.total_pixels);
    m.mse  = squared_error_total / pixel_count;
    m.rmse = std::sqrt(m.mse);
    m.mae  = absolute_error_total / pixel_count;
    m.mean_signed_error = signed_error_total / pixel_count;

    // PSNR = 10*log10(MAX^2 / MSE), MAX = 255. Guard the divide by zero.
    if (m.mse <= 0.0) {
        m.psnr_db = std::numeric_limits<double>::infinity();
    } else {
        m.psnr_db = 10.0 * std::log10(255.0 * 255.0 / m.mse);
    }

    m.ssim = compute_ssim(original, decoded);

    // Original is one byte per pixel. Note this uses the FULL .gsc2 size
    // including the header, so it reads slightly lower than the encoder's own
    // compression figure.
    m.original_bytes = m.total_pixels;
    m.compressed_bytes = compressed_bytes;
    if (compressed_bytes > 0) {
        m.compression_ratio =
            static_cast<double>(m.original_bytes) /
            static_cast<double>(compressed_bytes);
        m.bits_per_pixel =
            8.0 * static_cast<double>(compressed_bytes) / pixel_count;
    }

    return m;
}

// =============================================================================
// SECTION 7B: COEFFICIENT METRICS  --  THE HARDWARE CORRECTNESS TEST
//
// This is the section that actually tells you whether your FPGA DCT is right.
//
// Everything in 7A measures the codec as a whole, so it cannot distinguish
// "quantisation discarded information, exactly as intended" from "the hardware
// computed the wrong number". Here we isolate the second.
//
// The method:
//   1. Take the ORIGINAL image, subtract 128, run an exact double-precision
//      DCT, divide by the quantisation matrix and round. This is what a perfect
//      implementation would have produced.
//   2. Compare against the quantised coefficients recovered from the .gsc2,
//      which is what your hardware DID produce.
//   3. Report how many differ and by how much.
//
//   A small number of +/-1 differences is expected and harmless: they are
//   values that landed almost exactly halfway between two quantisation levels,
//   where a tiny difference in rounding tips them either way.
//
//   Any difference LARGER than 1 is a genuine numerical error in the hardware
//   DCT - most likely too few integer or fractional bits somewhere in the
//   datapath.
//
//   A large mismatch count concentrated at low natural indices (the top-left,
//   low-frequency corner) is more serious than the same count spread across
//   high indices, because low-frequency coefficients carry most of the image.
//
//   A non-zero mean signed difference means the hardware is biased - it is
//   probably truncating where the reference rounds to nearest. That is a real
//   finding worth reporting, not a bug as such.
// =============================================================================

struct CoefficientMetrics {
    std::size_t total_coefficients = 0;
    std::size_t mismatched = 0;
    std::size_t mismatched_by_1 = 0;       // benign rounding disagreement
    std::size_t mismatched_by_more = 0;    // genuine numerical error
    std::size_t dc_mismatches = 0;         // mismatches at natural index 0

    int max_absolute_difference = 0;
    std::uint32_t worst_block = 0;
    int worst_position = 0;                // natural index within that block

    double mean_absolute_difference = 0.0;
    double mean_signed_difference = 0.0;   // non-zero implies a rounding bias

    // Mismatch count per natural position, to see whether errors cluster in the
    // low frequencies (bad) or the high frequencies (less bad).
    std::array<std::size_t, BLOCK_ELEMENTS> mismatches_by_position{};

    double mismatch_percent() const {
        if (total_coefficients == 0) return 0.0;
        return 100.0 * static_cast<double>(mismatched) /
               static_cast<double>(total_coefficients);
    }
};

CoefficientMetrics compare_coefficients(
    const GrayImage& original,
    const Gsc2Header& header,
    const std::vector<std::int16_t>& actual_quantised
) {
    CoefficientMetrics m;
    m.total_coefficients = actual_quantised.size();

    const int blocks_per_row = original.width / BLOCK_SIZE;

    double absolute_total = 0.0;
    double signed_total = 0.0;

    std::array<double, BLOCK_ELEMENTS> samples{};
    std::array<double, BLOCK_ELEMENTS> coefficients{};

    for (std::uint32_t block = 0; block < header.num_blocks; ++block) {
        const int block_row    = static_cast<int>(block) / blocks_per_row;
        const int block_column = static_cast<int>(block) % blocks_per_row;
        const int origin_x     = block_column * BLOCK_SIZE;
        const int origin_y     = block_row * BLOCK_SIZE;

        // Step 1: gather the block and apply the same level shift the encoder
        // applied, so we are transforming identical input.
        for (int y = 0; y < BLOCK_SIZE; ++y) {
            for (int x = 0; x < BLOCK_SIZE; ++x) {
                samples[y * BLOCK_SIZE + x] =
                    static_cast<double>(original.at(origin_x + x, origin_y + y)) -
                    static_cast<double>(LEVEL_SHIFT);
            }
        }

        // Step 2: exact reference DCT.
        forward_dct_8x8(samples.data(), coefficients.data());

        const std::size_t block_base =
            static_cast<std::size_t>(block) * BLOCK_ELEMENTS;

        for (int natural = 0; natural < BLOCK_ELEMENTS; ++natural) {
            // Step 3: quantise, rounding to nearest. If your kernel truncates
            // instead, expect a consistent bias to show up in
            // mean_signed_difference rather than scattered mismatches.
            const double quantiser =
                static_cast<double>(header.quantisation_matrix[natural]);
            const long reference =
                std::lround(coefficients[natural] / quantiser);

            const long actual =
                static_cast<long>(actual_quantised[block_base + natural]);

            const long difference = actual - reference;
            const long magnitude = std::labs(difference);

            absolute_total += static_cast<double>(magnitude);
            signed_total   += static_cast<double>(difference);

            if (magnitude != 0) {
                ++m.mismatched;
                ++m.mismatches_by_position[natural];
                if (natural == 0) {
                    ++m.dc_mismatches;
                }
                if (magnitude == 1) {
                    ++m.mismatched_by_1;
                } else {
                    ++m.mismatched_by_more;
                }
                if (magnitude > m.max_absolute_difference) {
                    m.max_absolute_difference = static_cast<int>(magnitude);
                    m.worst_block = block;
                    m.worst_position = natural;
                }
            }
        }
    }

    if (m.total_coefficients > 0) {
        const double n = static_cast<double>(m.total_coefficients);
        m.mean_absolute_difference = absolute_total / n;
        m.mean_signed_difference = signed_total / n;
    }

    return m;
}

// =============================================================================
// SECTION 7C: BLOCKING ARTIFACTS
//
// Because each 8x8 block is quantised independently, neighbouring blocks can
// end up with slightly different average brightness. The result is a faint grid
// of seams every 8 pixels. PSNR barely reacts - only 1 pixel in 8 sits on a
// boundary - but the eye picks it up immediately, so it deserves its own number.
//
// The method: measure the average brightness step from one pixel to the next,
// separately for steps that CROSS a block boundary (x = 7 to 8, 15 to 16, ...)
// and steps INSIDE a block. Do it for both images.
//
// In the original, both should be about the same, since block boundaries are an
// artificial grid with no relation to real image content. In the decoded image,
// boundary steps growing relative to interior steps is exactly the artifact.
//
// blocking_index = (decoded boundary/interior) / (original boundary/interior)
//
//     around 1.00        no measurable blocking
//     1.00 to 1.20       mild, typical for this compression ratio
//     above 1.20         visible blocking, worth showing in a report
//     above 1.50         severe; check the DC coefficient path specifically,
//                        since DC errors shift a whole block's brightness
// =============================================================================

struct BlockingMetrics {
    double original_boundary_step = 0.0;
    double original_interior_step = 0.0;
    double decoded_boundary_step = 0.0;
    double decoded_interior_step = 0.0;
    double original_ratio = 0.0;
    double decoded_ratio = 0.0;
    double blocking_index = 0.0;
};

// Averages |difference| between horizontally and vertically adjacent pixels,
// split by whether the step crosses a block boundary.
void measure_steps(
    const GrayImage& image,
    double& boundary_step,
    double& interior_step
) {
    double boundary_total = 0.0, interior_total = 0.0;
    std::size_t boundary_count = 0, interior_count = 0;

    // Horizontal neighbours.
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x + 1 < image.width; ++x) {
            const double step = std::abs(
                static_cast<double>(image.at(x + 1, y)) -
                static_cast<double>(image.at(x, y))
            );
            // x % 8 == 7 means the next pixel starts a new block column.
            if ((x % BLOCK_SIZE) == BLOCK_SIZE - 1) {
                boundary_total += step;  ++boundary_count;
            } else {
                interior_total += step;  ++interior_count;
            }
        }
    }

    // Vertical neighbours.
    for (int y = 0; y + 1 < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const double step = std::abs(
                static_cast<double>(image.at(x, y + 1)) -
                static_cast<double>(image.at(x, y))
            );
            if ((y % BLOCK_SIZE) == BLOCK_SIZE - 1) {
                boundary_total += step;  ++boundary_count;
            } else {
                interior_total += step;  ++interior_count;
            }
        }
    }

    boundary_step = boundary_count ? boundary_total / boundary_count : 0.0;
    interior_step = interior_count ? interior_total / interior_count : 0.0;
}

BlockingMetrics compare_blocking(
    const GrayImage& original,
    const GrayImage& decoded
) {
    BlockingMetrics m;

    measure_steps(original, m.original_boundary_step, m.original_interior_step);
    measure_steps(decoded,  m.decoded_boundary_step,  m.decoded_interior_step);

    // Guard against a perfectly flat image, where interior steps are zero.
    m.original_ratio = (m.original_interior_step > 0.0)
        ? m.original_boundary_step / m.original_interior_step : 0.0;
    m.decoded_ratio = (m.decoded_interior_step > 0.0)
        ? m.decoded_boundary_step / m.decoded_interior_step : 0.0;

    // Dividing by the original's own ratio cancels out any natural tendency the
    // image already had for edges to fall on the 8-pixel grid.
    m.blocking_index = (m.original_ratio > 0.0)
        ? m.decoded_ratio / m.original_ratio : 0.0;

    return m;
}

// =============================================================================
// SECTION 7D: PER-BLOCK PSNR
//
// A whole-image PSNR is an average over 4096 blocks, so one completely broken
// block moves it almost imperceptibly. Computing PSNR per block and reporting
// the worst catches sporadic failures - a bit-packing edge case that only fires
// on certain data, say - that image-wide numbers hide completely.
//
// Report the worst block WITH its pixel coordinates so you can go and look at
// that exact spot in the decoded PGM.
// =============================================================================

struct BlockPsnrMetrics {
    double mean_block_psnr = 0.0;
    double worst_block_psnr = 0.0;
    std::uint32_t worst_block_index = 0;
    int worst_block_x = 0;                 // top-left pixel of that block
    int worst_block_y = 0;
    std::size_t blocks_below_30db = 0;
    std::size_t blocks_below_25db = 0;
    std::size_t blocks_below_20db = 0;     // almost certainly broken
    std::size_t total_blocks = 0;
};

BlockPsnrMetrics compare_per_block(
    const GrayImage& original,
    const GrayImage& decoded
) {
    BlockPsnrMetrics m;
    m.worst_block_psnr = std::numeric_limits<double>::infinity();

    const int blocks_per_row = original.width / BLOCK_SIZE;
    const int blocks_per_column = original.height / BLOCK_SIZE;

    double psnr_total = 0.0;
    std::size_t counted = 0;   // blocks with finite PSNR, for a sane mean

    for (int block_row = 0; block_row < blocks_per_column; ++block_row) {
        for (int block_column = 0; block_column < blocks_per_row; ++block_column) {

            const int origin_x = block_column * BLOCK_SIZE;
            const int origin_y = block_row * BLOCK_SIZE;

            double squared_total = 0.0;
            for (int y = 0; y < BLOCK_SIZE; ++y) {
                for (int x = 0; x < BLOCK_SIZE; ++x) {
                    const double difference =
                        static_cast<double>(decoded.at(origin_x + x, origin_y + y)) -
                        static_cast<double>(original.at(origin_x + x, origin_y + y));
                    squared_total += difference * difference;
                }
            }

            const double block_mse = squared_total / BLOCK_ELEMENTS;
            const double block_psnr =
                (block_mse <= 0.0)
                    ? std::numeric_limits<double>::infinity()
                    : 10.0 * std::log10(255.0 * 255.0 / block_mse);

            ++m.total_blocks;

            if (!std::isinf(block_psnr)) {
                psnr_total += block_psnr;
                ++counted;
                if (block_psnr < 30.0) ++m.blocks_below_30db;
                if (block_psnr < 25.0) ++m.blocks_below_25db;
                if (block_psnr < 20.0) ++m.blocks_below_20db;
            }

            if (block_psnr < m.worst_block_psnr) {
                m.worst_block_psnr = block_psnr;
                m.worst_block_index =
                    static_cast<std::uint32_t>(block_row * blocks_per_row +
                                               block_column);
                m.worst_block_x = origin_x;
                m.worst_block_y = origin_y;
            }
        }
    }

    m.mean_block_psnr = counted ? psnr_total / static_cast<double>(counted) : 0.0;
    return m;
}

// =============================================================================
// SECTION 8: PRINTING THE RESULTS
// =============================================================================

void print_metrics(
    const QualityMetrics& m,
    const DecodeStats& stats,
    const CoefficientMetrics& coefficients,
    const BlockingMetrics& blocking,
    const BlockPsnrMetrics& blocks
) {
    std::cout << "\n---------------- " << m.image_name
              << " ----------------\n";

    // ---- (A) pixel metrics --------------------------------------------------
    std::cout << "  [pixel metrics]\n"
              << std::fixed << std::setprecision(4)
              << "    MSE:                   " << m.mse << '\n'
              << "    RMSE:                  " << m.rmse << '\n';

    std::cout << "    PSNR:                  ";
    if (std::isinf(m.psnr_db)) {
        std::cout << "infinite (bit-exact match)\n";
    } else {
        std::cout << std::setprecision(2) << m.psnr_db << " dB\n";
    }

    std::cout << std::setprecision(4)
              << "    Mean absolute error:   " << m.mae << '\n'
              << "    Mean signed error:     " << m.mean_signed_error << '\n'
              << "    Max absolute error:    " << m.max_absolute_error
              << "  (at x=" << m.worst_x << ", y=" << m.worst_y << ")\n"
              << "    SSIM:                  " << m.ssim << '\n'
              << std::setprecision(2)
              << "    Within +/- 1:          " << m.percent(m.within_1) << " %\n"
              << "    Within +/- 2:          " << m.percent(m.within_2) << " %\n"
              << "    Within +/- 5:          " << m.percent(m.within_5) << " %\n"
              << "    Within +/- 10:         " << m.percent(m.within_10) << " %\n"
              << "    Compression ratio:     " << m.compression_ratio << ":1\n"
              << "    Bits per pixel:        " << m.bits_per_pixel << '\n';

    // ---- (B) coefficient metrics -------------------------------------------
    std::cout << "  [coefficient metrics vs exact CPU reference]\n"
              << "    Coefficients:          " << coefficients.total_coefficients << '\n'
              << "    Mismatched:            " << coefficients.mismatched
              << "  (" << std::setprecision(4)
              << coefficients.mismatch_percent() << " %)\n"
              << "      differ by 1:         " << coefficients.mismatched_by_1
              << "   (benign rounding)\n"
              << "      differ by >1:        " << coefficients.mismatched_by_more
              << "   (real numerical error)\n"
              << "    DC mismatches:         " << coefficients.dc_mismatches << '\n'
              << "    Max abs difference:    " << coefficients.max_absolute_difference
              << "  (block " << coefficients.worst_block
              << ", position " << coefficients.worst_position << ")\n"
              << std::setprecision(6)
              << "    Mean abs difference:   " << coefficients.mean_absolute_difference << '\n'
              << "    Mean signed diff:      " << coefficients.mean_signed_difference
              << "   (non-zero implies a rounding bias)\n";

    // The five worst positions, so you can see whether errors cluster in the
    // low frequencies (serious) or the high frequencies (much less so).
    if (coefficients.mismatched > 0) {
        std::vector<std::pair<std::size_t, int>> ranked;
        for (int i = 0; i < BLOCK_ELEMENTS; ++i) {
            if (coefficients.mismatches_by_position[i] > 0) {
                ranked.emplace_back(coefficients.mismatches_by_position[i], i);
            }
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        std::cout << "    Worst positions:       ";
        for (std::size_t i = 0; i < ranked.size() && i < 5; ++i) {
            const int natural = ranked[i].second;
            std::cout << "idx" << natural
                      << "(r" << (natural / BLOCK_SIZE)
                      << "c" << (natural % BLOCK_SIZE)
                      << ")=" << ranked[i].first << "  ";
        }
        std::cout << '\n';
    }

    // ---- (C) blocking artifacts --------------------------------------------
    std::cout << std::setprecision(4)
              << "  [blocking artifacts]\n"
              << "    Original bnd/int:      " << blocking.original_ratio << '\n'
              << "    Decoded  bnd/int:      " << blocking.decoded_ratio << '\n'
              << "    Blocking index:        " << blocking.blocking_index;
    if (blocking.blocking_index > 1.50) {
        std::cout << "   SEVERE\n";
    } else if (blocking.blocking_index > 1.20) {
        std::cout << "   visible\n";
    } else {
        std::cout << "   ok\n";
    }

    // ---- (D) per-block PSNR -------------------------------------------------
    std::cout << std::setprecision(2)
              << "  [per-block PSNR]\n"
              << "    Mean block PSNR:       " << blocks.mean_block_psnr << " dB\n"
              << "    Worst block PSNR:      " << blocks.worst_block_psnr
              << " dB  (block " << blocks.worst_block_index
              << " at x=" << blocks.worst_block_x
              << ", y=" << blocks.worst_block_y << ")\n"
              << "    Blocks below 30 dB:    " << blocks.blocks_below_30db
              << " / " << blocks.total_blocks << '\n'
              << "    Blocks below 25 dB:    " << blocks.blocks_below_25db << '\n'
              << "    Blocks below 20 dB:    " << blocks.blocks_below_20db << '\n';

    // ---- decode statistics --------------------------------------------------
    std::cout << "  [decode statistics]\n"
              << "    Non-zero AC coeffs:    " << stats.nonzero_ac_coefficients << '\n'
              << "    Blocks ending in EOB:  " << stats.blocks_with_eob << '\n';
}

// Everything gathered for one image, so the batch summary can average it.
struct ImageResult {
    QualityMetrics quality;
    CoefficientMetrics coefficients;
    BlockingMetrics blocking;
    BlockPsnrMetrics blocks;
};

void print_summary(const std::vector<ImageResult>& all) {
    if (all.empty()) return;

    double mse = 0.0, psnr = 0.0, mae = 0.0, ssim = 0.0;
    double ratio = 0.0, bpp = 0.0, blocking = 0.0;
    double coefficient_mismatch_percent = 0.0;

    int worst_max = 0;
    int worst_coefficient_difference = 0;
    std::size_t total_mismatched = 0;
    std::size_t total_mismatched_by_more = 0;
    std::size_t total_blocks_below_25 = 0;

    double worst_psnr = std::numeric_limits<double>::infinity();
    std::string worst_psnr_image;
    double worst_block_psnr = std::numeric_limits<double>::infinity();
    std::string worst_block_image;

    for (const auto& r : all) {
        mse   += r.quality.mse;
        mae   += r.quality.mae;
        ssim  += r.quality.ssim;
        ratio += r.quality.compression_ratio;
        bpp   += r.quality.bits_per_pixel;
        blocking += r.blocking.blocking_index;
        coefficient_mismatch_percent += r.coefficients.mismatch_percent();

        if (!std::isinf(r.quality.psnr_db)) {
            psnr += r.quality.psnr_db;   // adding infinity would poison the mean
        }
        if (r.quality.max_absolute_error > worst_max) {
            worst_max = r.quality.max_absolute_error;
        }
        if (r.coefficients.max_absolute_difference > worst_coefficient_difference) {
            worst_coefficient_difference = r.coefficients.max_absolute_difference;
        }

        total_mismatched         += r.coefficients.mismatched;
        total_mismatched_by_more += r.coefficients.mismatched_by_more;
        total_blocks_below_25    += r.blocks.blocks_below_25db;

        // Naming the worst image tells you which one to open and look at.
        if (r.quality.psnr_db < worst_psnr) {
            worst_psnr = r.quality.psnr_db;
            worst_psnr_image = r.quality.image_name;
        }
        if (r.blocks.worst_block_psnr < worst_block_psnr) {
            worst_block_psnr = r.blocks.worst_block_psnr;
            worst_block_image = r.quality.image_name;
        }
    }

    const double n = static_cast<double>(all.size());

    std::cout << "\n================ QUALITY SUMMARY ================\n"
              << "Images compared:              " << all.size() << '\n'
              << std::fixed << std::setprecision(4)
              << "Average MSE:                  " << mse / n << '\n'
              << std::setprecision(2)
              << "Average PSNR:                 " << psnr / n << " dB\n"
              << std::setprecision(4)
              << "Average mean abs error:       " << mae / n << '\n'
              << "Average SSIM:                 " << ssim / n << '\n'
              << std::setprecision(2)
              << "Average compression ratio:    " << ratio / n << ":1\n"
              << "Average bits per pixel:       " << bpp / n << '\n'
              << "Worst max absolute error:     " << worst_max << '\n'
              << "Lowest PSNR:                  " << worst_psnr << " dB ("
              << worst_psnr_image << ")\n";

    std::cout << "\n---- coefficient accuracy (FPGA vs exact CPU) ----\n"
              << "Total mismatched:             " << total_mismatched << '\n'
              << "  of which differ by >1:      " << total_mismatched_by_more << '\n'
              << std::setprecision(4)
              << "Average mismatch rate:        "
              << coefficient_mismatch_percent / n << " %\n"
              << "Worst single difference:      " << worst_coefficient_difference << '\n';

    if (total_mismatched_by_more == 0) {
        std::cout << "VERDICT: hardware DCT matches the reference to within "
                  << "rounding.\n";
    } else {
        std::cout << "VERDICT: " << total_mismatched_by_more
                  << " coefficient(s) differ by more than 1 - check the "
                  << "fixed-point widths in the DCT datapath.\n";
    }

    std::cout << "\n---- artifacts ----\n"
              << std::setprecision(4)
              << "Average blocking index:       " << blocking / n << '\n'
              << std::setprecision(2)
              << "Worst single block PSNR:      " << worst_block_psnr << " dB ("
              << worst_block_image << ")\n"
              << "Blocks below 25 dB, total:    " << total_blocks_below_25 << '\n';
}

// The CSVs are what you want for plotting or dropping into a report.
void save_quality_csv(const fs::path& path, const std::vector<ImageResult>& all) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not create CSV: " + path.string());
    }

    output << "image,mse,rmse,psnr_db,mean_absolute_error,mean_signed_error,"
           << "max_absolute_error,worst_x,worst_y,ssim,"
           << "percent_exact,percent_within_1,percent_within_2,"
           << "percent_within_5,percent_within_10,"
           << "original_bytes,compressed_bytes,compression_ratio,bits_per_pixel,"
           << "blocking_index,original_boundary_ratio,decoded_boundary_ratio,"
           << "mean_block_psnr_db,worst_block_psnr_db,worst_block_index,"
           << "worst_block_x,worst_block_y,blocks_below_30db,blocks_below_25db,"
           << "blocks_below_20db\n";

    for (const auto& r : all) {
        const auto& m = r.quality;
        output << '"' << m.image_name << "\","
               << std::fixed << std::setprecision(6)
               << m.mse << ',' << m.rmse << ',';

        if (std::isinf(m.psnr_db)) output << "inf,";
        else                       output << m.psnr_db << ',';

        output << m.mae << ',' << m.mean_signed_error << ','
               << m.max_absolute_error << ','
               << m.worst_x << ',' << m.worst_y << ','
               << m.ssim << ','
               << m.percent(m.exact_matches) << ','
               << m.percent(m.within_1) << ','
               << m.percent(m.within_2) << ','
               << m.percent(m.within_5) << ','
               << m.percent(m.within_10) << ','
               << m.original_bytes << ',' << m.compressed_bytes << ','
               << m.compression_ratio << ',' << m.bits_per_pixel << ','
               << r.blocking.blocking_index << ','
               << r.blocking.original_ratio << ','
               << r.blocking.decoded_ratio << ','
               << r.blocks.mean_block_psnr << ',';

        if (std::isinf(r.blocks.worst_block_psnr)) output << "inf,";
        else output << r.blocks.worst_block_psnr << ',';

        output << r.blocks.worst_block_index << ','
               << r.blocks.worst_block_x << ',' << r.blocks.worst_block_y << ','
               << r.blocks.blocks_below_30db << ','
               << r.blocks.blocks_below_25db << ','
               << r.blocks.blocks_below_20db << '\n';
    }
}

void save_coefficient_csv(const fs::path& path,
                          const std::vector<ImageResult>& all) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not create CSV: " + path.string());
    }

    output << "image,total_coefficients,mismatched,mismatch_percent,"
           << "mismatched_by_1,mismatched_by_more_than_1,dc_mismatches,"
           << "max_absolute_difference,worst_block,worst_position,"
           << "mean_absolute_difference,mean_signed_difference\n";

    for (const auto& r : all) {
        const auto& c = r.coefficients;
        output << '"' << r.quality.image_name << "\","
               << std::fixed << std::setprecision(6)
               << c.total_coefficients << ',' << c.mismatched << ','
               << c.mismatch_percent() << ','
               << c.mismatched_by_1 << ',' << c.mismatched_by_more << ','
               << c.dc_mismatches << ','
               << c.max_absolute_difference << ','
               << c.worst_block << ',' << c.worst_position << ','
               << c.mean_absolute_difference << ','
               << c.mean_signed_difference << '\n';
    }
}

// =============================================================================
// SECTION 9: TYING IT TOGETHER
// =============================================================================

ImageResult process_pair(
    const fs::path& original_path,
    const fs::path& compressed_path,
    const fs::path& decoded_path,
    bool write_decoded,
    DecodeStats& stats
) {
    const GrayImage original = load_pgm(original_path);
    const std::vector<std::uint8_t> data = read_entire_file(compressed_path);
    const Gsc2Header header = parse_header(data, compressed_path);

    // Catch wrongly paired files before producing a page of meaningless numbers.
    if (static_cast<int>(header.width) != original.width ||
        static_cast<int>(header.height) != original.height) {
        throw std::runtime_error(
            "GSC2 dimensions do not match the original PGM for " +
            original_path.filename().string()
        );
    }

    std::vector<std::int16_t> quantised_natural;
    const GrayImage decoded =
        decompress_gsc2(data, header, compressed_path, stats, quantised_natural);

    if (write_decoded) {
        save_pgm(decoded_path, decoded);
    }

    ImageResult result;
    result.quality = compare_images(original, decoded, data.size());
    result.quality.image_name = original_path.filename().string();
    result.coefficients = compare_coefficients(original, header, quantised_natural);
    result.blocking = compare_blocking(original, decoded);
    result.blocks = compare_per_block(original, decoded);
    return result;
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

void print_usage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  Single image:\n    " << program
        << " <original.pgm> <compressed.gsc2> [decoded_output.pgm]\n\n"
        << "  Batch:\n    " << program
        << " <originals_folder> <compressed_folder> [decoded_folder]\n\n"
        << "Examples:\n"
        << "  " << program << " images/t005.pgm images_output/t005.gsc2 decoded/t005.pgm\n"
        << "  " << program << " images/ images_output/ decoded/\n";
}

} // namespace

// =============================================================================
// HOW TO READ THE OUTPUT
// ---------------------
// Normal, healthy results for this codec at roughly 4:1:
//
//     PSNR                      32 to 38 dB
//     SSIM                      above 0.95
//     Mean signed error         very close to 0
//     Max abs error             tens, usually on a sharp edge
//     Coefficients differing    a small percentage, ALL of them by exactly 1
//     Blocking index            1.00 to 1.20
//     Worst block PSNR          within about 10 dB of the image average
//
// Signs something is actually WRONG rather than merely lossy:
//
//     Any coefficient differing by more than 1
//         A genuine numerical error in the FPGA DCT. Most likely too few
//         integer or fractional bits somewhere in the datapath. This is the
//         single most diagnostic number the tool produces.
//
//     Non-zero mean signed coefficient difference
//         The hardware is biased - probably truncating where the reference
//         rounds to nearest. Worth reporting; not necessarily a defect.
//
//     Coefficient errors clustered at low natural indices
//         Worse than the same count at high indices, because low-frequency
//         coefficients carry most of the image.
//
//     Blocking index above 1.20
//         Visible seams. If it is above 1.50, look at the DC path specifically,
//         since a DC error shifts a whole block's brightness at once.
//
//     Worst block PSNR far below the mean, or any block under 20 dB
//         Sporadic corruption. Open the decoded PGM at the printed coordinates.
//
//     PSNR below 25 dB, or SSIM below 0.9
//         Something is broken at the image level, not just quantisation.
//
//     Mean signed PIXEL error far from zero
//         The +128 level shift is being applied wrongly somewhere.
//
//     The image looks like shuffled tiles
//         The block ordering assumption in decompress_gsc2 does not match how
//         your kernel emits blocks.
// =============================================================================

int main(int argc, char** argv) {
    // Two arguments = compare only. Three = also write decoded images.
    if (argc != 3 && argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const fs::path original_argument = argv[1];
    const fs::path compressed_argument = argv[2];
    const bool write_decoded = (argc == 4);
    const fs::path decoded_argument = write_decoded ? fs::path(argv[3])
                                                    : fs::path();

    try {
        std::vector<ImageResult> all_results;
        fs::path csv_directory = ".";

        // Batch or single is decided purely by whether argument 1 is a folder.
        if (fs::is_directory(original_argument)) {

            if (!fs::is_directory(compressed_argument)) {
                throw std::runtime_error(
                    "First argument is a folder, so the second must be too."
                );
            }

            // Look up compressed files by filename stem, so pairing t005.pgm
            // with t005.gsc2 does not depend on directory iteration order.
            std::map<std::string, fs::path> compressed_by_stem;
            for (const auto& entry : fs::directory_iterator(compressed_argument)) {
                if (fs::is_regular_file(entry.path()) &&
                    to_lower(entry.path().extension().string()) == ".gsc2") {
                    compressed_by_stem[entry.path().stem().string()] = entry.path();
                }
            }

            std::vector<fs::path> originals;
            for (const auto& entry : fs::directory_iterator(original_argument)) {
                if (fs::is_regular_file(entry.path()) &&
                    to_lower(entry.path().extension().string()) == ".pgm") {
                    originals.push_back(entry.path());
                }
            }
            std::sort(originals.begin(), originals.end());   // stable report order

            if (originals.empty()) {
                throw std::runtime_error(
                    "No .pgm files in " + original_argument.string()
                );
            }

            if (write_decoded) {
                fs::create_directories(decoded_argument);
                csv_directory = decoded_argument;
            }

            std::cout << "Comparing " << originals.size() << " image(s)\n"
                      << "Originals:  " << original_argument << '\n'
                      << "Compressed: " << compressed_argument << '\n';
            if (write_decoded) {
                std::cout << "Decoded to: " << decoded_argument << '\n';
            }

            std::size_t skipped = 0;

            for (const auto& original_path : originals) {
                const std::string stem = original_path.stem().string();
                const auto found = compressed_by_stem.find(stem);

                if (found == compressed_by_stem.end()) {
                    std::cerr << "\nSKIPPED " << original_path.filename().string()
                              << ": no matching .gsc2 file.\n";
                    ++skipped;
                    continue;
                }

                const fs::path decoded_path =
                    write_decoded ? decoded_argument / (stem + "_decoded.pgm")
                                  : fs::path();

                // One bad file should not abandon the other twenty-nine.
                DecodeStats stats;
                try {
                    ImageResult result = process_pair(
                        original_path, found->second,
                        decoded_path, write_decoded, stats
                    );
                    print_metrics(result.quality, stats, result.coefficients,
                                  result.blocking, result.blocks);
                    all_results.push_back(std::move(result));
                } catch (const std::exception& error) {
                    std::cerr << "\nFAILED " << original_path.filename().string()
                              << ": " << error.what() << '\n';
                    ++skipped;
                }
            }

            print_summary(all_results);
            if (skipped != 0) {
                std::cout << "Skipped or failed:            " << skipped << '\n';
            }

        } else {
            // Single-image mode.
            if (write_decoded && decoded_argument.has_parent_path()) {
                csv_directory = decoded_argument.parent_path();
            }

            DecodeStats stats;
            ImageResult result = process_pair(
                original_argument, compressed_argument,
                decoded_argument, write_decoded, stats
            );
            print_metrics(result.quality, stats, result.coefficients,
                          result.blocking, result.blocks);
            all_results.push_back(std::move(result));
        }

        if (!all_results.empty()) {
            const fs::path quality_csv =
                csv_directory / "quality_metrics.csv";
            const fs::path coefficient_csv =
                csv_directory / "coefficient_metrics.csv";

            save_quality_csv(quality_csv, all_results);
            save_coefficient_csv(coefficient_csv, all_results);

            std::cout << "\nQuality CSV:     " << quality_csv << '\n'
                      << "Coefficient CSV: " << coefficient_csv << '\n';
        }

        return EXIT_SUCCESS;

    } catch (const std::exception& error) {
        std::cerr << "\nERROR: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

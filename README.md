# FPGA Acceleration of JPEG-Style Grayscale Image Compression on the Kria KV260

## Project Overview

This project investigates the acceleration of a JPEG-style grayscale image compression pipeline using programmable logic on the AMD Kria KV260 board. The goal is to identify the compute-intensive stages of an image compression algorithm, implement those stages in hardware using High-Level Synthesis (HLS), and compare the performance against a CPU-only software baseline.

The selected algorithm is a simplified JPEG-style compression pipeline based on:

1. 8×8 block processing
2. Discrete Cosine Transform (DCT)
3. Quantisation
4. Zigzag scan
5. Run-Length Encoding (RLE) of zero coefficients

The project focuses on accelerating the compression-side kernel using FPGA logic while keeping image loading, decompression, verification, and metric calculation in software.

This project does not aim to implement a full JPEG file encoder. Instead, it accelerates the core JPEG-style transform compression stages that are suitable for FPGA implementation and performance evaluation.

---

## Team Members

| Name          | zID      | Role / Responsibility                                       |
| ------------- | -------- | ----------------------------------------------------------- |
| Team Member 1 | zXXXXXXX | Project coordination, report, integration                   |
| Team Member 2 | zXXXXXXX | CPU baseline and verification                               |
| Team Member 3 | zXXXXXXX | HLS kernel implementation                                   |
| Team Member 4 | zXXXXXXX | KV260 deployment, benchmarking, energy/performance analysis |

---

## Problem Outline

Image compression reduces the amount of data required to store or transmit an image. In JPEG-style compression, the image is divided into small 8×8 blocks, and each block is transformed from the spatial domain into the frequency domain using the Discrete Cosine Transform.

The DCT stage is computationally expensive because it involves repeated multiply-accumulate operations over every 8×8 block. This makes it a good candidate for FPGA acceleration.

The main challenge is to partition the workload between the processor and the programmable logic in a way that improves performance while keeping data transfer and implementation complexity manageable.

The key problems addressed in this project are:

* Accelerating the compute-heavy DCT and quantisation stages.
* Handling simple but realistic image-compression data structures.
* Managing data transfer between the ARM processor and FPGA fabric.
* Measuring kernel-only and end-to-end speedup.
* Evaluating compression ratio and reconstructed image quality.
* Keeping the project scope realistic within the COMP4601 timeline.

---

## Project Aims

The main aims of this project are:

1. Implement a CPU-only baseline for JPEG-style grayscale image compression.
2. Identify the computational bottleneck in the software implementation.
3. Implement an FPGA-accelerated HLS kernel for the compression-side pipeline.
4. Compare CPU-only and FPGA-accelerated performance.
5. Measure compression ratio, zero coefficient percentage, RLE effectiveness, and reconstructed image quality.
6. Evaluate trade-offs between speedup, compression strength, image quality, and hardware resource usage.
7. Demonstrate a working image compression pipeline on the Kria KV260.

---

## Algorithm to be Accelerated

The selected algorithm is a simplified JPEG-style grayscale image compression pipeline.

### Compression Pipeline

```text
Input grayscale image
        ↓
Split image into 8×8 blocks
        ↓
Level shift pixels by subtracting 128
        ↓
Apply 8×8 Discrete Cosine Transform
        ↓
Quantise DCT coefficients
        ↓
Apply zigzag scan
        ↓
Run-Length Encode zero coefficients
        ↓
Compressed block representation
```

### Decompression / Verification Pipeline

```text
Compressed block representation
        ↓
Run-Length decoding
        ↓
Inverse zigzag scan
        ↓
Inverse quantisation
        ↓
Inverse DCT
        ↓
Add 128 and clamp to 0–255
        ↓
Reconstructed image
```

The decompression pipeline is mainly used for verification and quality measurement.

---

## Why This Algorithm Is Suitable for FPGA Acceleration

The 8×8 DCT is suitable for FPGA acceleration because:

* Each 8×8 image block is independent.
* The loops are fixed-size and predictable.
* The computation contains many repeated multiply-add operations.
* The data structure is simple and regular.
* The algorithm can benefit from HLS optimisations such as pipelining, loop unrolling, array partitioning, and fixed-point arithmetic.

The DCT stage is expected to be the main performance bottleneck in the CPU baseline and therefore the main target for acceleration.

---

## Description of Each Stage

### 1. Grayscale Image Input

The input image is treated as a 2D array of 8-bit grayscale pixel values.

```text
0   = black
255 = white
```

For implementation simplicity, images are expected to be stored as raw grayscale arrays or simple PGM-style images.

---

### 2. 8×8 Block Splitting

The image is divided into independent 8×8 blocks.

For example, a 256×256 image contains:

```text
256 / 8 = 32 blocks per row
32 × 32 = 1024 total blocks
```

Each block contains 64 pixels and is processed independently.

---

### 3. Level Shifting

Before applying the DCT, each pixel is shifted from the unsigned range 0–255 to a signed range centred around zero.

```text
shifted_pixel = pixel - 128
```

This changes the approximate range from:

```text
0 to 255
```

to:

```text
-128 to +127
```

This is standard in JPEG-style compression and makes the DCT representation more effective.

---

### 4. Discrete Cosine Transform

The DCT converts an 8×8 block of pixel intensities into an 8×8 block of frequency coefficients.

The top-left coefficient is the DC coefficient, which represents the average brightness of the block.

The remaining coefficients are AC coefficients, which represent increasing levels of detail and frequency content.

Most natural images have most of their important information concentrated in the low-frequency coefficients near the top-left of the DCT matrix.

---

### 5. Quantisation

Quantisation reduces the precision of the DCT coefficients.

```text
quantised_coeff = round(DCT_coeff / quantisation_value)
```

Larger quantisation values cause more coefficients to become zero, especially high-frequency coefficients.

This is where most of the lossy compression occurs.

Different quantisation scales can be tested to compare quality and compression trade-offs.

---

### 6. Zigzag Scan

After quantisation, the 8×8 coefficient matrix is converted into a 1D array using zigzag order.

The zigzag scan places low-frequency coefficients first and high-frequency coefficients later.

This is useful because high-frequency coefficients are often zero after quantisation, so zigzag ordering tends to group zeros near the end of the block.

---

### 7. Run-Length Encoding of Zero Coefficients

After zigzag scan, many AC coefficients are zero.

Instead of storing every zero, Run-Length Encoding stores the number of zeros before each non-zero value.

For example:

```text
Zigzag coefficients:
[52, -4, 3, 0, 0, 0, 5, 0, 0, -2, 0, 0, 0, ...]

Compressed representation:
DC = 52
AC pairs:
(0, -4)
(0, 3)
(3, 5)
(2, -2)
```

Each pair stores:

```text
zero_run = number of zeros before the next non-zero value
value    = next non-zero coefficient
```

For hardware simplicity, each block reserves space for up to 63 RLE pairs because an 8×8 block has 63 AC coefficients.

The actual number of valid pairs is stored separately using `num_pairs`.

---

## Hardware / Software Partition

The project uses a hardware/software co-design approach.

### Software Responsibilities

The CPU handles:

* Image loading
* Input buffer preparation
* CPU baseline implementation
* Launching the FPGA kernel
* Receiving compressed output
* RLE decoding for verification
* Inverse zigzag
* Inverse quantisation
* Inverse DCT
* Reconstructed image generation
* MSE and PSNR calculation
* Compression ratio calculation
* Runtime and speedup reporting

### FPGA Responsibilities

The FPGA kernel handles:

* 8×8 block extraction
* Level shifting
* 2D DCT
* Quantisation
* Zigzag scan
* Zero Run-Length Encoding
* Writing compressed block data to output buffers

---

## Baseline Implementations

The project will compare the FPGA implementation against CPU-only baselines.

### Baseline 1: Naive CPU Direct 2D DCT

This is the slowest baseline.

Characteristics:

* CPU-only implementation
* Direct 2D DCT formula
* Floating-point arithmetic
* Nested loops
* No FPGA acceleration
* No major optimisation

This baseline is useful for showing the benefit of acceleration compared to a simple software implementation.

---

### Baseline 2: Optimised CPU Separable DCT

This is a fairer CPU comparison.

Characteristics:

* CPU-only implementation
* Uses separable DCT
* Applies 1D DCT across rows and then columns
* Uses precomputed cosine constants
* May use more efficient arithmetic than the naive baseline

This baseline is useful for comparing the FPGA implementation against a more realistic optimised software version.

---

## Expected Acceleration

Expected performance improvement depends on image size, arithmetic type, memory transfer overhead, and HLS optimisation quality.

Estimated speedup ranges:

| Comparison                                  |        Expected Speedup |
| ------------------------------------------- | ----------------------: |
| FPGA vs naive CPU direct DCT                | 20× to 100× kernel-only |
| FPGA vs optimised CPU separable DCT         |   5× to 30× kernel-only |
| FPGA end-to-end including transfer overhead |       2× to 15× overall |

The project will report both:

1. Kernel-only runtime
2. End-to-end runtime including data transfer

This is important because data transfer overhead can hide acceleration, especially for small images.

---

## Evaluation Metrics

The project will evaluate performance and quality using the following metrics.

### 1. Runtime

Measured for:

* CPU-only naive baseline
* CPU-only optimised baseline
* FPGA kernel-only runtime
* FPGA end-to-end runtime

---

### 2. Speedup

```text
speedup = CPU runtime / FPGA runtime
```

Both kernel-only and end-to-end speedup will be reported.

---

### 3. Compression Ratio

```text
compression_ratio = original_size / compressed_size
```

Original size:

```text
original_size = width × height bytes
```

Compressed size:

```text
compressed_size =
    num_blocks × sizeof(dc)
  + num_blocks × sizeof(num_pairs)
  + total_pairs × (sizeof(run) + sizeof(value))
```

---

### 4. Mean Squared Error

```text
MSE = average((original_pixel - reconstructed_pixel)^2)
```

Lower MSE means the reconstructed image is closer to the original.

---

### 5. Peak Signal-to-Noise Ratio

```text
PSNR = 10 × log10(255² / MSE)
```

Higher PSNR means better reconstructed image quality.

General guide:

|        PSNR | Quality                      |
| ----------: | ---------------------------- |
|      40 dB+ | Very high quality            |
|    30–40 dB | Good quality                 |
|    25–30 dB | Acceptable with visible loss |
| Below 25 dB | Noticeable degradation       |

---

### 6. Zero Coefficient Percentage

This measures how many quantised DCT coefficients become zero.

A higher zero percentage usually means better RLE compression.

---

### 7. Average RLE Pairs Per Block

This measures how effective the RLE stage is.

Fewer RLE pairs per block means stronger compression.

---

### 8. FPGA Resource Usage

The HLS and implementation reports will be used to record:

* LUT usage
* FF usage
* BRAM usage
* DSP usage
* Latency
* Initiation interval
* Clock frequency

---

### 9. Energy Consumption

If measurable within the available setup, the project will compare energy consumption between:

* CPU-only software execution
* FPGA-accelerated execution

---

## HLS Optimisation Strategy

The FPGA kernel will be optimised using common HLS techniques.

### 1. Loop Pipelining

Loop pipelining will be used to improve throughput and reduce initiation interval.

Target:

```text
II = 1 where practical
```

---

### 2. Loop Unrolling

Small loops, especially loops of size 8 inside the DCT computation, may be unrolled to increase parallelism.

---

### 3. Array Partitioning

Local arrays such as the 8×8 image block, intermediate DCT buffer, and quantised coefficient matrix may be partitioned to allow multiple parallel memory accesses.

Example local arrays:

```cpp
block[8][8]
temp[8][8]
dct[8][8]
quantised[8][8]
zigzag[64]
```

---

### 4. Fixed-Point Arithmetic

The first implementation may use floating-point arithmetic for correctness and simplicity.

The optimised FPGA version should use fixed-point arithmetic to reduce resource usage and improve performance.

---

### 5. Dataflow

If time permits, the kernel can be separated into stages and optimised using dataflow:

```text
load block → DCT → quantisation → zigzag → RLE → store output
```

This allows different stages to overlap and improves throughput.

---

## Investigation So Far

The project investigation has identified that:

* Full JPEG implementation is too complex for the project timeline.
* Huffman coding and JPEG bitstream generation introduce variable-length bit-level complexity.
* A simplified JPEG-style pipeline is more suitable for HLS and the KV260.
* The DCT stage is the most compute-heavy part of the compression pipeline.
* Quantisation creates many zero coefficients.
* Zigzag scan improves the effectiveness of RLE by grouping high-frequency zeros.
* RLE creates variable-length output, so a fixed maximum buffer with `num_pairs` is used for hardware simplicity.
* FPGA acceleration is expected to be most visible for larger images such as 512×512 or 1024×1024.
* Both kernel-only and end-to-end timings should be reported.

---

## Risks and Contingencies

### Risk 1: RLE Produces Variable-Length Output

RLE output size differs from block to block.

Contingency:

* Allocate fixed 63-pair slots per block.
* Store `num_pairs` for each block.
* Report actual compressed size separately from allocated buffer size.

---

### Risk 2: DCT Numerical Mismatch

The CPU reference may use floating-point arithmetic, while the FPGA implementation may use fixed-point arithmetic.

Contingency:

* Start with a CPU floating-point reference.
* Use fixed-point HLS after correctness is confirmed.
* Compare outputs using tolerances.
* Use reconstructed image quality metrics such as MSE and PSNR.

---

### Risk 3: Data Transfer Overhead Hides Speedup

For small images, CPU-FPGA data transfer may dominate runtime.

Contingency:

* Measure both kernel-only and end-to-end runtimes.
* Test larger images such as 512×512 and 1024×1024 if stable.
* Report transfer overhead separately.

---

### Risk 4: Schedule Risk

Implementing DCT, quantisation, zigzag, and RLE fully in FPGA may take longer than expected.

Contingency:

* Minimum target: FPGA acceleration of DCT + quantisation.
* If time permits: add zigzag scan to FPGA.
* If stable: add RLE to FPGA.
* If integration time is tight, zigzag and RLE can remain in software.

---

### Risk 5: Hardware Resource Pressure

Unrolling and parallelising the DCT may use many DSPs, LUTs, and registers.

Contingency:

* Start with a simple working kernel.
* Optimise gradually.
* Compare performance-resource trade-offs.
* Use partial unrolling if complete unrolling is too expensive.

---

## Project Plan and Timeline

### Week 5: Software Baseline and Algorithm Verification

Goals:

* Implement grayscale image loading.
* Implement CPU naive DCT + quantisation.
* Implement zigzag scan and RLE.
* Implement decompression for verification.
* Calculate MSE, PSNR, compression ratio, and runtime.
* Test with small images first.

Deliverables:

* Working CPU-only compressor.
* Working decompressor.
* Initial baseline runtime and quality metrics.

---

### Week 6: HLS Kernel Development

Goals:

* Implement the HLS kernel for DCT + quantisation.
* Add zigzag scan if DCT + quantisation is stable.
* Add RLE if integration is manageable.
* Run HLS C simulation.
* Compare HLS output with CPU reference.
* Begin fixed-point conversion if time permits.

Deliverables:

* HLS kernel passing C simulation.
* Initial HLS synthesis report.
* Latency and resource estimates.

---

### Week 7: KV260 Integration and Testing

Goals:

* Build the FPGA binary.
* Write or adapt host code for KV260 execution.
* Transfer input image buffer to FPGA.
* Retrieve compressed output buffers.
* Verify FPGA output against software reference.
* Measure kernel-only and end-to-end runtime.

Deliverables:

* Working FPGA-accelerated compression pipeline on KV260.
* Runtime comparison between CPU and FPGA.
* Verified reconstructed image output.

---

### Week 8: Optimisation, Evaluation, and Final Report

Goals:

* Apply HLS optimisations such as pipelining, unrolling, array partitioning, and fixed-point arithmetic.
* Test different image sizes.
* Test different quantisation strengths.
* Collect final metrics.
* Prepare graphs, tables, final presentation, demo, and report.

Deliverables:

* Final optimised FPGA implementation.
* Final performance and quality results.
* Final project report and presentation slides.

---

---

## Limitations

This project is intentionally scoped as a simplified JPEG-style compressor.

The following features are not included in the minimum implementation:

* Full JPEG file output
* JPEG headers
* Huffman coding
* Arithmetic coding
* RGB to YCbCr conversion
* Chroma subsampling
* Full entropy-coded bitstream generation
* Dynamic image size support in the first implementation

These features are excluded to keep the project focused on FPGA acceleration of the compute-heavy compression stages.

---

## Possible Extensions

If the main implementation is completed early, possible extensions include:

* Fixed-point optimisation of the DCT.
* Comparison of multiple quantisation scales.
* Support for larger image sizes.
* Software Huffman coding after FPGA RLE.
* Colour image support using YCbCr conversion.
* Chroma subsampling.
* FPGA acceleration of inverse DCT for decompression.
* Streaming implementation using HLS dataflow.

---

## Summary

This project accelerates a simplified JPEG-style grayscale image compression pipeline on the Kria KV260. The main compute-heavy operation is the 8×8 DCT, followed by quantisation, zigzag scan, and zero Run-Length Encoding.

The expected outcome is a working hardware/software image compression system that demonstrates measurable speedup over CPU-only baselines while also reporting compression ratio and reconstructed image quality.

The project is designed to be realistic within the COMP4601 timeline by focusing on the core compression kernel rather than implementing a complete JPEG file format.


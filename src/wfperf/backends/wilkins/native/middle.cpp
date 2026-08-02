// Copyright (c) 2026, University of Florida. All rights reserved.
//
// This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
// group at the University of Florida. See LICENSE in the top-level directory.

#include "module_common.hpp"

int main(int argc, char** argv) {
    const auto task_start = wfperf::Clock::now();
    try {
        wfperf::require(
            argc == 8,
            "usage: middle ITERATIONS SLEEP STAGE OUTPUT_STAGE PARTICLES INPUTS INPUT_FILE");
        wfperf::MpiSession mpi(&argc, &argv);
        const int iterations = wfperf::parse_positive_int(argv[1], "iterations");
        const double sleep_seconds =
            wfperf::parse_nonnegative_double(argv[2], "sleep");
        const int stage = wfperf::parse_nonnegative_int(argv[3], "stage");
        const int output_stage =
            wfperf::parse_positive_int(argv[4], "output stage");
        const std::uint64_t particles =
            wfperf::parse_positive_size(argv[5], "particles");
        const int input_count = wfperf::parse_positive_int(argv[6], "input count");
        const std::string input = argv[7];
        wfperf::require(input.find("{filename}") == std::string::npos,
                        "Wilkins did not resolve the input filename");
        const std::string module_name = "middle" + std::to_string(stage);
        const std::string output =
            wfperf::output_filename(argv[0], module_name, output_stage);

        const std::uint64_t invocations =
            static_cast<std::uint64_t>(iterations) * input_count;
        volatile double checksum = 0.0;
        for (std::uint64_t invocation = 0; invocation < invocations; ++invocation) {
            const auto input_start = wfperf::Clock::now();
            checksum += wfperf::read_particles(input, mpi);
            wfperf::print_timing(
                "Middle" + std::to_string(stage) +
                    " Input H5Fcreate to H5Fclose",
                wfperf::elapsed_ms(input_start));

            wfperf::emulate_compute(sleep_seconds);
            const auto output_start = wfperf::Clock::now();
            wfperf::write_particles(output, particles, invocation, mpi);
            wfperf::print_timing(
                "Middle" + std::to_string(stage) +
                    " Output H5Fcreate to H5Fclose",
                wfperf::elapsed_ms(output_start));
        }
        (void)checksum;

        wfperf::print_timing("Middle" + std::to_string(stage) + " Task",
                             wfperf::elapsed_ms(task_start));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "WfPerf intermediate: " << error.what() << std::endl;
        return 2;
    }
}

// Copyright (c) 2026, University of Florida. All rights reserved.
//
// This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
// group at the University of Florida. See LICENSE in the top-level directory.

#include "module_common.hpp"

int main(int argc, char** argv) {
    const auto task_start = wfperf::Clock::now();
    try {
        wfperf::require(
            argc == 6,
            "usage: con-henson ITERATIONS INPUTS INPUT_FILE FINAL_STAGE SLEEP");
        wfperf::MpiSession mpi(&argc, &argv);
        const int iterations = wfperf::parse_positive_int(argv[1], "iterations");
        const int input_count = wfperf::parse_positive_int(argv[2], "input count");
        const std::string input = argv[3];
        wfperf::require(input.find("{filename}") == std::string::npos,
                        "Wilkins did not resolve the input filename");
        (void)wfperf::parse_nonnegative_int(argv[4], "final stage");
        const double sleep_seconds =
            wfperf::parse_nonnegative_double(argv[5], "sleep");

        const std::uint64_t invocations =
            static_cast<std::uint64_t>(iterations) * input_count;
        volatile double checksum = 0.0;
        for (std::uint64_t invocation = 0; invocation < invocations; ++invocation) {
            wfperf::emulate_compute(sleep_seconds);
            const auto input_start = wfperf::Clock::now();
            checksum += wfperf::read_particles(input, mpi);
            wfperf::print_timing("Consumer H5Fcreate to H5Fclose",
                                 wfperf::elapsed_ms(input_start));
        }
        (void)checksum;

        wfperf::print_timing("Consumer Task", wfperf::elapsed_ms(task_start));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "WfPerf consumer: " << error.what() << std::endl;
        return 2;
    }
}

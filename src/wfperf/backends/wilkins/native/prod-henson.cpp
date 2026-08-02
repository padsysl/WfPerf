// Copyright (c) 2026, University of Florida. All rights reserved.
//
// This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
// group at the University of Florida. See LICENSE in the top-level directory.

#include "module_common.hpp"

int main(int argc, char** argv) {
    const auto task_start = wfperf::Clock::now();
    try {
        wfperf::require(argc == 5,
                        "usage: prod-henson ITERATIONS SLEEP PARTICLES SINKS");
        wfperf::MpiSession mpi(&argc, &argv);
        const int iterations = wfperf::parse_positive_int(argv[1], "iterations");
        const double sleep_seconds =
            wfperf::parse_nonnegative_double(argv[2], "sleep");
        const std::uint64_t particles =
            wfperf::parse_positive_size(argv[3], "particles");
        (void)wfperf::parse_positive_int(argv[4], "sink count");
        const std::string output =
            wfperf::output_filename(argv[0], "prod-henson0", 0);

        for (int iteration = 0; iteration < iterations; ++iteration) {
            wfperf::emulate_compute(sleep_seconds);
            const auto output_start = wfperf::Clock::now();
            wfperf::write_particles(output, particles, iteration, mpi);
            wfperf::print_timing("Producer H5Fcreate to H5Fclose",
                                 wfperf::elapsed_ms(output_start));
        }

        wfperf::print_timing("Producer Task", wfperf::elapsed_ms(task_start));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "WfPerf producer: " << error.what() << std::endl;
        return 2;
    }
}

// Copyright (c) 2026, University of Florida. All rights reserved.
//
// This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
// group at the University of Florida. See LICENSE in the top-level directory.

#ifndef WFPERF_WILKINS_MODULE_COMMON_HPP
#define WFPERF_WILKINS_MODULE_COMMON_HPP

#include <hdf5.h>
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace wfperf {

using Clock = std::chrono::steady_clock;

inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline int parse_positive_int(const char* value, const std::string& name) {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    require(value[consumed] == '\0' && parsed > 0 &&
                parsed <= std::numeric_limits<int>::max(),
            name + " must be a positive integer");
    return static_cast<int>(parsed);
}

inline int parse_nonnegative_int(const char* value, const std::string& name) {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    require(value[consumed] == '\0' && parsed >= 0 &&
                parsed <= std::numeric_limits<int>::max(),
            name + " must be a non-negative integer");
    return static_cast<int>(parsed);
}

inline std::uint64_t parse_positive_size(const char* value, const std::string& name) {
    std::size_t consumed = 0;
    require(value[0] != '-', name + " must be a positive integer");
    const unsigned long long parsed = std::stoull(value, &consumed);
    require(value[consumed] == '\0' && parsed > 0,
            name + " must be a positive integer");
    return static_cast<std::uint64_t>(parsed);
}

inline double parse_nonnegative_double(const char* value, const std::string& name) {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    require(value[consumed] == '\0' && parsed >= 0.0,
            name + " must be a non-negative number");
    return parsed;
}

inline void emulate_compute(double seconds) {
    if (seconds > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    }
}

class MpiSession {
  public:
    MpiSession(int* argc, char*** argv) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            require(MPI_Init(argc, argv) == MPI_SUCCESS, "MPI_Init failed");
            owns_mpi_ = true;
        }
        require(MPI_Comm_rank(MPI_COMM_WORLD, &rank_) == MPI_SUCCESS,
                "MPI_Comm_rank failed");
        require(MPI_Comm_size(MPI_COMM_WORLD, &size_) == MPI_SUCCESS,
                "MPI_Comm_size failed");
    }

    ~MpiSession() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (owns_mpi_ && !finalized) {
            MPI_Finalize();
        }
    }

    int rank() const { return rank_; }
    int size() const { return size_; }

  private:
    bool owns_mpi_ = false;
    int rank_ = 0;
    int size_ = 1;
};

inline void check_hdf5(herr_t status, const std::string& operation) {
    require(status >= 0, operation + " failed");
}

inline hid_t parallel_file_access() {
    const hid_t property = H5Pcreate(H5P_FILE_ACCESS);
    require(property >= 0, "H5Pcreate(H5P_FILE_ACCESS) failed");
    check_hdf5(H5Pset_fapl_mpio(property, MPI_COMM_WORLD, MPI_INFO_NULL),
               "H5Pset_fapl_mpio");
    return property;
}

inline hid_t collective_transfer() {
    const hid_t property = H5Pcreate(H5P_DATASET_XFER);
    require(property >= 0, "H5Pcreate(H5P_DATASET_XFER) failed");
    check_hdf5(H5Pset_dxpl_mpio(property, H5FD_MPIO_COLLECTIVE),
               "H5Pset_dxpl_mpio");
    return property;
}

inline std::string module_stem(const char* program) {
    std::string stem = std::filesystem::path(program).filename().string();
    if (stem.size() > 3 && stem.substr(stem.size() - 3) == ".hx") {
        stem.resize(stem.size() - 3);
    }
    return stem;
}

inline std::string output_filename(const char* program,
                                   const std::string& module_base,
                                   int output_stage) {
    const std::string stem = module_stem(program);
    require(stem.compare(0, module_base.size(), module_base) == 0,
            "unexpected module name: " + stem);
    const std::string task_suffix = stem.substr(module_base.size());
    return "outfile" + std::to_string(output_stage) + task_suffix + ".h5";
}

inline void write_particles(const std::string& filename,
                            std::uint64_t particles_per_process,
                            std::uint64_t iteration,
                            const MpiSession& mpi) {
    require(particles_per_process <=
                std::numeric_limits<hsize_t>::max() /
                    static_cast<hsize_t>(mpi.size()),
            "particle count exceeds HDF5 extent range");

    const hsize_t local_rows = static_cast<hsize_t>(particles_per_process);
    const hsize_t total_rows = local_rows * static_cast<hsize_t>(mpi.size());
    const hsize_t file_dimensions[2] = {total_rows, 3};
    const hsize_t memory_dimensions[2] = {local_rows, 3};
    const hsize_t offset[2] = {
        local_rows * static_cast<hsize_t>(mpi.rank()), 0};
    const hsize_t count[2] = {local_rows, 3};

    const hid_t access = parallel_file_access();
    const hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access);
    require(file >= 0, "cannot create " + filename);
    const hid_t group = H5Gcreate2(file, "/group1", H5P_DEFAULT, H5P_DEFAULT,
                                   H5P_DEFAULT);
    require(group >= 0, "cannot create /group1 in " + filename);
    const hid_t file_space = H5Screate_simple(2, file_dimensions, nullptr);
    require(file_space >= 0, "cannot create particle file dataspace");
    const hid_t dataset = H5Dcreate2(file, "/group1/particles", H5T_IEEE_F64LE,
                                     file_space, H5P_DEFAULT, H5P_DEFAULT,
                                     H5P_DEFAULT);
    require(dataset >= 0, "cannot create /group1/particles in " + filename);
    const hid_t memory_space = H5Screate_simple(2, memory_dimensions, nullptr);
    require(memory_space >= 0, "cannot create particle memory dataspace");
    check_hdf5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, nullptr,
                                  count, nullptr),
               "H5Sselect_hyperslab");

    std::vector<double> particles(static_cast<std::size_t>(local_rows) * 3);
    for (hsize_t index = 0; index < local_rows; ++index) {
        const double global_index = static_cast<double>(offset[0] + index);
        particles[static_cast<std::size_t>(index) * 3] = global_index;
        particles[static_cast<std::size_t>(index) * 3 + 1] =
            static_cast<double>(iteration);
        particles[static_cast<std::size_t>(index) * 3 + 2] =
            static_cast<double>(mpi.rank());
    }

    const hid_t transfer = collective_transfer();
    check_hdf5(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space,
                        transfer, particles.data()),
               "H5Dwrite(/group1/particles)");

    check_hdf5(H5Pclose(transfer), "H5Pclose(dataset transfer)");
    check_hdf5(H5Sclose(memory_space), "H5Sclose(memory space)");
    check_hdf5(H5Dclose(dataset), "H5Dclose(/group1/particles)");
    check_hdf5(H5Sclose(file_space), "H5Sclose(file space)");
    check_hdf5(H5Gclose(group), "H5Gclose(/group1)");
    check_hdf5(H5Fclose(file), "H5Fclose");
    check_hdf5(H5Pclose(access), "H5Pclose(file access)");
}

inline double read_particles(const std::string& filename, const MpiSession& mpi) {
    const hid_t access = parallel_file_access();
    const hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, access);
    require(file >= 0, "cannot open " + filename);
    const hid_t dataset = H5Dopen2(file, "/group1/particles", H5P_DEFAULT);
    require(dataset >= 0, "cannot open /group1/particles in " + filename);
    const hid_t file_space = H5Dget_space(dataset);
    require(file_space >= 0, "cannot get particle dataspace");

    hsize_t dimensions[2] = {0, 0};
    require(H5Sget_simple_extent_ndims(file_space) == 2,
            "/group1/particles must be a two-dimensional dataset");
    check_hdf5(H5Sget_simple_extent_dims(file_space, dimensions, nullptr),
               "H5Sget_simple_extent_dims");
    require(dimensions[1] == 3,
            "/group1/particles must contain three coordinates per particle");

    const hsize_t ranks = static_cast<hsize_t>(mpi.size());
    const hsize_t rank = static_cast<hsize_t>(mpi.rank());
    const hsize_t base = dimensions[0] / ranks;
    const hsize_t remainder = dimensions[0] % ranks;
    const hsize_t local_rows = base + (rank < remainder ? 1 : 0);
    const hsize_t row_offset = rank * base + (rank < remainder ? rank : remainder);

    hid_t memory_space = -1;
    if (local_rows == 0) {
        check_hdf5(H5Sselect_none(file_space), "H5Sselect_none(file space)");
        memory_space = H5Screate(H5S_NULL);
    } else {
        const hsize_t offset[2] = {row_offset, 0};
        const hsize_t count[2] = {local_rows, 3};
        const hsize_t memory_dimensions[2] = {local_rows, 3};
        check_hdf5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset,
                                      nullptr, count, nullptr),
                   "H5Sselect_hyperslab");
        memory_space = H5Screate_simple(2, memory_dimensions, nullptr);
    }
    require(memory_space >= 0, "cannot create particle read dataspace");

    std::vector<double> particles(static_cast<std::size_t>(local_rows) * 3);
    const hid_t transfer = collective_transfer();
    check_hdf5(H5Dread(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space,
                       transfer, particles.empty() ? nullptr : particles.data()),
               "H5Dread(/group1/particles)");

    double checksum = 0.0;
    for (double value : particles) {
        checksum += value;
    }

    check_hdf5(H5Pclose(transfer), "H5Pclose(dataset transfer)");
    check_hdf5(H5Sclose(memory_space), "H5Sclose(memory space)");
    check_hdf5(H5Sclose(file_space), "H5Sclose(file space)");
    check_hdf5(H5Dclose(dataset), "H5Dclose(/group1/particles)");
    check_hdf5(H5Fclose(file), "H5Fclose");
    check_hdf5(H5Pclose(access), "H5Pclose(file access)");
    return checksum;
}

inline void print_timing(const std::string& label, double milliseconds) {
    std::cout << std::fixed << std::setprecision(3) << label << " Time: "
              << milliseconds << " ms" << std::endl;
}

}  // namespace wfperf

#endif

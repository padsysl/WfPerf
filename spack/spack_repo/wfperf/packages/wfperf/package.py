# Copyright (c) 2026, University of Florida. All rights reserved.
#
# This file is part of WfPerf, developed by members of Prof. Xiaoyi Lu's
# group at the University of Florida. See LICENSE in the top-level directory.

from spack_repo.builtin.build_systems.python import PythonPackage

from spack.package import *


class Wfperf(PythonPackage):
    """Benchmark distributed and in situ HPC/AI workflow performance."""

    homepage = "https://github.com/padsysl/WfPerf"
    git = "https://github.com/padsysl/WfPerf.git"

    maintainers("hqi6")

    version("main", branch="main")

    variant("parsl", default=False, description="Enable the Parsl backend")
    variant("wilkins", default=False, description="Enable the Wilkins backend")
    variant("ml", default=False, description="Enable offline training and inference")
    conflicts("~parsl~wilkins", msg="Select at least one WfPerf backend")

    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-pyyaml@6:", type=("build", "run"))

    depends_on("py-joblib@1.2:", when="+ml", type=("build", "run"))
    depends_on("py-numpy@1.23:", when="+ml", type=("build", "run"))
    depends_on("py-scikit-learn@1.2:", when="+ml", type=("build", "run"))

    depends_on("python@3.10:", when="+parsl", type=("build", "run"))
    depends_on("py-h5py@3.8:", when="+parsl", type=("build", "run"))
    depends_on("py-numpy@1.23:", when="+parsl", type=("build", "run"))
    depends_on("py-parsl@2026.02.16:", when="+parsl", type=("build", "run"))

    depends_on("mpi", when="+wilkins", type=("build", "link", "run"))
    depends_on("hdf5+mpi+hl", when="+wilkins", type=("build", "link", "run"))
    depends_on("henson+mpi-wrappers", when="+wilkins", type=("build", "link", "run"))
    depends_on("py-mpi4py", when="+wilkins", type=("build", "run"))
    depends_on("lowfive+python", when="+wilkins", type=("build", "link", "run"))
    depends_on("wilkins", when="+wilkins", type=("build", "link", "run"))
    depends_on("cmake@3.18:", when="+wilkins", type="build")

    @run_after("install")
    def install_wilkins_modules(self):
        if "+wilkins" not in self.spec:
            return

        source_dir = join_path(
            self.stage.source_path, "src", "wfperf", "backends", "wilkins", "native"
        )
        build_dir = join_path(self.stage.path, "wfperf-wilkins-build")
        mkdirp(build_dir)
        cmake = which("cmake", required=True)
        cmake(
            "-S",
            source_dir,
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX={0}".format(self.prefix),
            "-DCMAKE_C_COMPILER={0}".format(self.spec["mpi"].mpicc),
            "-DCMAKE_CXX_COMPILER={0}".format(self.spec["mpi"].mpicxx),
            "-DHDF5_ROOT={0}".format(self.spec["hdf5"].prefix),
            "-DHENSON_ROOT={0}".format(self.spec["henson"].prefix),
        )
        cmake("--build", build_dir, "--parallel", str(make_jobs))
        cmake("--install", build_dir)

    def setup_run_environment(self, env):
        if "+wilkins" in self.spec:
            env.set(
                "WFPERF_WILKINS_RUNTIME",
                join_path(self.prefix, "libexec", "wfperf", "wilkins"),
            )

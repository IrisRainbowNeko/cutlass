/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "testbed.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct TestbedSplitK : public Testbed<Gemm> {
    using Base = Testbed<Gemm>;

    using ElementCompute = typename Base::ElementCompute;

    //
    // Methods
    //

    TestbedSplitK(cutlass::Distribution::Kind init_A_ =
                          cutlass::Distribution::Uniform,
                  cutlass::Distribution::Kind init_B_ =
                          cutlass::Distribution::Uniform,
                  cutlass::Distribution::Kind init_C_ =
                          cutlass::Distribution::Uniform,
                  uint64_t seed_ = 2080)
            : Base(init_A_, init_B_, init_C_, seed_) {}

    /// Executes one test
    bool run(cutlass::gemm::GemmCoord problem_size, int split_k_slices,
             ElementCompute alpha = ElementCompute(1),
             ElementCompute beta = ElementCompute(0)) {
        this->initialize(problem_size);

        //
        // Initialize the GEMM operator
        //

        typename Gemm::Arguments arguments{problem_size,
                                           this->tensor_A.device_ref(),
                                           this->tensor_B.device_ref(),
                                           this->tensor_C.device_ref(),
                                           this->tensor_D.device_ref(),
                                           {alpha, beta},
                                           split_k_slices};

        Gemm gemm_op;

        size_t workspace_size = Gemm::get_workspace_size(arguments);

        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

        EXPECT_TRUE(status == cutlass::Status::kSuccess);

        //
        // Run the GEMM
        //

        status = gemm_op();

        EXPECT_TRUE(status == cutlass::Status::kSuccess);

        //
        // Verify
        //

        return this->verify(problem_size, alpha, beta);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestAllGemmSplitK() {
    bool passed = true;

    cutlass::gemm::GemmCoord problem_sizes[] = {{8, 8, 2048},
                                                {8, 8, 2056},
                                                {264, 72, 520},
                                                {264, 520, 120},
                                                {264, 520, 264}};

    int split_k_slices[] = {1, 2, 4, 5, 7};

    double problem_alpha[] = {0.5};

    double problem_beta[] = {2.0};

    using Testbed = TestbedSplitK<Gemm>;
    using ElementCompute = typename Testbed::ElementCompute;

    Testbed testbed;

    for (auto problem_size : problem_sizes) {
        for (int split_k_count : split_k_slices) {
            for (double alpha : problem_alpha) {
                for (double beta : problem_beta) {
                    passed = testbed.run(problem_size, split_k_count,
                                         ElementCompute(alpha),
                                         ElementCompute(beta));

                    if (!passed) {
                        std::cout << "Failed on size " << problem_size
                                  << " with split_k_count " << split_k_count
                                  << std::endl;
                        return false;
                    }
                }
            }
        }
    }

    EXPECT_TRUE(passed);

    return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace gemm
}  // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

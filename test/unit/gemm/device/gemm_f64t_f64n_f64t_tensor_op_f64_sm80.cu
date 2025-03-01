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

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Gemm_f64t_f64n_f64t_tensor_op_f64, 32x32x16_16x16x16) {
    using ElementOutput = double;
    using ElementAccumulator = double;

    using Gemm = cutlass::gemm::device::Gemm<
            double, cutlass::layout::RowMajor, double,
            cutlass::layout::ColumnMajor, ElementOutput,
            cutlass::layout::RowMajor, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<32, 32, 16>,
            cutlass::gemm::GemmShape<16, 16, 16>,
            cutlass::gemm::GemmShape<8, 8, 4>,
            cutlass::epilogue::thread::LinearCombination<
                    ElementOutput, 1, ElementAccumulator, ElementAccumulator>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Gemm_f64t_f64n_f64t_tensor_op_f64, 64x64x16_32x32x16) {
    using ElementOutput = double;
    using ElementAccumulator = double;

    using Gemm = cutlass::gemm::device::Gemm<
            double, cutlass::layout::RowMajor, double,
            cutlass::layout::ColumnMajor, ElementOutput,
            cutlass::layout::RowMajor, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<64, 64, 16>,
            cutlass::gemm::GemmShape<32, 32, 16>,
            cutlass::gemm::GemmShape<8, 8, 4>,
            cutlass::epilogue::thread::LinearCombination<
                    ElementOutput, 1, ElementAccumulator, ElementAccumulator>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Gemm_f64t_f64n_f64t_tensor_op_f64, 64x128x16_32x64x16) {
    using ElementOutput = double;
    using ElementAccumulator = double;

    using Gemm = cutlass::gemm::device::Gemm<
            double, cutlass::layout::RowMajor, double,
            cutlass::layout::ColumnMajor, ElementOutput,
            cutlass::layout::RowMajor, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<64, 128, 16>,
            cutlass::gemm::GemmShape<32, 64, 16>,
            cutlass::gemm::GemmShape<8, 8, 4>,
            cutlass::epilogue::thread::LinearCombination<
                    ElementOutput, 1, ElementAccumulator, ElementAccumulator>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Gemm_f64t_f64n_f64t_tensor_op_f64, 128x64x16_64x32x16) {
    using ElementOutput = double;
    using ElementAccumulator = double;

    using Gemm = cutlass::gemm::device::Gemm<
            double, cutlass::layout::RowMajor, double,
            cutlass::layout::ColumnMajor, ElementOutput,
            cutlass::layout::RowMajor, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 64, 16>,
            cutlass::gemm::GemmShape<64, 32, 16>,
            cutlass::gemm::GemmShape<8, 8, 4>,
            cutlass::epilogue::thread::LinearCombination<
                    ElementOutput, 1, ElementAccumulator, ElementAccumulator>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Gemm_f64t_f64n_f64t_tensor_op_f64, 128x128x16_32x64x16) {
    using ElementOutput = double;
    using ElementAccumulator = double;

    using Gemm = cutlass::gemm::device::Gemm<
            double, cutlass::layout::RowMajor, double,
            cutlass::layout::ColumnMajor, ElementOutput,
            cutlass::layout::RowMajor, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 16>,
            cutlass::gemm::GemmShape<32, 64, 16>,
            cutlass::gemm::GemmShape<8, 8, 4>,
            cutlass::epilogue::thread::LinearCombination<
                    ElementOutput, 1, ElementAccumulator, ElementAccumulator>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

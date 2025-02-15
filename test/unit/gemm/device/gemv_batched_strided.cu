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
/**
 * \file test/unit/gemm/device/gemv_batched_strided.cu
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/gemv_batched_strided.h"
#include "cutlass/gemm/kernel/default_gemv.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_gemv_batched_strided.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Device_Gemv_Batched_Strided_f32n_f32n_f32n_simt, 1x128x32_1x4x4) {
    using ElementOutput = float;
    using ElementAccumulator = float;

    using ThreadBlockShape = cutlass::gemm::GemmShape<1, 128, 32>;
    using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
    using GemvKernel = cutlass::gemm::kernel::DefaultGemv<
            ThreadBlockShape, ThreadShape, float, cutlass::layout::RowMajor,
            float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor>;
    test::gemm::device::TestAllGemvKernel<GemvKernel>();
}

TEST(SM50_Device_Gemv_Batched_Strided_f32n_f32n_f32n_simt, 1x64x64_1x4x4) {
    using ElementOutput = float;
    using ElementAccumulator = float;

    using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 64>;
    using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
    using GemvKernel = cutlass::gemm::kernel::DefaultGemv<
            ThreadBlockShape, ThreadShape, float, cutlass::layout::RowMajor,
            float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor>;
    test::gemm::device::TestAllGemvKernel<GemvKernel>();
}

TEST(SM50_Device_Gemv_Batched_Strided_f32n_f32n_f32n_simt, 1x32x128_1x4x4) {
    using ElementOutput = float;
    using ElementAccumulator = float;

    using ThreadBlockShape = cutlass::gemm::GemmShape<1, 32, 128>;
    using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
    using GemvKernel = cutlass::gemm::kernel::DefaultGemv<
            ThreadBlockShape, ThreadShape, float, cutlass::layout::RowMajor,
            float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor>;
    test::gemm::device::TestAllGemvKernel<GemvKernel>();
}

TEST(SM50_Device_Gemv_Batched_Strided_f32n_f32n_f32n_simt, 1x128x8_1x2x2) {
    using ElementOutput = float;
    using ElementAccumulator = float;

    using ThreadBlockShape = cutlass::gemm::GemmShape<1, 128, 8>;
    using ThreadShape = cutlass::gemm::GemmShape<1, 2, 2>;
    using GemvKernel = cutlass::gemm::kernel::DefaultGemv<
            ThreadBlockShape, ThreadShape, float, cutlass::layout::RowMajor,
            float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor>;
    test::gemm::device::TestAllGemvKernel<GemvKernel>();
}

TEST(SM50_Device_Gemv_Batched_Strided_f32n_f32n_f32n_simt, 1x64x16_1x2x2) {
    using ElementOutput = float;
    using ElementAccumulator = float;

    using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 16>;
    using ThreadShape = cutlass::gemm::GemmShape<1, 2, 2>;
    using GemvKernel = cutlass::gemm::kernel::DefaultGemv<
            ThreadBlockShape, ThreadShape, float, cutlass::layout::RowMajor,
            float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor>;
    test::gemm::device::TestAllGemvKernel<GemvKernel>();
}

TEST(SM50_Device_Gemv_Batched_Strided_f32n_f32n_f32n_simt, 1x32x32_1x2x2) {
    using ElementOutput = float;
    using ElementAccumulator = float;

    using ThreadBlockShape = cutlass::gemm::GemmShape<1, 32, 32>;
    using ThreadShape = cutlass::gemm::GemmShape<1, 2, 2>;
    using GemvKernel = cutlass::gemm::kernel::DefaultGemv<
            ThreadBlockShape, ThreadShape, float, cutlass::layout::RowMajor,
            float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor>;
    test::gemm::device::TestAllGemvKernel<GemvKernel>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

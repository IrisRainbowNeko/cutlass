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
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 *(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file test/unit/convolution/device/simt_int8_iconv_sm61_perf.cu
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/convolution/device/convolution.h"
#include "cutlass/cutlass.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed.h"

////////////////////////////////////////////////////////////////////////////////

TEST(SM61_Device_Convolution_s8NC4HW4_s8NC4HW4_s8NC4HW4_simt_op_dp4a_perf,
     16x64x8_16x64x8) {
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            int8_t, cutlass::layout::TensorNCxHWx<4>, int8_t,
            cutlass::layout::TensorCxRSKx<4>, ElementOutput,
            cutlass::layout::TensorNCxHWx<4>, int32_t,
            cutlass::layout::TensorNCxHWx<4>, int32_t,
            cutlass::conv::ConvType::kConvolution, cutlass::arch::OpClassSimt,
            cutlass::arch::Sm61, cutlass::gemm::GemmShape<16, 64, 8>,
            cutlass::gemm::GemmShape<16, 64, 8>,
            cutlass::gemm::GemmShape<1, 1, 4>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 4, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::
                    ConvolutionFpropNCxHWxThreadblockSwizzle,
            2, 4, 4>;

    EXPECT_TRUE(
            test::convolution::device::TestDetectionPerf<Convolution>(1000));
}

TEST(SM61_Device_Convolution_s8NC4HW4_s8NC4HW4_f32NCHW_simt_op_dp4a_perf,
     16x64x8_16x64x8) {
    using ElementOutput = float;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            int8_t, cutlass::layout::TensorNCxHWx<4>, int8_t,
            cutlass::layout::TensorCxRSKx<4>, ElementOutput,
            cutlass::layout::TensorNCHW, float, cutlass::layout::TensorNCHW,
            int32_t, cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassSimt, cutlass::arch::Sm61,
            cutlass::gemm::GemmShape<16, 64, 8>,
            cutlass::gemm::GemmShape<16, 64, 8>,
            cutlass::gemm::GemmShape<1, 1, 4>,
            cutlass::epilogue::thread::BiasAddLinearCombination<
                    ElementOutput, 1, ElementAccumulator, float,
                    ElementCompute>,
            cutlass::conv::threadblock::
                    ConvolutionFpropNCxHWxThreadblockSwizzle,
            2, 4, 4, cutlass::conv::SpecialOptimizeDesc::NONE,
            cutlass::arch::OpMultiplyAdd>;

    EXPECT_TRUE(
            test::convolution::device::TestDetectionPerf<Convolution>(1000));
}

////////////////////////////////////////////////////////////////////////////////

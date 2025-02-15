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
 * \file
 * test/unit/convolution/device/convolution_s8nchw32_s8chwn32_s8nchw32_tensor_op_s32_sm75_perf.cu
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

#include "cutlass/cutlass.h"
#include "cutlass/convolution/device/convolution.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed.h"

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Convolution_s8_s8_NC32HW32_tensor_op_mmai8816_perf,
     256x128x64_64x64x64) {
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            int8_t, cutlass::layout::TensorNCxHWx<32>, int8_t,
            cutlass::layout::TensorCxRSKx<32>, ElementOutput,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<256, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<8, 8, 16>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 8, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::
                    ConvolutionFpropNCxHWxThreadblockSwizzle,
            2, 16, 16>;

    EXPECT_TRUE(test::convolution::device::TestConvolutionPerf<Convolution>(
            1000, 256, true, false));
}

TEST(SM75_Device_Convolution_s8_s8_NC32HW32_tensor_op_mmai8816_reorderK_perf,
     128x256x64_64x64x64) {
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            int8_t, cutlass::layout::TensorNCxHWx<32>, int8_t,
            cutlass::layout::TensorCxRSKx<32>, ElementOutput,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<128, 256, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<8, 8, 16>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 8, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::ConvolutionFpropTransThreadblockSwizzle,
            2, 16, 16, cutlass::conv::SpecialOptimizeDesc::NONE,
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::ImplicitGemmMode::GEMM_TN, true>;

    EXPECT_TRUE(
            (test::convolution::device::TestConvolutionPerf<Convolution, true>(
                    1000, 256, true, false)));
}

TEST(SM75_Device_Convolution1x1_s8_s8_NC32HW32_tensor_op_mmai8816_perf,
     128x128x64_64x64x64) {
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            int8_t, cutlass::layout::TensorNCxHWx<32>, int8_t,
            cutlass::layout::TensorCxRSKx<32>, ElementOutput,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<8, 8, 16>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 8, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::
                    ConvolutionFpropNCxHWxThreadblockSwizzle,
            2, 16, 16, cutlass::conv::SpecialOptimizeDesc::CONV_FILTER_UNITY>;

    EXPECT_TRUE(test::convolution::device::TestConvolution1x1Perf<Convolution>(
            1000, 256, true, false));
}

TEST(SM75_Device_Convolution1x1_s8_s8_NC32HW32_tensor_op_mmai8816_reorderK_perf,
     128x128x64_64x64x64) {
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            int8_t, cutlass::layout::TensorNCxHWx<32>, int8_t,
            cutlass::layout::TensorCxRSKx<32>, ElementOutput,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<8, 8, 16>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 8, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::ConvolutionFpropTransThreadblockSwizzle,
            2, 16, 16, cutlass::conv::SpecialOptimizeDesc::CONV_FILTER_UNITY,
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::ImplicitGemmMode::GEMM_TN, true>;

    EXPECT_TRUE((test::convolution::device::TestConvolution1x1Perf<Convolution,
                                                                   true>(
            1000, 256, true, false)));
}

TEST(SM75_Device_Convolution_s8_s8_NC32HW32_tensor_op_mmai8816_perf,
     32x128x32_32x64x32) {
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            int8_t, cutlass::layout::TensorNCxHWx<32>, int8_t,
            cutlass::layout::TensorCxRSKx<32>, ElementOutput,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<32, 128, 32>,
            cutlass::gemm::GemmShape<32, 64, 32>,
            cutlass::gemm::GemmShape<8, 8, 16>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 8, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::
                    ConvolutionFpropNCxHWxThreadblockSwizzle,
            1, 16, 16, cutlass::conv::SpecialOptimizeDesc::NONE>;

    EXPECT_TRUE(test::convolution::device::TestDetectionPerf<Convolution>(
            1000, 16, true, false));
}

TEST(SM75_Device_Convolution_s8_s8_NC32HW32_tensor_op_mmai8816_reorderK_perf,
     128x32x32_64x32x32) {
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            int8_t, cutlass::layout::TensorNCxHWx<32>, int8_t,
            cutlass::layout::TensorCxRSKx<32>, ElementOutput,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::layout::TensorNCxHWx<32>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<128, 32, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<8, 8, 16>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 8, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::ConvolutionFpropTransThreadblockSwizzle,
            1, 16, 16, cutlass::conv::SpecialOptimizeDesc::NONE,
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::ImplicitGemmMode::GEMM_TN, true>;

    EXPECT_TRUE(
            (test::convolution::device::TestDetectionPerf<Convolution, true>(
                    1000, 16, true, false)));
}

////////////////////////////////////////////////////////////////////////////////

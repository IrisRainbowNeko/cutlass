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

#include "cutlass/cutlass.h"
#include "cutlass/convolution/device/convolution.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "region_restricted_conv2d_wgrad_testbed.h"

////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Device_Region_Resticted_Depthwise_Conv2d_Wgrad_f32_f32_nchw_simt_op_perf,
     128x128x8_32x64x8) {
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using Convolution =
            cutlass::conv::device::RegionRestrictedConvolutionBackwardFilter<
                    float, cutlass::layout::TensorNCHW, float,
                    cutlass::layout::TensorNCHW, int32_t,
                    cutlass::layout::TensorNCHW, int32_t,
                    cutlass::layout::TensorNCHW, ElementOutput,
                    cutlass::layout::TensorNCHW, float,
                    cutlass::conv::ConvType::kDepthwiseConvolution,
                    cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
                    cutlass::gemm::GemmShape<128, 128, 8>,
                    cutlass::gemm::GemmShape<32, 64, 8>,
                    cutlass::gemm::GemmShape<1, 1, 1>,
                    cutlass::epilogue::thread::LinearCombination<
                            ElementOutput, 1, ElementAccumulator,
                            ElementCompute>,
                    cutlass::conv::threadblock::
                            DepthwiseConvolutionWgradThreadblockSwizzle,
                    1, 1, 1, 1, 1, cutlass::conv::SpecialOptimizeDesc::NONE,
                    cutlass::arch::OpMultiplyAdd,
                    cutlass::conv::ImplicitGemmMode::GEMM_NT>;

    EXPECT_TRUE(
            test::convolution::device::BenchRegionRestrictedDepthwiseConv2dWgrad<Convolution>(
                    64, 1000));
}

////////////////////////////////////////////////////////////////////////////////

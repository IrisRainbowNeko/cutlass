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
 * test/unit/convolution/device/depthwise_conv2d_dgrad_f32nchw_f32nchw_f32nchw_simt_f32_sm50.cu
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
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 11)
#include "cutlass/convolution/device/convolution.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/tensor_view_io.h"

#include "conv2d_wgrad_testbed.h"

#define RUN_DEPTHWISE_CONVOLUTION(stage)                                       \
    do {                                                                       \
        using ElementOutput = float;                                           \
        using ElementAccumulator = float;                                      \
        using ElementCompute = float;                                          \
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;          \
        using Convolution = cutlass::conv::device::ConvolutionBackwardFilter<  \
                cutlass::half_t, cutlass::layout::TensorNCHW, cutlass::half_t, \
                cutlass::layout::TensorNCHW, ElementOutput,                    \
                cutlass::layout::TensorNCHW, ElementAccumulator,               \
                cutlass::conv::ConvType::kDepthwiseConvolution,                \
                cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,           \
                ThreadBlockShape, WarpShape, InstructionShape,                 \
                cutlass::epilogue::thread::LinearCombination<                  \
                        ElementOutput, 1, ElementAccumulator, ElementCompute>, \
                cutlass::conv::threadblock::                                   \
                        DepthwiseConvolutionWgradThreadblockSwizzle,           \
                stage, 1, 1, cutlass::conv::SpecialOptimizeDesc::NONE,         \
                cutlass::arch::OpMultiplyAdd,                                  \
                cutlass::conv::ImplicitGemmMode::GEMM_NT>;                     \
        EXPECT_TRUE(test::convolution::device::TestDepthwiseConv2dWgrad<       \
                    Convolution>());                                           \
    } while (0)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     128x256x64_64x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 256, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     256x128x64_64x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     128x128x64_64x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     64x128x64_32x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     128x64x64_64x32x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     64x64x64_32x32x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     128x256x32_64x64x32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 256, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     256x128x32_64x64x32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     128x128x32_64x64x32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     256x64x32_64x64x32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<256, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     64x256x32_64x64x32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 256, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     64x128x32_32x64x32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     128x64x32_64x32x32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

TEST(SM80_Device_Depthwise_Conv2dWgrad_f16_f16_NCHW_tensor_op_f32,
     64x64x32_32x32x32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
    RUN_DEPTHWISE_CONVOLUTION(1);
    RUN_DEPTHWISE_CONVOLUTION(2);
}

//////////////////////////////////////////////////////////////////////////////////
#endif

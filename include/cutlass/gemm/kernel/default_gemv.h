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

#pragma once

#include "cutlass/gemm/threadblock/gemv.h"
#include "cutlass/gemm/threadblock/default_gemv_core.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
        /// Size of the ThreadBlock tile - concept: gemm::GemmShape<>
        typename ThreadBlockShape_,
        /// Size of the per-thread shape - concept: gemm::GemmShape<>
        typename ThreadShape_,
        /// Data type of A elements
        typename ElementA_,
        /// Layout of A matrix (concept: MatrixLayout)
        typename LayoutA_,
        /// Data type of B elements
        typename ElementB_,
        /// Layout of B matrix (concept: MatrixLayout)
        typename LayoutB_,
        /// Element type of C/D matrix
        typename ElementCD_,
        /// Layout of C/D matrix (concept: MatrixLayout)
        typename LayoutCD_,
        ///  Data type of the accumulator
        typename ElementAccumulator_ = ElementCD_>
struct DefaultGemv {
    /// Shape of Threadblock-level matrix operation (concept: GemmShape)
    using ThreadBlockShape = ThreadBlockShape_;

    /// Shape of warp-level matrix operation (concept: GemmShape)
    using ThreadShape = ThreadShape_;

    /// Data type of multiplicand A
    using ElementA = ElementA_;

    /// Layout of multiplicand A
    using LayoutA = LayoutA_;

    /// Data type of multiplicand B
    using ElementB = ElementB_;

    /// Layout of multiplicand B
    using LayoutB = LayoutB_;

    /// Data type of accumulators
    using ElementAccumulator = ElementAccumulator_;

    /// Data type of accumulators (same as C/D)
    using LayoutAccumulator = LayoutCD_;

    /// Data type of input/output matrix C/D
    using ElementCD = ElementCD_;

    /// Layout of input/output matrix C/D
    using LayoutCD = LayoutCD_;

    // Define the core components
    using Core = typename cutlass::gemm::threadblock::DefaultGemvCore<
            ThreadBlockShape, ThreadShape, ElementA, LayoutA, ElementB, LayoutB,
            ElementAccumulator, LayoutAccumulator>;

    // Define the threadblock-scoped gemv
    using ThreadBlockGemv = cutlass::gemm::threadblock::Gemv<Core>;

    // Iterator for multiplicand A
    using IteratorA = typename ThreadBlockGemv::IteratorA;

    // Iterator for multiplicand B
    using IteratorB = typename ThreadBlockGemv::IteratorB;

    /// Policy for the iterator that reads/writes C/D
    using IteratorPolicyCD = typename platform::conditional<
            platform::is_same<LayoutCD, layout::RowMajor>::value,
            cutlass::transform::PitchLinearTilePolicyStripminedThreadContiguous<
                    layout::PitchLinearShape<ThreadBlockShape::kN,
                                             ThreadBlockShape::kM>,
                    Core::kThreadsPerN, ThreadShape::kN>,
            cutlass::transform::PitchLinearTilePolicyStripminedThreadStrided<
                    layout::PitchLinearShape<ThreadBlockShape::kM,
                                             ThreadBlockShape::kN>,
                    Core::kThreadsPerN, ThreadShape::kM>>::type;

    /// Iterator that reads/writes C/D
    using IteratorCD = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<ThreadBlockShape::kM, ThreadBlockShape::kN>,
            ElementCD, LayoutCD, 0, IteratorPolicyCD>;

    /// Fragment storage for C/D
    using FragmentCD = typename IteratorCD::Fragment;

    // Define the threadblock swizzle
    using ThreadBlockSwizzle = cutlass::gemm::threadblock::
            GemvBatchedStridedThreadblockDefaultSwizzle;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
        /// Size of the ThreadBlock tile - concept: gemm::GemmShape<>
        typename ThreadBlockShape_,
        /// Size of the per-thread shape - concept: gemm::GemmShape<>
        typename ThreadShape_,
        /// Data type of A elements
        typename ElementA_,
        /// Layout of A matrix (concept: MatrixLayout)
        typename ElementB_,
        /// Element type of C/D matrix
        typename ElementCD_,
        ///  Data type of the accumulator
        typename ElementAccumulator_>
struct DefaultGemv<ThreadBlockShape_, ThreadShape_, ElementA_,
                   cutlass::layout::RowMajor, ElementB_,
                   cutlass::layout::RowMajor, ElementCD_,
                   cutlass::layout::RowMajor, ElementAccumulator_> {
    /// Shape of Threadblock-level matrix operation (concept: GemmShape)
    using ThreadBlockShape = ThreadBlockShape_;

    /// Shape of warp-level matrix operation (concept: GemmShape)
    using ThreadShape = ThreadShape_;

    /// Data type of multiplicand A
    using ElementA = ElementA_;

    /// Layout of multiplicand A
    using LayoutA = cutlass::layout::RowMajor;

    /// Data type of multiplicand B
    using ElementB = ElementB_;

    /// Layout of multiplicand B
    using LayoutB = cutlass::layout::RowMajor;

    /// Data type of accumulators
    using ElementAccumulator = ElementAccumulator_;

    /// Data type of accumulators (same as C/D)
    using LayoutAccumulator = cutlass::layout::RowMajor;

    /// Data type of input/output matrix C/D
    using ElementCD = ElementCD_;

    /// Layout of input/output matrix C/D
    using LayoutCD = cutlass::layout::RowMajor;

    // Define the core components
    using Core = typename cutlass::gemm::threadblock::DefaultGemvCore<
            ThreadBlockShape, ThreadShape, ElementA, LayoutA, ElementB, LayoutB,
            ElementAccumulator, LayoutAccumulator>;

    // Define the threadblock-scoped gemv
    using ThreadBlockGemv = cutlass::gemm::threadblock::GemvBatchedReduction<Core>;

    // Iterator for multiplicand A
    using IteratorA = typename ThreadBlockGemv::IteratorA;

    // Iterator for multiplicand B
    using IteratorB = typename ThreadBlockGemv::IteratorB;

    /// Policy for the iterator that reads/writes C/D
    using IteratorPolicyCD =
            cutlass::transform::PitchLinearTilePolicyStripminedThreadContiguous<
                    layout::PitchLinearShape<ThreadBlockShape::kN,
                                             ThreadBlockShape::kM>,
                    Core::kThreadsPerN, ThreadShape::kN>;

    /// Iterator that reads/writes C/D
    using IteratorCD = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<ThreadBlockShape::kM, ThreadBlockShape::kN>,
            ElementCD, LayoutCD, 0, IteratorPolicyCD>;

    /// Fragment storage for C/D
    using FragmentCD = typename IteratorCD::Fragment;

    // Define the threadblock swizzle
    using ThreadBlockSwizzle = cutlass::gemm::threadblock::
            GemvBatchedStridedThreadblockReductionSwizzle;
};

/////////////////////////////////////////////////////////////////////////////////////////////////


}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

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
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix
   multiply-add with the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major
   outputs are accommodated by exchanging A and B operands and assuming
   transposed layouts. Partial specializations here choose
   'device::GemmTransposed' to implement this functionality.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
        /// Element type for A matrix operand
        typename ElementA_,
        /// Layout type for A matrix operand
        typename LayoutA_,
        /// Complex elementwise transformation on A operand
        ComplexTransform TransformA,
        /// Access granularity of A matrix in units of elements
        int kAlignmentA,
        /// Element type for B matrix operand
        typename ElementB_,
        /// Layout type for B matrix operand
        typename LayoutB_,
        /// Complex elementwise transformation on B operand
        ComplexTransform TransformB,
        /// Access granularity of B matrix in units of elements
        int kAlignmentB,
        /// Element type for C and D matrix operands
        typename ElementC_,
        /// Layout type for C and D matrix operands
        typename LayoutC_,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Operator class tag
        typename OperatorClass,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape,
        /// Warp-level tile size (concept: GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by GEMM
        typename Operator,
        /// Use zfill or predicate for out-of-bound cp.async
        SharedMemoryClearOption SharedMemoryClear =
                SharedMemoryClearOption::kNone,
        ///
        typename Enable = void>
struct DefaultGemmUniversal;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued GEMM kernels
//

template <
        /// Element type for A matrix operand
        typename ElementA,
        /// Layout type for A matrix operand
        typename LayoutA,
        /// Access granularity of A matrix in units of elements
        int kAlignmentA,
        /// Element type for B matrix operand
        typename ElementB,
        /// Layout type for B matrix operand
        typename LayoutB,
        /// Access granularity of B matrix in units of elements
        int kAlignmentB,
        /// Element type for C and D matrix operands
        typename ElementC,
        /// Layout type for C and D matrix operands
        typename LayoutC,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Operator class tag
        typename OperatorClass,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape,
        /// Warp-level tile size (concept: GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by GEMM
        typename Operator,
        /// Use zfill or predicate for out-of-bound cp.async
        SharedMemoryClearOption SharedMemoryClear>
struct DefaultGemmUniversal<
        ElementA, LayoutA,
        ComplexTransform::kNone,  // transform A
        kAlignmentA, ElementB, LayoutB,
        ComplexTransform::kNone,  // transform B
        kAlignmentB, ElementC, LayoutC, ElementAccumulator, OperatorClass,
        ArchTag, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator,
        SharedMemoryClear,
        typename std::enable_if<
                !cutlass::is_complex<ElementAccumulator>::value>::type> {
    using DefaultGemmKernel = typename kernel::DefaultGemm<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
            ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
            ThreadblockSwizzle, Stages, true, Operator,
            false, SharedMemoryClear>::GemmKernel;

    /// Define the kernel in terms of the default kernel
    using GemmKernel =
            kernel::GemmUniversal<typename DefaultGemmKernel::Mma,
                                  typename DefaultGemmKernel::Epilogue,
                                  ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Complex-valued GEMM kernels
//

template <
        /// Element type for A matrix operand
        typename ElementA,
        /// Layout type for A matrix operand
        typename LayoutA,
        /// Complex elementwise transformation on A operand
        ComplexTransform TransformA,
        /// Access granularity of A matrix in units of elements
        int kAlignmentA,
        /// Element type for B matrix operand
        typename ElementB,
        /// Layout type for B matrix operand
        typename LayoutB,
        /// Complex elementwise transformation on B operand
        ComplexTransform TransformB,
        /// Access granularity of B matrix in units of elements
        int kAlignmentB,
        /// Element type for C and D matrix operands
        typename ElementC,
        /// Layout type for C and D matrix operands
        typename LayoutC,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Operator class tag
        typename OperatorClass,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape,
        /// Warp-level tile size (concept: GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by GEMM
        typename Operator,
        /// Use zfill or predicate for out-of-bound cp.async
        SharedMemoryClearOption SharedMemoryClear>
struct DefaultGemmUniversal<
        ElementA, LayoutA, TransformA, kAlignmentA, ElementB, LayoutB,
        TransformB, kAlignmentB, ElementC, LayoutC, ElementAccumulator,
        OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator,
        SharedMemoryClear,
        typename std::enable_if<
                cutlass::is_complex<ElementAccumulator>::value>::type> {
    using DefaultGemmKernel = typename kernel::DefaultGemmComplex<
            ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
            WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
            Stages, TransformA, TransformB, Operator, false>::GemmKernel;

    /// Define the kernel in terms of the default kernel
    using GemmKernel =
            kernel::GemmUniversal<typename DefaultGemmKernel::Mma,
                                  typename DefaultGemmKernel::Epilogue,
                                  ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

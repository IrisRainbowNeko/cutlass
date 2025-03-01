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

#include "cutlass/cutlass.h"
#include "cutlass/complex.h"

namespace cutlass {
namespace transform {
namespace thread {

namespace UnaryTransform {
struct Identity;   ///< None (i.e., identity)
struct Conjugate;  ///< Complex conjugate
}  // namespace UnaryTransform

/// Element-wise unary operator that transforms one element of a fragment at a
/// time
template <typename FragmentIn,   ///< Input Fragment
          typename FragmentOut,  ///< Output Fragment
          typename Transform>    ///< Unary transform operator
class UnaryOp {
public:
    CUTLASS_DEVICE
    static FragmentOut execute(FragmentIn& in) {
        static_assert(FragmentIn::kElements == FragmentOut::kElements,
                      "Number of elements must match.");
        static_assert(
                std::is_same<Transform, UnaryTransform::Identity>::value ||
                        std::is_same<Transform,
                                     UnaryTransform::Conjugate>::value,
                "Unary Operator not supported.");

        FragmentOut out;
        if (std::is_same<Transform, UnaryTransform::Identity>::value) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentIn::kElements; ++i) {
                out[i] = static_cast<typename FragmentOut::Element>(in[i]);
            }
        } else if (std::is_same<Transform, UnaryTransform::Conjugate>::value) {
            for (int i = 0; i < FragmentIn::kElements; ++i) {
                out[i] =
                        conj(static_cast<typename FragmentOut::Element>(in[i]));
            }
        }
        return out;
    }
};

template <typename FragmentIn, typename Transform>
class UnaryOp<FragmentIn, FragmentIn, Transform> {
public:
    CUTLASS_DEVICE
    static FragmentIn execute(FragmentIn& in) {
        static_assert(
                std::is_same<Transform, UnaryTransform::Identity>::value ||
                        std::is_same<Transform,
                                     UnaryTransform::Conjugate>::value,
                "Unary Operator not supported.");

        if (std::is_same<Transform, UnaryTransform::Identity>::value) {
            return in;
        } else if (std::is_same<Transform, UnaryTransform::Conjugate>::value) {
            for (int i = 0; i < FragmentIn::kElements; ++i) {
                in[i] = conj(in[i]);
            }
        }
        return in;
    }
};
}  // namespace thread
}  // namespace transform
}  // namespace cutlass

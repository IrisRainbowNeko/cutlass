/***************************************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
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
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator followed by an activation function to
/// an array of elements.
///
/// D = activation(alpha * accumulator + beta * source + uniform)
///
template <template <typename T> class ActivationFunctor,
          typename ElementOutput_,  ///< Data type used to load and store
                                    ///< tensors
          int Count,  ///< Number of elements computed per operation
                      ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                      ///< but we use 64 or 32 sometimes when there are not
                      ///< enough data to store
          typename ElementAccumulator_ =
                  ElementOutput_,  ///< Accumulator data type
          typename ElementCompute_ =
                  ElementOutput_,  ///< Data type used to compute linear
                                   ///< combination
          ScaleType::Kind Scale =
                  ScaleType::Default,  ///< Control Alpha and Beta scaling
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          bool IsHeavy = false>
class LinearCombinationGeneric {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;

    static bool const kIsHeavy = IsHeavy;
    static int const kCount = Count;
    static const ScaleType::Kind kScale = Scale;

    using FragmentOutput = Array<ElementOutput, kCount>;
    using FragmentAccumulator = Array<ElementAccumulator, kCount>;
    using FragmentCompute = Array<ElementCompute, kCount>;

    static FloatRoundStyle const kRound = Round;

    /// Host-constructable parameters structure
    struct Params {
        ElementCompute alpha;             ///< scales accumulators
        ElementCompute beta;              ///< scales source tensor
        ElementCompute const* alpha_ptr;  ///< pointer to accumulator scalar -
                                          ///< if not null, loads it from memory
        ElementCompute const* beta_ptr;   ///< pointer to source scalar - if not
                                          ///< null, loads it from memory

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params()
                : alpha(ElementCompute(1)),
                  beta(ElementCompute(0)),
                  alpha_ptr(nullptr),
                  beta_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute alpha, ElementCompute beta)
                : alpha(alpha),
                  beta(beta),
                  alpha_ptr(nullptr),
                  beta_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr)
                : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {}
    };

private:
    //
    // Data members
    //

    ElementCompute alpha_;
    ElementCompute beta_;

public:
    /// Constructs the function object, possibly loading from pointers in host
    /// memory
    CUTLASS_HOST_DEVICE
    LinearCombinationGeneric(Params const& params) {
        alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
        beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    }

    /// Returns true if source is needed
    CUTLASS_HOST_DEVICE
    bool is_source_needed() const {
        if (Scale == ScaleType::NoBetaScaling)
            return true;

        if (Scale == ScaleType::OnlyAlphaScaling)
            return false;

        if (Scale == ScaleType::Nothing)
            return false;

        return beta_ != ElementCompute(0);
    }

    /// Functionally required for serial reduction in the epilogue
    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int k_partition_count) {
        if (k_partition) {
            beta_ = ElementCompute(1);
        }
    }

    /// Computes linear scaling: D = alpha * accumulator + beta * source
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator,
                              FragmentOutput const& source) const {
        // Convert source to interal compute numeric type
        NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
                source_converter;
        NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
                accumulator_converter;

        FragmentCompute converted_source = source_converter(source);
        FragmentCompute converted_accumulator =
                accumulator_converter(accumulator);

        // Perform binary operations

        FragmentCompute intermediate;

        multiplies<FragmentCompute> mul_add_source;
        multiply_add<FragmentCompute> mul_add_accumulator;
        ActivationFunctor<FragmentCompute> activation;

        if (Scale == ScaleType::NoBetaScaling) {
            intermediate = converted_source;
            intermediate =
                    mul_add_accumulator(alpha_, converted_accumulator,
                                        intermediate);  // D = alpha * Accum + X
        } else if (Scale == ScaleType::Nothing) {
            intermediate = converted_accumulator;
        } else {
            intermediate = mul_add_source(
                    beta_, converted_source);  // X =  beta * C + uniform
            intermediate =
                    mul_add_accumulator(alpha_, converted_accumulator,
                                        intermediate);  // D = alpha * Accum + X
        }

        intermediate = activation(intermediate);

        // Convert to destination numeric type
        NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
                destination_converter;

        return destination_converter(intermediate);
    }

    /// Computes linear scaling: D = alpha * accumulator
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
        // Convert source to interal compute numeric type
        NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
                accumulator_converter;

        FragmentCompute converted_accumulator =
                accumulator_converter(accumulator);

        // Perform binary operations

        FragmentCompute intermediate;

        multiplies<FragmentCompute> mul_add_accumulator;
        ActivationFunctor<FragmentCompute> activation;

        if (Scale == ScaleType::Nothing) {
            intermediate = converted_accumulator;
        } else {
            intermediate = mul_add_accumulator(
                    alpha_, converted_accumulator);  // D = alpha * Accum
        }

        intermediate = activation(intermediate);

        // Convert to destination numeric type
        NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
                destination_converter;

        return destination_converter(intermediate);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

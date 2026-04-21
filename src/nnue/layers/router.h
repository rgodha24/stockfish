/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Definition of Router layer for NNUE evaluation function
// A tiny dense affine transform (uint8 input -> int32 output) with argmax.
// Intended for routing LayerStacks based on transformed feature slices.

#ifndef NNUE_LAYERS_ROUTER_H_INCLUDED
#define NNUE_LAYERS_ROUTER_H_INCLUDED

#include <cstdint>
#include <iostream>

#include "../../memory.h"
#include "../nnue_common.h"
#include "../simd.h"

namespace Stockfish::Eval::NNUE::Layers {

#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)
    #define ENABLE_ROUTER_SEQ_OPT
#endif

template<IndexType InDims, IndexType OutDims>
class Router {
   public:
    using InputType  = std::uint8_t;
    using OutputType = std::int32_t;

    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0xABCD1234u;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        return hashValue;
    }

    static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4
             + i / PaddedInputDimensions * 4 + i % 4;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#ifdef ENABLE_ROUTER_SEQ_OPT
        return get_weight_index_scrambled(i);
#else
        return i;
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);
        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        write_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);
        return !stream.fail();
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation: returns the argmax index (lower index on ties)
    IndexType propagate(const InputType* input) const {
        alignas(CacheLineSize) OutputType output[PaddedOutputDimensions];
        propagate_matmul(input, output);
        return argmax(output);
    }

   private:
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    using BiasType   = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];

    void propagate_matmul(const InputType* input, OutputType* output) const {
#ifdef ENABLE_ROUTER_SEQ_OPT
    #if defined(USE_AVX2)
        using vec_t = __m256i;
        #define vec_set_32 _mm256_set1_epi32
        #define vec_add_32 _mm256_add_epi32
        #define vec_add_dpbusd_32 SIMD::m256_add_dpbusd_epi32
    #elif defined(USE_SSSE3)
        using vec_t = __m128i;
        #define vec_set_32 _mm_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m128_add_dpbusd_epi32
    #elif defined(USE_NEON_DOTPROD)
        using vec_t = int32x4_t;
        #define vec_set_32 vdupq_n_s32
        #define vec_add_dpbusd_32(acc, a, b) \
            SIMD::dotprod_m128_add_dpbusd_epi32(acc, vreinterpretq_s8_s32(a), \
                                                vreinterpretq_s8_s32(b))
    #endif

        static constexpr IndexType OutputSimdWidth = sizeof(vec_t) / sizeof(OutputType);
        static_assert(OutputDimensions % OutputSimdWidth == 0);

        constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
        constexpr IndexType NumAccums = OutputDimensions / OutputSimdWidth;

        const vec_t* biasvec = reinterpret_cast<const vec_t*>(biases);
        vec_t        acc[NumAccums];
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = biasvec[k];

        for (IndexType i = 0; i < NumChunks; ++i)
        {
            const vec_t in0  = vec_set_32(load_as<std::int32_t>(input + i * sizeof(std::int32_t)));
            const auto  col0 = reinterpret_cast<const vec_t*>(&weights[i * OutputDimensions * 4]);

            for (IndexType k = 0; k < NumAccums; ++k)
                vec_add_dpbusd_32(acc[k], in0, col0[k]);
        }

        vec_t* outptr = reinterpret_cast<vec_t*>(output);
        for (IndexType k = 0; k < NumAccums; ++k)
            outptr[k] = acc[k];

    #undef vec_set_32
    #undef vec_add_32
    #undef vec_add_dpbusd_32
#else
        // Scalar fallback
        std::memcpy(output, biases, sizeof(OutputType) * OutputDimensions);
        for (IndexType i = 0; i < InputDimensions; ++i)
        {
            const InputType in = input[i];
            if (in == 0)
                continue;
            const WeightType* w = &weights[i];
            for (IndexType j = 0; j < OutputDimensions; ++j)
                output[j] += w[j * PaddedInputDimensions] * in;
        }
#endif
    }

    static IndexType argmax(const OutputType* output) {
        IndexType  bestIdx = 0;
        OutputType bestVal = output[0];
        for (IndexType i = 1; i < OutputDimensions; ++i)
        {
            if (output[i] > bestVal)
            {
                bestVal = output[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
};

}  // namespace Stockfish::Eval::NNUE::Layers

template<Stockfish::Eval::NNUE::IndexType InDims, Stockfish::Eval::NNUE::IndexType OutDims>
struct std::hash<Stockfish::Eval::NNUE::Layers::Router<InDims, OutDims>> {
    std::size_t operator()(
      const Stockfish::Eval::NNUE::Layers::Router<InDims, OutDims>& router) const noexcept {
        return router.get_content_hash();
    }
};

#endif  // #ifndef NNUE_LAYERS_ROUTER_H_INCLUDED

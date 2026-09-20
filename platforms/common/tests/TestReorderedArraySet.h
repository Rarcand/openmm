// Device tests shared by CUDA and OpenCL test drivers.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "openmm/common/ReorderedArraySet.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ContextSelector.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <random>

namespace OpenMM {
inline int testReorderedArraySet(ComputeContext& cc, ComputeContext& otherContext) {
    ContextSelector select(cc);
    const int n = 131, padded = 160;
    int assertions = 0;
    auto require = [&](bool condition) {
        if (!condition) throw OpenMMException("Reordered array device test failed");
        assertions++;
    };
    auto rejects = [&](std::function<void()> operation) {
        bool rejected = false;
        try { operation(); } catch (const OpenMMException&) { rejected = true; }
        require(rejected);
    };
    ReorderedArraySet arrays(cc, n, padded);
    rejects([&]() { arrays.getArguments("GLOBAL"); });
    rejects([&]() { ReorderedArraySet invalid(cc, -1, padded); });
    rejects([&]() { ReorderedArraySet invalid(cc, n, n-1); });
    ComputeArray missing, small, wrongSize, original, sorted, foreign;
    original.initialize(cc, n, 4, "original");
    sorted.initialize(cc, padded, 4, "sorted");
    small.initialize(cc, n-1, 4, "small");
    wrongSize.initialize(cc, padded, 8, "wrongSize");
    {
        ContextSelector selectOther(otherContext);
        foreign.initialize(otherContext, padded, 4, "foreign");
    }
    rejects([&]() { arrays.addArray(missing, sorted); });
    rejects([&]() { arrays.addArray(small, sorted); });
    rejects([&]() { arrays.addArray(original, small); });
    rejects([&]() { arrays.addArray(original, wrongSize); });
    rejects([&]() { arrays.addArray(original, foreign); });
    rejects([&]() { arrays.addArray(sorted, sorted.getArray()); });
    {
        ReorderedArraySet aliases(cc, n, padded);
        aliases.addArray(original, sorted);
        rejects([&]() { aliases.addArray(original.getArray(), sorted.getArray()); });
        ComputeArray another;
        another.initialize(cc, padded, 4, "another");
        rejects([&]() { aliases.addArray(sorted, another); });
        rejects([&]() { aliases.addArray(another, original); });
    }
    const std::vector<int> strides = {1, 2, 3, 4, 7, 8, 12, 16, 24, 32, 48, 64};
    std::vector<std::unique_ptr<ComputeArray>> sources, destinations;
    for (int stride : strides) {
        sources.emplace_back(new ComputeArray());
        destinations.emplace_back(new ComputeArray());
        sources.back()->initialize(cc, n, stride, "source");
        destinations.back()->initialize(cc, padded, stride, "destination");
        arrays.addArray(*sources.back(), *destinations.back());
    }
    arrays.seal();
    rejects([&]() { arrays.addArray(original, sorted); });
    ComputeArray orderArray;
    orderArray.initialize<int>(cc, padded, "order");
    std::string source = "KERNEL void testGather(GLOBAL const int* order"+arrays.getArguments("GLOBAL")+") {\n"
        "for (int s=GLOBAL_ID; s<131; s+=GLOBAL_SIZE) {\n"+arrays.getGatherSource("s", "order[s]")+"}}\n";
    ComputeKernel kernel = cc.compileProgram(source)->createKernel("testGather");
    kernel->addArg(orderArray);
    for (const auto& entry : arrays.getEntries()) {
        kernel->addArg(*entry.original);
        kernel->addArg(*entry.sorted);
    }
    std::mt19937 rng(5406);
    std::vector<int> order(padded);
    std::iota(order.begin(), order.end(), 0);
    for (int round = 0; round < 3; round++) {
        if (round == 1) std::reverse(order.begin(), order.begin()+n);
        if (round == 2) std::shuffle(order.begin(), order.begin()+n, rng);
        orderArray.upload(order);
        std::vector<std::vector<unsigned char>> expected;
        for (int i = 0; i < strides.size(); i++) {
            int stride = strides[i];
            expected.emplace_back(n*stride);
            for (auto& byte : expected.back()) byte = rng() & 255;
            std::vector<unsigned char> sentinel(padded*stride, 0xa5);
            sources[i]->upload(expected.back().data());
            destinations[i]->upload(sentinel.data());
        }
        arrays.validate();
        kernel->execute(padded, 64);
        for (int i = 0; i < strides.size(); i++) {
            int stride = strides[i];
            std::vector<unsigned char> actual(padded*stride), unchanged(n*stride);
            destinations[i]->download(actual.data());
            sources[i]->download(unchanged.data());
            require(unchanged == expected[i]);
            for (int atom = 0; atom < padded; atom++)
                for (int byte = 0; byte < stride; byte++)
                    require(actual[atom*stride+byte] == (atom < n ? expected[i][order[atom]*stride+byte] : 0xa5));
        }
    }
    sources[0]->resize(n-1);
    rejects([&]() { arrays.validate(); });
    sources[0]->resize(n);
    destinations[0]->resize(padded-1);
    rejects([&]() { arrays.validate(); });
    destinations[0]->resize(padded);
    arrays.validate();
    // An empty set still permits fusing other work such as charge gathering.
    ReorderedArraySet empty(cc, n, padded);
    empty.seal();
    require(empty.getArguments("GLOBAL").empty() && empty.getGatherSource("s", "order[s]").empty());
    return assertions;
}
} // namespace OpenMM

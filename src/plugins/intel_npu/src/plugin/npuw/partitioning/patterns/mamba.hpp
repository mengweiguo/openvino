// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

namespace online {
class Snapshot;  // Forward declaration
}  // namespace online

namespace patterns {
namespace ssm {

// SSM/Mamba block pattern matching tag
constexpr const char* SSM_TAG = "ssm";

// Matches the Mamba conv-state output chain:
//   ScatterNDUpdate (opset4) → Reshape → Multiply → ReduceSum → Add → Swish
// Validates Roll (opset7) is 2 hops upstream of ScatterNDUpdate to confirm
// this is a Mamba block. Isolates only the 6-node output chain — Roll and
// GatherAssign intermediates are left for repeatedBlocks() to handle.
//
// Verified on IBM Granite 4.0-H Micro INT4 (36 Mamba layers).
class MambaBlock : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::ssm::MambaBlock");
    MambaBlock(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

// Matches Mamba/SSM blocks by layer name boundaries (DISABLED — BFS scope
// issues pending redesign). Constructor is a no-op.
class MambaBlockByName : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::ssm::MambaBlockByName");
    MambaBlockByName(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace ssm
}  // namespace patterns
}  // namespace npuw
}  // namespace ov

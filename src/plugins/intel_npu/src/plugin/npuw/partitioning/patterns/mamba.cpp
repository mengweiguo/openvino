// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mamba.hpp"

#include <queue>
#include <regex>
#include <unordered_set>

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace ssm {

namespace opp = ov::pass::pattern;

// Mamba Block Conv-State Pattern
//
// Matches the conv-state output chain unique to Mamba layers:
//   ScatterNDUpdate → Reshape → Multiply → ReduceSum → Add → Swish
// and validates Roll (opset7) feeds ScatterNDUpdate via Reshape (2 hops up).
//
// Only the 6 nodes in the linear output chain are isolated.
// Roll and GatherAssign intermediates are NOT isolated — they stay naturally
// grouped by repeatedBlocks(). Isolating Roll would sever it from its i64
// shape-computation consumers, creating a stranded group with no results.
//
// Verified topology (IBM Granite 4.0-H Micro INT4, 36 Mamba layers):
//
//   ReadValue (opset6, conv state, not isolated)
//       |
//      Roll (opset7)              ← NOT isolated (keeps natural grouping)
//       |   \
//       |    [GatherAssign_XXX intermediates — NOT isolated]
//       |                           |
//       └──────────────────────────→ ScatterNDUpdate ← isolated
//                                    |
//                                   Reshape          ← isolated
//                                    |
//                                   Multiply         ← isolated (conv weights)
//                                    |
//                                   ReduceSum        ← isolated (kernel dim)
//                                    |
//                                   Add              ← isolated (bias)
//                                    |
//                                   Swish            ← isolated, pattern root
MambaBlock::MambaBlock(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto scatter = opp::wrap_type<ov::op::v3::ScatterNDUpdate>({opp::any_input(), opp::any_input(), opp::any_input()});
    auto reshape = opp::wrap_type<ov::op::v1::Reshape>({scatter->output(0), opp::any_input()});
    auto multiply = opp::wrap_type<ov::op::v1::Multiply>({reshape->output(0), opp::any_input()});
    auto reduce = opp::wrap_type<ov::op::v1::ReduceSum>({multiply->output(0), opp::any_input()});
    auto add = opp::wrap_type<ov::op::v1::Add>({reduce->output(0), opp::any_input()});
    auto swish = opp::wrap_type<ov::op::v4::Swish>({add->output(0)});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto scatter_it = node_to_output.find(scatter);
        if (scatter_it == node_to_output.end()) {
            return false;
        }
        auto scatter_node = scatter_it->second.get_node_shared_ptr();

        // Validate: ScatterNDUpdate port 0 → Reshape_1 → Roll (exactly 2 hops).
        // This confirms this is a Mamba conv-state block, not an unrelated ScatterNDUpdate.
        auto port0_src = scatter_node->get_input_source_output(0).get_node_shared_ptr();
        std::shared_ptr<ov::Node> roll_node;
        if (ov::is_type<ov::op::v1::Reshape>(port0_src)) {
            auto reshape_input = port0_src->get_input_source_output(0).get_node_shared_ptr();
            if (ov::is_type<ov::op::v7::Roll>(reshape_input)) {
                roll_node = reshape_input;
            }
        }
        if (!roll_node) {
            return false;  // Not a Mamba conv-state ScatterNDUpdate
        }

        LOG_DEBUG("MambaBlock: matched at " << scatter_node->get_friendly_name());

        // Isolate only the 6-node linear chain — no BFS, no Roll isolation.
        // Roll and GatherAssign intermediates are left for repeatedBlocks() to group
        // naturally. Isolating Roll would sever it from its i64 shape-computation
        // consumers, creating a stranded group with no non-optimized outputs.
        auto isolate_node = [&](const std::shared_ptr<ov::Node>& node) {
            auto it = node_to_gptr->find(node);
            if (it != node_to_gptr->end()) {
                it->second->isolate(isol_tag);
            }
        };

        // Isolate the 6-node linear chain: ScatterNDUpdate → Reshape → Multiply → ReduceSum → Add → Swish
        for (const auto& pat : {scatter, reshape, multiply, reduce, add, swish}) {
            auto it = node_to_output.find(pat);
            if (it != node_to_output.end()) {
                isolate_node(it->second.get_node_shared_ptr());
            }
        }

        return false;  // root hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(swish, "TagMambaBlock"), std::move(callback));
}

// MambaBlockByName — disabled (BFS scope issues, to be fixed separately)
MambaBlockByName::MambaBlockByName(const std::shared_ptr<ov::npuw::online::Snapshot>& /*snapshot*/,
                                   const std::string& /*isol_tag*/) {
    // No-op: disabled pending redesign of name-based block boundary detection.
}

}  // namespace ssm
}  // namespace patterns
}  // namespace npuw
}  // namespace ov

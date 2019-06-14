#include "puct2/node.h"

#include "gdltransformer.h"

#include <statemachine/statemachine.h>
#include <statemachine/jointmove.h>
#include <statemachine/basestate.h>

#include <k273/strutils.h>
#include <k273/exception.h>
#include <k273/logging.h>

#include <vector>
#include <algorithm>

using namespace std;
using namespace GGPZero::PuctV2;

static string scoreString(const PuctNode* node,
                          GGPLib::StateMachineInterface* sm, bool final=false) {
    const int role_count = sm->getRoleCount();
    string res = "(";
    for (int ii=0; ii<role_count; ii++) {
        if (ii > 0) {
            res += " ";
        }

        if (final) {
            res += K273::fmtString("%.2f", node->getFinalScore(ii));
        } else {
            res += K273::fmtString("%.2f", node->getCurrentScore(ii));
        }
    }

    res += ")";

    return res;
}

///////////////////////////////////////////////////////////////////////////////

static PuctNode* createNode(const GGPLib::BaseState* base_state,
                            bool is_finalised,
                            int lead_role_index,
                            int num_children,
                            int role_count) {

    const int child_size = sizeof(PuctNodeChild);
    int current_score_bytes = round_up_8(role_count * sizeof(Score));
    int final_score_bytes = round_up_8(role_count * sizeof(Score));
    int base_state_bytes = round_up_8(sizeof(GGPLib::BaseState) + base_state->byte_count);

    // remember that the JointMove is inline, so we only need to count the indices
    int node_child_bytes = round_up_8(child_size + role_count * sizeof(GGPLib::JointMove::IndexType));

    int total_bytes = (sizeof(PuctNode) + current_score_bytes + final_score_bytes +
                       base_state_bytes + (num_children * node_child_bytes));

    // K273::l_debug("total_bytes %d #child %d (%d / %d / %d / %d)",
    //               total_bytes, num_children, current_score_bytes,
    //               final_score_bytes, base_state_bytes,
    //               (num_children * node_child_bytes));

    PuctNode* node = static_cast <PuctNode*>(malloc(total_bytes));
    node->parent = nullptr;
    node->visits = 0;
    node->inflight_visits = 0;
    node->ref_count = 1;

    node->unselectable_count = 0;

    node->num_children = num_children;
    node->num_children_expanded = 0;

    // will be set in selectChild()
    node->puct_constant = 1.44;

    node->is_finalised = is_finalised;
    node->force_terminal = false;

    node->dirichlet_noise_set = false;

    node->lead_role_index = lead_role_index;

    // will be set by the evaluator
    node->game_depth = 0;

    node->final_score_ptr_incr = current_score_bytes;
    node->basestate_ptr_incr = current_score_bytes + final_score_bytes;
    node->children_ptr_incr = current_score_bytes + final_score_bytes + base_state_bytes;

    // store the allocated size
    node->allocated_size = total_bytes;

    // initialise all scores to zero (will be set by evaluation/statemachine anyways)
    for (int ii=0; ii<role_count; ii++) {
        node->setFinalScore(ii, 0.0);
        node->setCurrentScore(ii, 0.0);
    }

    // copy the base state
    GGPLib::BaseState* node_bs = node->getBaseState();
    node_bs->init(base_state->size);
    node_bs->assign(base_state);

    // children initialised in initialiseChildHelper()...
    return node;
}

static int initialiseChildHelper(PuctNode* node, int role_index, int child_index,
                                 int role_count, GGPLib::StateMachineInterface* sm,
                                 GGPLib::JointMove* joint_move) {

    GGPLib::LegalState* ls = sm->getLegalState(role_index);
    bool final_role = role_index == role_count - 1;

    for (int ii=0; ii<ls->getCount(); ii++) {
        int choice = ls->getLegal(ii);
        joint_move->set(role_index, choice);

        if (final_role) {
            PuctNodeChild* child = node->getNodeChild(role_count, child_index++);
            child->to_node = nullptr;

            child->unselectable = false;
            child->traversals = 0;

            // by default set to 1.0, will be overridden
            child->policy_prob_orig = 1.0f;
            child->policy_prob = 1.0f;
            child->next_prob = 0.0f;

            child->debug_node_score = 0.0;
            child->debug_puct_score = 0.0;

            child->move.setSize(role_count);
            child->move.assign(joint_move);

        } else {
            // recurses, needs to set children
            child_index = initialiseChildHelper(node, role_index + 1,
                                                child_index, role_count, sm, joint_move);
        }
    }

    return child_index;
}


// This is a static method.
PuctNode* PuctNode::create(const GGPLib::BaseState* base_state,
                           GGPLib::StateMachineInterface* sm) {

    const int role_count = sm->getRoleCount();
    sm->updateBases(base_state);

    int lead_role_index = 0;
    bool is_finalised = true;
    int total_children = 0;
    if (!sm->isTerminal()) {

        total_children = 1;
        is_finalised = false;

        // how many children do we need? (effectively a cross product)
        int max_moves_for_a_role = 1;
        for (int ri=0; ri<role_count; ri++) {
            GGPLib::LegalState* ls = sm->getLegalState(ri);
            total_children *= ls->getCount();
            if (ls->getCount() > max_moves_for_a_role) {
                max_moves_for_a_role = ls->getCount();
                lead_role_index = ri;
            }
        }

        //k_debug("total %d", total_children);
        if (max_moves_for_a_role > 1) {

            // note lead_role_index could be any player at this point

            // are the rest 1?
            bool rest_one = true;
            for (int ri=0; ri<role_count; ri++) {
                GGPLib::LegalState* ls = sm->getLegalState(ri);
                if (ri != lead_role_index && ls->getCount() > 1) {
                    rest_one = false;
                }
            }

            // simultaneous?
            if (!rest_one) {
                lead_role_index = PuctNode::lead_role_index_simultaneous;
            }
        }
    }

    PuctNode* node = ::createNode(base_state,
                                  is_finalised,
                                  lead_role_index,
                                  total_children,
                                  role_count);

    if (!node->is_finalised) {
        char buf[GGPLib::JointMove::mallocSize(role_count)];
        GGPLib::JointMove* move = (GGPLib::JointMove*) buf;
        int count = initialiseChildHelper(node, 0, 0, role_count, sm, move);
        ASSERT (count == total_children);

    } else {
        // set the scores
        for (int ii=0; ii<role_count; ii++) {
            int score = sm->getGoalValue(ii);
            node->setFinalScore(ii, score / 100.0);
            node->setCurrentScore(ii, score / 100.0);
        }
    }

    return node;
}

///////////////////////////////////////////////////////////////////////////////

string PuctNode::moveString(const GGPLib::JointMove& move,
                            GGPLib::StateMachineInterface* sm) {
    const int role_count = sm->getRoleCount();

    string res = "(";
    for (int ii=0; ii<role_count; ii++) {
        if (ii > 0) {
            res += " ";
        }

        int choice = move.get(ii);
        res += sm->legalToMove(ii, choice);
    }

    res += ")";

    return res;
}

string finalisedString(const PuctNodeChild* child) {
    if (child->to_node == nullptr) {
        return "?";
    } else if (child->to_node->force_terminal) {
        return "Z";
    } else if (child->to_node->isTerminal()) {
        return "T";
    } else if (child->to_node->is_finalised) {
        return "F";
    } else {
        return "*";
    }
}

void PuctNode::dumpNode(const PuctNode* node,
                        const PuctNodeChild* highlight,
                        const std::string& indent,
                        bool sort_by_next_probability,
                        GGPLib::StateMachineInterface* sm) {

    const int role_count = sm->getRoleCount();


    string finalised_top = node->isTerminal() ? "[Terminal]" : (node->is_finalised ? "[Final]" : ".");
    K273::l_verbose("%s(%d) :: %s / #childs %d / %s / Depth: %d, Lead : %d / PUCT %.2f",
                    indent.c_str(),
                    node->visits,
                    scoreString(node, sm, true).c_str(),
                    node->num_children,
                    finalised_top.c_str(),
                    node->game_depth,
                    node->lead_role_index,
                    node->puct_constant);


    auto children = PuctNode::sortedChildren(node, role_count, sort_by_next_probability);

    for (auto child : children) {
        string finalised = finalisedString(child);
        string move = moveString(child->move, sm);
        string score = "(----, ----)";
        int visits = 0;
        if (child->to_node != nullptr) {
            score = scoreString(child->to_node, sm);
            visits = child->to_node->visits;
        }

        string msg = K273::fmtString("%s %s %d(%d):%s %.2f/%.2f/%.2f   %s   %.3f/%.3f/%.3f",
                                     indent.c_str(),
                                     move.c_str(),
                                     child->traversals,
                                     visits - child->traversals,
                                     finalised.c_str(),
                                     child->policy_prob_orig * 100,
                                     child->policy_prob * 100,
                                     child->next_prob * 100,
                                     score.c_str(),
                                     child->debug_node_score,
                                     child->debug_puct_score,
                                     child->debug_node_score + child->debug_puct_score);


        if (child == highlight) {
            K273::l_info(msg);
        } else {
            K273::l_debug(msg);
        }
    }
}


/* sorts children first by visits, then by policy_prob */
Children PuctNode::sortedChildren(const PuctNode* node,
                                  int role_count,
                                  bool next_probability) {

    Children children;
    for (int ii=0; ii<node->num_children; ii++) {
        const PuctNodeChild* child = node->getNodeChild(role_count, ii);
        children.push_back(child);
    }

    auto f = [next_probability](const PuctNodeChild* a, const PuctNodeChild* b) {
        int visits_a = a->to_node == nullptr ? 0 : a->to_node->visits;
        int visits_b = b->to_node == nullptr ? 0 : b->to_node->visits;

        if (visits_a == visits_b) {
            if (next_probability) {
                return a->next_prob > b->next_prob;
            } else {
                return a->policy_prob > b->policy_prob;
            }
        }

        return visits_a > visits_b;
    };

    std::sort(children.begin(), children.end(), f);
    return children;
}

/* sorts children first by traversals, then by policy_prob */
Children PuctNode::sortedChildrenTraversals(const PuctNode* node,
                                            int role_count,
                                            bool next_probability) {

    Children children;
    for (int ii=0; ii<node->num_children; ii++) {
        const PuctNodeChild* child = node->getNodeChild(role_count, ii);
        children.push_back(child);
    }

    auto f = [next_probability](const PuctNodeChild* a, const PuctNodeChild* b) {
        int traversals_a = a->traversals;
        int traversals_b = b->traversals;

        if (traversals_a == traversals_b) {
            if (next_probability) {
                return a->next_prob > b->next_prob;
            } else {
                return a->policy_prob > b->policy_prob;
            }
        }

        return traversals_a > traversals_b;
    };

    std::sort(children.begin(), children.end(), f);
    return children;
}

///////////////////////////////////////////////////////////////////////////////

const GGPLib::BaseState* PuctNodeRequest::getBaseState() const {
    return this->node->getBaseState();
}

void PuctNodeRequest::add(float* buf, const GdlBasesTransformer* transformer) {
    std::vector <const GGPLib::BaseState*> prev_states;
    const PuctNode* cur = this->node->parent;

    for (int ii=0; ii<transformer->getNumberPrevStates(); ii++) {
        if (cur != nullptr) {
            prev_states.push_back(cur->getBaseState());
            cur = cur->parent;
        }
    }

    transformer->toChannels(node->getBaseState(), prev_states, buf);
}

void PuctNodeRequest::reply(const ModelResult& result,
                            const GdlBasesTransformer* transformer) {

    const int role_count = transformer->getNumberPolicies();
    // Update children in new_node with prediction
    float total_prediction = 0.0f;

    auto* raw_policy = result.getPolicy(node->lead_role_index);
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(role_count, ii);

        c->policy_prob_orig = raw_policy[c->move.get(node->lead_role_index)];

        // give each time at least some probability
        c->policy_prob_orig = std::max(0.001f, c->policy_prob_orig);
        total_prediction += c->policy_prob_orig;
    }

    // XXX well this can't happen (since we set a minimum for each child)
    ASSERT(total_prediction > std::numeric_limits<float>::min());

    // normalise:
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(role_count, ii);
        c->policy_prob_orig /= total_prediction;
        c->policy_prob = c->policy_prob_orig;
    }

    for (int ri=0; ri<role_count; ri++) {
        float s = result.getReward(ri);
        if (transformer->getNumberRewards() == 3) {
            float mid = result.getReward(2) / 2.0f;
            s += mid;
        }

        // clamp
        if (s > 1.0) {
            s = 1.0f;

        } else if (s < 0.0) {
            s = 0.0f;
        }

        node->setFinalScore(ri, s);

        // clamp and set current score
        node->setCurrentScore(ri, node->getFinalScore(ri, true));
    }
}

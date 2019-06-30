
#include "puct/evaluator.h"
#include "puct/node.h"

#include "scheduler.h"
#include "gdltransformer.h"

#include <k273/util.h>
#include <k273/logging.h>
#include <k273/strutils.h>
#include <k273/exception.h>

#include <cmath>
#include <climits>
#include <unistd.h>
#include <random>
#include <numeric>
#include <string>
#include <vector>

using namespace GGPZero;

#include "unify.cpp"

///////////////////////////////////////////////////////////////////////////////

PuctEvaluator::PuctEvaluator(GGPLib::StateMachineInterface* sm,
                             const PuctConfig* conf, NetworkScheduler* scheduler,
                             const GGPZero::GdlBasesTransformer* transformer) :
    sm(sm),
    basestate_expand_node(nullptr),
    conf(conf),
    scheduler(scheduler),
    game_depth(0),
    evaluations(0),
    initial_root(nullptr),
    root(nullptr),
    number_of_nodes(0),
    node_allocated_memory(0) {

    this->basestate_expand_node = this->sm->newBaseState();

    this->updateConf(this->conf);
}

PuctEvaluator::~PuctEvaluator() {
    free(this->basestate_expand_node);

    this->reset(0);

    delete this->conf;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::updateConf(const PuctConfig* conf) {
    if (conf->verbose) {
        K273::l_verbose("config verbose: %d, max_dump_depth: %d",
                        conf->verbose, conf->max_dump_depth);

        K273::l_verbose("puct_constant: %f, root_expansions_preset_visits: %d",
                        conf->puct_constant, conf->root_expansions_preset_visits);

        K273::l_verbose("dirichlet_noise (pct: %.2f), fpu_prior_discount: %.2f/%.2f",
                        conf->dirichlet_noise_pct, conf->fpu_prior_discount, conf->fpu_prior_discount_root);

        K273::l_verbose("noise policy squash (pct: %.2f, prob: %.2f),",
                        conf->noise_policy_squash_pct, conf->noise_policy_squash_prob);

        K273::l_verbose("choose: %s",
                        (conf->choose == ChooseFn::choose_top_visits) ? "choose_top_visits" : "choose_temperature");

        K273::l_verbose("temperature: %.2f, start(%d), stop(%d), incr(%.2f), max(%.2f) scale(%.2f)",
                        conf->temperature, conf->depth_temperature_start, conf->depth_temperature_stop,
                        conf->depth_temperature_max, conf->depth_temperature_increment, conf->random_scale);

        K273::l_verbose("top_visits_best_guess_converge_ratio: %.2f",
                        conf->top_visits_best_guess_converge_ratio);

        K273::l_verbose("evaluation_multiplier convergence %.2f",
                        conf->evaluation_multiplier_to_convergence);
    }

    this->conf = conf;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::removeNode(PuctNode* node) {
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        if (child->to_node != nullptr) {
            this->removeNode(child->to_node);
        }

        child->to_node = nullptr;
    }

    this->node_allocated_memory -= node->allocated_size;

    free(node);
    this->number_of_nodes--;
}

PuctNode* PuctEvaluator::createNode(PuctNode* parent, const GGPLib::BaseState* state) {
    PuctNode* new_node = PuctNode::create(state, this->sm);

    // update stats
    this->number_of_nodes++;
    this->node_allocated_memory += new_node->allocated_size;

    new_node->parent = parent;
    if (parent != nullptr) {
        new_node->game_depth = parent->game_depth + 1;
        parent->num_children_expanded++;

    } else {
        new_node->game_depth = 0;
    }

    if (new_node->is_finalised) {
        // hack to try and focus more on winning lines
        // (XXX) actually a very good hack... maybe make it less hacky somehow
        for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
            const float s = new_node->getCurrentScore(ii);
            if (s > 0.99) {
                new_node->setCurrentScore(ii, 1.05);
            } else if (s < 0.01) {
                new_node->setCurrentScore(ii, -0.05);
            }
        }

    } else {
        // don't evaluate in this case.  Note the score will be invalid...
        if (new_node->num_children == 1) {
            return new_node;
        }

        // goodbye kansas
        PuctNodeRequest req(new_node);
        this->scheduler->evaluate(&req);
        this->evaluations++;
    }

    return new_node;
}

void PuctEvaluator::expandChild(PuctNode* parent, PuctNodeChild* child) {
    // update the statemachine
    this->sm->updateBases(parent->getBaseState());
    this->sm->nextState(&child->move, this->basestate_expand_node);

    // create node
    child->to_node = this->createNode(parent, this->basestate_expand_node);
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::backPropagate(float* new_scores) {
    const int start_index = this->path.size() - 1;

    // back propagation:
    for (int index=start_index; index >= 0; index--) {
        const PathElement& cur = this->path[index];

        const float visits = cur.to_node->visits;

        for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
            const float score = ((visits * cur.to_node->getCurrentScore(ii) + new_scores[ii]) /
                                 (visits + 1.0));

            cur.to_node->setCurrentScore(ii, score);
        }

        cur.to_node->visits++;
    }
}

PuctNodeChild* PuctEvaluator::selectChild(PuctNode* node, int depth) {
    ASSERT(!node->isTerminal());
    ASSERT(node->num_children > 0);

    // dynamically set the PUCT constant
    this->setPuctConstant(node, depth);

    if (node->num_children == 1) {
        return node->getNodeChild(this->sm->getRoleCount(), 0);
    }

    if (depth < 2) {
        this->setDirichletNoise(node);
    }

    const float sqrt_node_visits = std::sqrt(node->visits + 1);
    const float prior_score = this->priorScore(node, depth);

    // get best:
    float best_score = -1;
    PuctNodeChild* best_child = nullptr;

    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);

        float child_visits = 0.0f;

        float node_score = prior_score;

        float child_pct = c->policy_prob;
        if (c->to_node != nullptr) {
            PuctNode* cn = c->to_node;
            child_visits = cn->visits;
            node_score = cn->getCurrentScore(node->lead_role_index);

            // ensure finalised are enforced more than other nodes (network can return 1.0f for
            // basically dumb moves, if it thinks it will win regardless)
            // XXX isn't this redundant now, cause of 1.05 for wins?
            if (cn->is_finalised) {
                if (node_score > 0.99) {
                    if (depth > 0) {
                        return c;
                    }

                    node_score *= 1.0f + node->puct_constant;

                } else {
                    // no more exploration for you
                    child_pct = 0.0;
                }
            }
        }

        float cv = child_visits + 1;
        float puct_score = node->puct_constant * child_pct * (sqrt_node_visits / cv);

        // end product
        float score = node_score + puct_score;

        // use for debug/display
        c->debug_node_score = node_score;
        c->debug_puct_score = puct_score;

        if (score > best_score) {
            best_child = c;
            best_score = score;
        }
    }

    ASSERT(best_child != nullptr);
    return best_child;
}

int PuctEvaluator::treePlayout() {
    PuctNode* current = this->root;
    ASSERT(current != nullptr && !current->isTerminal());

    int tree_playout_depth = 0;

    this->path.clear();
    float scores[this->sm->getRoleCount()];

    PuctNodeChild* child = nullptr;

    while (true) {
        ASSERT(current != nullptr);
        this->path.emplace_back(child, current);

        // End of the road
        // ZZZ isFinalised???
        if (current->isTerminal()) {
            break;
        }

        // Choose selection
        PuctNodeChild* child = this->selectChild(current, tree_playout_depth);

        // if does not exist, then create it (will incur a nn prediction)
        if (child->to_node == nullptr) {
            this->expandChild(current, child);

            if (this->conf->use_legals_count_draw > 0) {
                this->checkDrawStates(current, child->to_node);
            }

            current = child->to_node;

            // end of the road.  We continue if num_children == 1, since there is nothing to
            // select
            if (current->is_finalised || current->num_children > 1) {
                this->path.emplace_back(child, current);
                break;
            }
        }

        current = child->to_node;
        tree_playout_depth++;
    }

    for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
        scores[ii] = current->getCurrentScore(ii);
    }

    this->backPropagate(scores);
    return tree_playout_depth;
}

void PuctEvaluator::playoutLoop(int max_evaluations, double end_time) {
    int max_depth = -1;
    int total_depth = 0;

    // configurable XXX?  Will only run at the very end of the game, and it is really only here so
    // we exit in a small finite amount of time

    // XXX normally constrained by evaluations anyways
    int max_iterations = max_evaluations * 2;

    if (max_evaluations < 0) {
        max_iterations = INT_MAX;
    }

    int iterations = 0;
    this->evaluations = 0;
    double start_time = K273::get_time();

    double next_report_time = -1;

    if (this->conf->matchmode) {
        next_report_time = K273::get_time() + 2.5;
    }

    while (iterations < max_iterations) {
        if (max_evaluations > 0 && this->evaluations > max_evaluations) {

            if (this->converged(8)) {
                break;

            } else {
                const int max_convergence_evaluations = max_evaluations * this->conf->evaluation_multiplier_to_convergence;
                if (this->evaluations > max_convergence_evaluations) {
                    break;
                }
            }
        }

        if (end_time > 0 && K273::get_time() > end_time) {
            break;
        }

        int depth = this->treePlayout();
        max_depth = std::max(depth, max_depth);
        total_depth += depth;

        iterations++;

        if (next_report_time > 0 && K273::get_time() > next_report_time) {
            next_report_time = K273::get_time() + 2.5;

            const PuctNodeChild* best = this->chooseTopVisits(this->root);
            if (best->to_node != nullptr) {
                const int our_role_index = this->root->lead_role_index;
                const int choice = best->move.get(our_role_index);
                K273::l_info("Evals %d/%d, depth %.2f/%d, best: %.4f, move: %s",
                             this->evaluations, iterations,
                             total_depth / float(iterations), max_depth,
                             best->to_node->getCurrentScore(our_role_index),
                             this->sm->legalToMove(our_role_index, choice));
            }
        }
    }

    if (this->conf->verbose) {
        if (iterations) {
            K273::l_info("Time taken for %d/%d evaluations/iterations in %.3f seconds",
                         this->evaluations, iterations, K273::get_time() - start_time);

            K273::l_debug("The average depth explored: %.2f, max depth: %d",
                          total_depth / float(iterations), max_depth);
        } else {
            K273::l_debug("Did no iterations.");
        }
    }
}



//////////////////////////////////////////////////////////////////////

PuctNode* PuctEvaluator::fastApplyMove(const PuctNodeChild* next) {
    ASSERT(this->root != nullptr);
    ASSERT(this->initial_root != nullptr);

    const int number_of_nodes_before = this->number_of_nodes;

    PuctNode* new_root = nullptr;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);

        if (c == next) {
            ASSERT(new_root == nullptr);
            if (c->to_node == nullptr) {
                this->expandChild(this->root, c);
            }

            new_root = c->to_node;

        } else {
            if (c->to_node != nullptr) {
                this->removeNode(c->to_node);

                // avoid a double delete at end of game
                c->to_node = nullptr;
            }
        }
    }

    ASSERT(new_root != nullptr);

    this->root = new_root;
    this->game_depth++;

    if (this->conf->verbose && number_of_nodes_before - this->number_of_nodes > 0) {
        K273::l_info("deleted %d nodes", number_of_nodes_before - this->number_of_nodes);
    }

    return this->root;
}

void PuctEvaluator::applyMove(const GGPLib::JointMove* move) {
    // XXX this is only here for the player.  We should probably have a player class, and not
    // simplify code greatly.

    bool found = false;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);
        if (c->move.equals(move)) {
            this->fastApplyMove(c);
            found = true;
            break;
        }
    }

    std::string move_str = PuctNode::moveString(*move, this->sm);

    if (this->conf->verbose) {
        if (!found) {
            K273::l_warning("PuctEvaluator::applyMove(): Did not find move %s",
                            move_str.c_str());
        } else {
            K273::l_info("PuctEvaluator::applyMove(): %s", move_str.c_str());
        }
    }

    ASSERT(this->root != nullptr);
}

void PuctEvaluator::reset(int game_depth) {
    // really free all
    if (this->initial_root != nullptr) {
        this->removeNode(this->initial_root);
        this->initial_root = nullptr;
        this->root = nullptr;
    }

    ASSERT(this->root == nullptr);

    if (this->number_of_nodes) {
        K273::l_warning("Number of nodes not zero %d", this->number_of_nodes);
    }

    if (this->node_allocated_memory) {
        K273::l_warning("Leaked memory %ld", this->node_allocated_memory);
    }

    // this is the only place we set game_depth
    this->game_depth = game_depth;
}

PuctNode* PuctEvaluator::establishRoot(const GGPLib::BaseState* current_state) {
    ASSERT(this->root == nullptr && this->initial_root == nullptr);

    if (current_state == nullptr) {
        current_state = this->sm->getInitialState();
    }

    this->initial_root = this->root = this->createNode(nullptr, current_state);
    this->root->game_depth = this->game_depth;

    ASSERT(!this->root->isTerminal());
    return this->root;
}

const PuctNodeChild* PuctEvaluator::onNextMove(int max_evaluations, double end_time) {
    ASSERT(this->root != nullptr && this->initial_root != nullptr);

    if (this->conf->root_expansions_preset_visits > 0) {
        int number_of_expansions = 0;
        for (int ii=0; ii<this->root->num_children; ii++) {
            PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);

            if (c->to_node == nullptr) {
                this->expandChild(this->root, c);

                // should be traversal on child XXX wait for puctplus
                c->to_node->visits = std::max(c->to_node->visits,
                                              this->conf->root_expansions_preset_visits);

                number_of_expansions++;
            }
        }
    }

    this->playoutLoop(max_evaluations, end_time);

    const PuctNodeChild* choice = this->choose(this->root);

    // this is a hack to only show tree when it is our 'turn'.  Be better to use bypass opponent turn
    // flag than abuse this value (XXX).
    if (max_evaluations != 0 && this->conf->verbose) {
        this->logDebug(choice);
    }

    return choice;
}


Children PuctEvaluator::getProbabilities(PuctNode* node, float temperature, bool use_policy) {
    // XXX this makes the assumption that our legals are unique for each child.

    ASSERT(node->num_children > 0);

    // we add 0.001 to each our children, so zero chance doesn't happen
    float node_visits = node->visits + 0.001 * node->num_children;

    float total_probability = 0.0f;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        float child_visits = child->to_node ? child->to_node->visits + 0.001f : 0.001f;
        if (use_policy) {
            child->next_prob = child->policy_prob + 0.001f;

        } else {
            child->next_prob = child_visits / node_visits;
        }

        // apply temperature
        child->next_prob = ::pow(child->next_prob, temperature);
        total_probability += child->next_prob;
    }

    // normalise it
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        child->next_prob /= total_probability;
    }

    return PuctNode::sortedChildren(node, this->sm->getRoleCount(), true);
}


const PuctNodeChild* PuctEvaluator::chooseTopVisits(const PuctNode* node) const {
    ASSERT(node != nullptr);

    auto children = PuctNode::sortedChildren(node, this->sm->getRoleCount());
    ASSERT(children.size() > 0);

    const int role_index = node->lead_role_index;

    // look for finalised first
    if (node->is_finalised && node->getCurrentScore(role_index) > 1.0) {
        for (auto c : children) {
            if (c->to_node != nullptr && c->to_node->is_finalised &&
                c->to_node->getCurrentScore(role_index) > 0.99) {
                return c;
            }
        }
    }

    // compare top two.  This is a heuristic to cheaply check if the node hasn't yet converged and
    // chooses the one with the best score.  It isn't very accurate, the only way to get 100%
    // accuracy is to keep running for longer, until it cleanly converges.  This is a best guess for now.
    if (this->conf->top_visits_best_guess_converge_ratio > 0 && children.size() >= 2) {
        const PuctNodeChild* c0 = children[0];
        const PuctNodeChild* c1 = children[1];

        if (c0->to_node != nullptr && c1->to_node != nullptr) {
            if (c1->to_node->visits > c0->to_node->visits * this->conf->top_visits_best_guess_converge_ratio &&
                c1->to_node->getCurrentScore(role_index) > c0->to_node->getCurrentScore(role_index)) {
                return c1;
            } else {
                return c0;
            }
        }
    }

    return children[0];
}


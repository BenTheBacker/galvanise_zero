import os
import math
import time
import random
import numpy as np

from keras.models import model_from_json

from ggplib.util import log
from ggplib.player.base import MatchPlayer

from ggplearn.utils.bt import pretty_print_board
from ggplearn import net_config

VERBOSE = False

class Child(object):
    def __init__(self, parent, move, legal):
        self.parent = parent
        self.move = move
        self.legal = legal

        # from NN
        self.p_visits_pct = None

        # to the next node
        self.to_node = None

    def __repr__(self):
        n = self.to_node
        ri = self.parent.lead_role_index
        if n and n.predicted:
            prior_score = n.prior_scores[ri] or 0.0
            final_score = n.final_scores[ri] or 0.0
            if n.is_terminal:
                prior_score = -1
                final_score = n.terminal_scores[ri] / 100.0

            return "%s %.2f %.2f/%.2f %s" % (self.move,
                                             self.p_visits_pct * 100,
                                             prior_score,
                                             final_score,
                                             "terminal" if n.is_terminal else "")
        else:
            return self.move
    __str__ = __repr__


class Node(object):
    def __init__(self, state, lead_role_index, is_terminal):
        self.state = state
        self.lead_role_index = lead_role_index
        self.is_terminal = is_terminal
        self.children = []

        self.predicted = False

        # from NN
        self.prior_scores = None
        self.final_scores = None

        # from sm.get_goal_value() (0 - 100)
        self.terminal_scores = None

        self.mcts_visits = None
        self.mcts_score = None

    def add_child(self, move, legal):
        self.children.append(Child(self, move, legal))

    @property
    def expanded(self):
        for c in self.children:
            if c.to_node is None:
                return False
        return True

    def get_top_k(self, k):
        assert self.predicted

        children = self.children[:]
        children.sort(key=lambda c: c.p_visits_pct, reverse=True)
        return children[:k]


def get_legals(ls):
    return [ls.get_legal(i) for i in range(ls.get_count())]


def bs_to_state(bs):
    return tuple(bs.get(i) for i in range(bs.len()))


def state_to_bs(state, bs):
    for i, v in enumerate(state):
        bs.set(i, v)

def opp(role_index):
    if role_index:
        return 0
    else:
        return 1

models_path = os.path.join(os.environ["GGPLEARN_PATH"], "src", "ggplearn", "models")


class NNMonteCarlo(MatchPlayer):
    NUM_OF_PLAYOUTS_PER_ITERATION = 42

    UCB_CONSTANT = 0.2
    POLICY_VALUE = 16

    def __init__(self, postfix):
        MatchPlayer.__init__(self, "NNMonteCarlo-" + postfix)
        self.postfix = postfix
        self.nn_config = None
        self.root = None

    def on_meta_gaming(self, finish_time):
        self.root = None
        log.info("NNExpander, match id: %s" % self.match.match_id)

        sm = self.match.sm
        game_info = self.match.game_info

        if self.nn_config is None:
            self.nn_config = net_config.get_bases_config(game_info.game)
            self.base_infos = net_config.create_base_infos(self.nn_config, game_info.model)

            # load neural network model and weights
            model_filename = os.path.join(models_path, "model_nn_%s_%s.json" % (game_info.game, self.postfix))
            weights_filename = os.path.join(models_path, "weights_nn_%s_%s.h5" % (game_info.game, self.postfix))

            with open(model_filename, "r") as f:
                self.nn_model = model_from_json(f.read())

            self.nn_model.load_weights(weights_filename)

            # cache joint move, and basestate
            self.joint_move = sm.get_joint_move()
            self.basestate_expand_node = sm.new_base_state()
            self.basestate_expanded_node = sm.new_base_state()

            def get_noop_idx(actions):
                for idx, a in enumerate(actions):
                    if "noop" in a:
                        return idx
                assert False, "did not find noop"

            self.role0_noop_legal, self.role1_noop_legal = map(get_noop_idx, game_info.model.actions)
        self.nodes_to_predict = []

    def predict_n(self, nodes):
        num_states = len(nodes)

        X_0 = [net_config.state_to_channels(n.state,
                                            n.lead_role_index,
                                            self.nn_config,
                                            self.base_infos) for n in nodes]

        X_0 = np.array(X_0).reshape(num_states, self.nn_config.num_rows,
                                    self.nn_config.num_cols, self.nn_config.num_channels)

        X_1 = [[v for v, base_info in zip(n.state, self.base_infos)
                if base_info.channel is None] for n in nodes]
        X_1 = np.array(X_1).reshape(num_states, len(X_1[0]))

        s = time.time()
        Y = self.nn_model.predict([X_0, X_1], batch_size=num_states)

        assert len(Y) == 3

        result = []
        for i in range(num_states):
            policy, scores0, scores1 = Y[0][i], Y[1][i], Y[2][i]
            result.append((policy, scores0, scores1))

        return result

    def predict_1(self, node):
        return self.predict_n([node])[0]

    def do_predictions(self):
        result = self.predict_n(self.nodes_to_predict)
        for node, (pred_policy, priors, finals) in zip(self.nodes_to_predict, result):
            self.update_node(node, pred_policy, priors, finals)
        self.nodes_to_predict = []

    def create_node(self, basestate):
        sm = self.match.sm
        sm.update_bases(basestate)

        if sm.get_legal_state(0).get_count() == 1 and sm.get_legal_state(0).get_legal(0) == self.role0_noop_legal:
            lead_role_index = 1
        else:
            assert (sm.get_legal_state(1).get_count() == 1 and
                    sm.get_legal_state(1).get_legal(0) == self.role1_noop_legal)
            lead_role_index = 0

        node = Node(bs_to_state(basestate),
                    lead_role_index,
                    sm.is_terminal())

        ls = sm.get_legal_state(0) if lead_role_index == 0 else sm.get_legal_state(1)

        for l in get_legals(ls):
            node.add_child(sm.legal_to_move(lead_role_index, l), l)

        if node.is_terminal:
            node.terminal_scores = [sm.get_goal_value(i) for i in range(2)]
            node.mcts_visits = 1
            node.mcts_score = [s/100.0 for s in node.terminal_scores]

        return node

    def expand_child(self, child):
        sm = self.match.sm
        assert child.to_node is None
        node = child.parent
        state_to_bs(node.state, self.basestate_expand_node)
        sm.update_bases(self.basestate_expand_node)

        if node.lead_role_index == 0:
            self.joint_move.set(1, self.role1_noop_legal)
        else:
            self.joint_move.set(0, self.role0_noop_legal)

        self.joint_move.set(node.lead_role_index, child.legal)
        sm.next_state(self.joint_move, self.basestate_expanded_node)

        child.to_node = self.create_node(self.basestate_expanded_node)

    def expand_children(self, children):
        for c in children:
            self.expand_child(c)

    def update_node(self, node, pred_policy, priors, finals):
        assert not node.predicted
        if node.lead_role_index == 0:
            start_pos = 0
        else:
            start_pos = len(self.match.game_info.model.actions[0])

        for c in node.children:
            ridx = start_pos + c.legal
            c.p_visits_pct = pred_policy[ridx]

        node.prior_scores = priors
        node.final_scores = finals
        node.predicted = True

        node.mcts_visits = 1
        node.mcts_score = [(x+y)/2.0 for x, y in zip(node.final_scores, node.prior_scores)]

    def back_propagate(self, path, scores):
        for child in reversed(path):
            assert child.to_node is not None

            p = child.parent
            for i, s in enumerate(scores):
                p.mcts_score[i] = (p.mcts_visits * p.mcts_score[i] + s) / float(p.mcts_visits + 1)
            p.mcts_visits += 1

    def select_child(self, node, depth):
        assert node.mcts_visits >= 1
        log_visits = math.log(node.mcts_visits)

        # get best
        best_child = None
        best_score = -10000000

        for child in node.children:
            cn = child.to_node

            child_visits = 1.0
            node_score = node.prior_scores[node.lead_role_index]
            if cn is not None:
                # break early?
                if cn.is_terminal and cn.terminal_scores[node.lead_role_index] == 100:
                    return child

                child_visits = float(cn.mcts_visits)
                node_score = cn.mcts_score[node.lead_role_index]
            else:
                # tiny bit of randomness
                node_score += random.random() / 100.0

            ucb_score = self.UCB_CONSTANT * math.sqrt(log_visits / child_visits)
            policy_score = (self.POLICY_VALUE * child.p_visits_pct) / child_visits
            score = node_score + ucb_score + policy_score

            # use for debug/display
            child.tmp_node_score = node_score
            child.tmp_ucb_score = ucb_score
            child.tmp_policy_score = policy_score

            if score > best_score:
                best_child = child
                best_score = score

        return best_child

    def playout(self, current):
        path = []
        depth = 0
        scores = None
        while True:
            if current.is_terminal:
                scores = current.mcts_score
                break

            child = self.select_child(current, depth)
            path.append(child)

            if child.to_node is None:
                self.expand_child(child)
                if not child.to_node.is_terminal:
                    self.nodes_to_predict.append(child.to_node)
                    self.do_predictions()
                scores = child.to_node.mcts_score
                break

            current = child.to_node
            depth += 1

        assert scores is not None
        self.back_propagate(path, scores)

    def on_apply_move(self, joint_move):
        # need to fish for it in children?
        if self.root is not None:
            lead, other = self.root.lead_role_index, opp(self.root.lead_role_index)
            if other == 0:
                assert joint_move.get(other) == self.role0_noop_legal
            else:
                assert joint_move.get(other) == self.role0_noop_legal

            played = joint_move.get(lead)

            for c in self.root.children:
                c.parent = None
                if c.legal == played:
                    found = True
                    # might be none, that is fine
                    new_root = c.to_node

            assert found
            self.root = new_root

            def visit_count(node):
                if node is None:
                    return 0
                total = 1
                for c in node.children:
                    total += visit_count(c.to_node)
                return total

            if VERBOSE:
                print "NEW ROOT:", new_root, visit_count(new_root)

    def on_next_move(self, finish_time):
        sm = self.match.sm
        sm.update_bases(self.match.get_current_state())

        if self.root is not None:
            assert self.root.state == bs_to_state(self.match.get_current_state())
        else:
            if VERBOSE:
                print 'creating root'

            self.root = self.create_node(self.match.get_current_state())
            assert not self.root.is_terminal

            # predict root
            pred_policy, priors, finals = self.predict_1(self.root)
            self.update_node(self.root, pred_policy, priors, finals)

        if VERBOSE and self.match.game_info.game == "breakthrough":
            pretty_print_board(sm, self.root.state)
            print

        if self.root.lead_role_index != self.match.our_role_index:
            if self.match.our_role_index:
                return self.role1_noop_legal
            else:
                return self.role0_noop_legal

        s = time.time()
        for i in range(self.NUM_OF_PLAYOUTS_PER_ITERATION):
            if time.time() + 1 > finish_time:
                print "RAN OUT OF TIME"
                break
            self.playout(self.root)

        print "Time taken for %s iteratons %.1f" % (self.NUM_OF_PLAYOUTS_PER_ITERATION, time.time() - s)

        best = None
        best_visits = 0

        def f(x): return -1 if x.to_node is None else x.to_node.mcts_visits
        children = self.root.children[:]
        children.sort(key=f, reverse=True)

        best = children[0]

        if VERBOSE:
            for child in children:
                print child, "\t->\t",
                if child.to_node is not None:
                    n = child.to_node
                    print "node", n.mcts_visits, n.mcts_score,
                else:
                    print "NONE",

                print '\t\t', child.tmp_node_score, child.tmp_ucb_score, child.tmp_policy_score

            print "BEST", best

        return best.legal

###############################################################################

def main():
    import sys
    from twisted.internet import reactor
    from twisted.web import server

    from ggplib.util import log
    from ggplib.server import GGPServer
    from ggplib import interface

    port = int(sys.argv[1])
    model_postfix = sys.argv[2]

    player_name = NNMonteCarlo.__class__.__name__ + "_" + model_postfix
    interface.initialise_k273(1, log_name_base=player_name)
    log.initialise()

    # this still uses more than once cpu :(
    import tensorflow as tf
    config = tf.ConfigProto(device_count=dict(CPU=1),
                            allow_soft_placement=False,
                            log_device_placement=False,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(config=config)

    from keras import backend
    backend.set_session(sess)

    player = NNMonteCarlo(model_postfix)

    ggp = GGPServer()
    ggp.set_player(player)
    site = server.Site(ggp)

    reactor.listenTCP(port, site)
    reactor.run()


if __name__ == "__main__":
    main()

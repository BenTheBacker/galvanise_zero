import time
import random
from operator import itemgetter, attrgetter

import numpy as np
from tabulate import tabulate

from ggplib.util import log
from ggplib.player.base import MatchPlayer

from ggplearn.util.bt import pretty_print_board

from ggplearn import msgdefs

from ggplearn.nn import bases


###############################################################################

class Child(object):
    def __init__(self, parent, move, legal):
        self.parent = parent
        self.move = move
        self.legal = legal

        # from NN
        self.policy_prob = None

        # to the next node
        # this deviates from AlphaGoZero paper, where the keep statistics on child.  But I am
        # following how I did things in galvanise, as it is simpler to keep it my head.
        self.to_node = None

        # debug
        self.debug_node_score = -1
        self.debug_puct_score = -1

    def visits(self):
        if self.to_node is None:
            return 0
        return self.to_node.mc_visits

    def __repr__(self):
        n = self.to_node
        if n:
            ri = self.parent.lead_role_index
            if n.is_terminal:
                score = n.terminal_scores[ri] / 100.0
            else:
                score = n.final_score[ri] or 0.0

            return "%s %d %.2f%%   %.2f %s" % (self.move,
                                               self.visits(),
                                               self.policy_prob * 100,
                                               score,
                                               "T " if n.is_terminal else "* ")
        else:
            return "%s %d %.2f%%   ---- ? " % (self.move,
                                               self.visits(),
                                               self.policy_prob * 100)
    __str__ = __repr__


class Node(object):
    def __init__(self, state, lead_role_index, is_terminal):
        self.state = state
        self.lead_role_index = lead_role_index
        self.is_terminal = is_terminal
        self.children = []

        self.predicted = False

        # from NN
        self.final_score = None

        # from sm.get_goal_value() (0 - 100)
        self.terminal_scores = None

        self.mc_visits = 0
        self.mc_score = None

    def add_child(self, move, legal):
        self.children.append(Child(self, move, legal))

    def sorted_children(self, by_score=False):
        ' sorts by mcts visits OR score '

        if not self.children:
            return self.children

        if by_score:
            def f(x):
                return -1 if x.to_node is None else x.to_node.mc_score[self.lead_role_index]
        else:
            def f(x):
                return -1 if x.to_node is None else x.to_node.mc_visits

        children = self.children[:]
        children.sort(key=f, reverse=True)
        return children


class PUCTPlayer(MatchPlayer):
    def __init__(self, conf=None):
        if conf is None:
            conf = msgdefs.PUCTPlayerConf()
        self.conf = conf

        self.nn = None
        self.root = None

        if self.conf.verbose:
            assert self.conf.playouts_per_iteration_noop > 0, "DONT KNOW WHY THIS DOESNT WORK, BUT XXX"

        self.choose = getattr(self, self.conf.choose)

        identifier = "%s_%s_%s" % (self.conf.name, self.conf.playouts_per_iteration, conf.generation)
        MatchPlayer.__init__(self, identifier)

    def on_meta_gaming(self, finish_time):
        if self.conf.verbose:
            log.info("PUCTPlayer, match id: %s" % self.match.match_id)

        self.root = None

        sm = self.match.sm
        game_info = self.match.game_info

        # this is a performance hack, where once we get the nn/config we don't reget it.
        # if latest is set will always get the latest

        if self.conf.generation == 'latest' or self.nn is None:
            self.bases_config = bases.get_config(game_info.game, game_info.model, self.conf.generation)
            self.nn = self.bases_config.create_network()
            self.nn.load()

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

        self.our_noop_legal = self.role1_noop_legal if self.match.our_role_index == 1 else self.role0_noop_legal

        self.root = None
        self.nodes_to_predict = []

    def update_node_policy(self, node, pred_policy):
        if node.lead_role_index == 0:
            start_pos = 0
        else:
            start_pos = len(self.match.game_info.model.actions[0])

        total = 0
        for c in node.children:
            ridx = start_pos + c.legal
            c.policy_prob = pred_policy[ridx]

            total += c.policy_prob

        # normalise
        for c in node.children:
            c.policy_prob /= total

        # sort the children now rather than every iteration
        node.children.sort(key=attrgetter("policy_prob"), reverse=True)

    def do_predictions(self):
        actual_nodes_to_predict = []
        for node in self.nodes_to_predict:
            if node.is_terminal:
                node.mc_score = [s / 100.0 for s in node.terminal_scores]
            else:
                assert not node.predicted
                actual_nodes_to_predict.append(node)

        self.nodes_to_predict = []

        # nothing to do
        if not actual_nodes_to_predict:
            return

        states = [n.state for n in actual_nodes_to_predict]
        lead_role_indexs = [n.lead_role_index for n in actual_nodes_to_predict]

        result = self.nn.predict_n(states, lead_role_indexs)

        for node, (pred_policy, pred_final_score) in zip(actual_nodes_to_predict,
                                                         result):
            node.predicted = True
            node.final_score = pred_final_score
            node.mc_score = pred_final_score[:]
            self.update_node_policy(node, pred_policy)

    def create_node(self, basestate):
        sm = self.match.sm
        sm.update_bases(basestate)

        if sm.get_legal_state(0).get_count() == 1 and sm.get_legal_state(0).get_legal(0) == self.role0_noop_legal:
            lead_role_index = 1
        else:
            assert (sm.get_legal_state(1).get_count() == 1 and
                    sm.get_legal_state(1).get_legal(0) == self.role1_noop_legal)
            lead_role_index = 0

        node = Node(basestate.to_list(),
                    lead_role_index,
                    sm.is_terminal())

        if node.is_terminal:
            node.terminal_scores = [sm.get_goal_value(i) for i in range(2)]
        else:
            legal_state = sm.get_legal_state(0) if lead_role_index == 0 else sm.get_legal_state(1)
            for l in legal_state.to_list():
                node.add_child(sm.legal_to_move(lead_role_index, l), l)

        return node

    def expand_child(self, child):
        sm = self.match.sm
        assert child.to_node is None
        node = child.parent

        self.basestate_expand_node.from_list(node.state)
        sm.update_bases(self.basestate_expand_node)

        if node.lead_role_index == 0:
            self.joint_move.set(1, self.role1_noop_legal)
        else:
            self.joint_move.set(0, self.role0_noop_legal)

        self.joint_move.set(node.lead_role_index, child.legal)
        sm.next_state(self.joint_move, self.basestate_expanded_node)

        child.to_node = self.create_node(self.basestate_expanded_node)

    def back_propagate(self, path, scores):
        self.count_bp += 1

        for node in reversed(path):
            for i, s in enumerate(scores):
                node.mc_score[i] = (node.mc_visits *
                                    node.mc_score[i] + s) / float(node.mc_visits + 1)
            node.mc_visits += 1

    def dirichlet_noise(self, node, depth):
        if depth != 0:
            return None

        if self.conf.dirichlet_noise_alpha < 0:
            return None

        return np.random.dirichlet([self.conf.dirichlet_noise_alpha] * len(node.children))

    def puct_constant(self, node):
        constant = self.conf.puct_constant_after

        expansions = self.conf.puct_before_root_expansions if node is self.root else self.conf.puct_before_expansions

        expanded = sum(1 for c in node.children if c.to_node is not None)
        if expanded <= expansions:
            constant = self.conf.puct_constant_before

        if self.conf.puct_constant_tune:
            constant *= node.final_score[node.lead_role_index]

        return constant

    def select_child(self, node, depth):
        dirichlet_noise = self.dirichlet_noise(node, depth)
        puct_constant = self.puct_constant(node)

        # get best
        best_child = None
        best_score = -1

        for idx, child in enumerate(node.children):
            cn = child.to_node

            child_visits = 0.0

            # prior... (alpha go zero said 0 but there score ranges from [-1,1]
            node_score = 0.0

            if cn is not None:
                child_visits = float(cn.mc_visits)
                node_score = cn.mc_score[node.lead_role_index]

                # ensure terminals are enforced more than other nodes
                if cn.is_terminal:
                    node_score *= 1.02

            child_pct = child.policy_prob

            if dirichlet_noise is not None:
                noise_pct = self.conf.dirichlet_noise_pct
                child_pct = (1 - noise_pct) * child_pct + noise_pct * dirichlet_noise[idx]

            v = float(node.mc_visits + 1)
            cv = float(child_visits + 1)
            puct_score = puct_constant * child_pct * (v ** 0.5) / cv

            score = node_score + puct_score + random.random() * 0.001

            # use for debug/display
            child.debug_node_score = node_score
            child.debug_puct_score = puct_score

            if score > best_score:
                best_child = child
                best_score = score

        assert best_child is not None
        return best_child

    def playout(self, current):
        assert current is not None and not current.is_terminal

        path = []
        scores = None

        while True:
            path.append(current)

            # already expanded terminal
            if current.is_terminal:
                scores = [s / 100.0 for s in current.terminal_scores]
                break

            child = self.select_child(current, len(path) - 1)

            if child.to_node is None:
                self.expand_child(child)
                self.nodes_to_predict.append(child.to_node)
                self.do_predictions()

                scores = child.to_node.mc_score

                path.append(child.to_node)
                break

            current = child.to_node

        assert scores is not None
        self.back_propagate(path, scores)
        return len(path)

    def playout_loop(self, node, finish_time, cb=None):
        max_depth = -1
        total_depth = 0
        iterations = 0

        start_time = time.time()

        if self.root.lead_role_index != self.match.our_role_index:
            max_iterations = self.conf.playouts_per_iteration_noop
        else:
            max_iterations = self.conf.playouts_per_iteration

        while True:
            if time.time() > finish_time:
                log.info("RAN OUT OF TIME")
                break

            depth = self.playout(node)
            max_depth = max(depth, max_depth)
            total_depth += depth

            iterations += 1

            if max_iterations > 0 and iterations > max_iterations:
                break

            if cb and cb():
                break

        if self.conf.verbose:
            log.info("Time taken for %s iteratons %.3f" % (iterations,
                                                           time.time() - start_time))

            log.debug("The average depth explored: %.2f, max depth: %d" % (total_depth / float(iterations),
                                                                           max_depth))

    def on_apply_move(self, joint_move):

        # need to fish for it in children?
        if self.root is not None:
            lead = self.root.lead_role_index
            other = 0 if lead else 1
            if other == 0:
                assert joint_move.get(other) == self.role0_noop_legal
            else:
                assert joint_move.get(other) == self.role1_noop_legal

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

            if self.conf.verbose:
                log.verbose("ROOT FOUND: %s / %d" % (new_root, visit_count(new_root)))

    def noop(self):
        if self.root.lead_role_index != self.match.our_role_index:
            if self.match.our_role_index:
                return self.role1_noop_legal
            else:
                return self.role0_noop_legal
        return None

    def on_next_move(self, finish_time):
        self.count_bp = 0
        sm = self.match.sm
        sm.update_bases(self.match.get_current_state())

        # break early as possible
        if not self.conf.verbose and self.conf.playouts_per_iteration_noop == 0:
            ri = self.match.our_role_index
            ls = sm.get_legal_state(ri)
            if ls.get_count() == 1 and ls.get_legal(0) == self.our_noop_legal:
                return self.our_noop_legal

        start_time = time.time()
        if self.root is not None:
            assert self.root.state == self.match.get_current_state().to_list()
        else:
            if self.conf.verbose:
                log.info('creating root')

            self.root = self.create_node(self.match.get_current_state())
            assert not self.root.is_terminal

            # predict root
            self.nodes_to_predict.append(self.root)

        # s = time.time()
        # for c in self.root.children[:self.conf.expand_root]:
        #     if c.to_node is None:
        #         self.expand_child(c)

        #         for cc in c.to_node.children:
        #             if cc.to_node is None:
        #                 self.expand_child(cc)
        #                 print cc.to_node

        # print "XXXtime taken", time.time() - s

        # we do predictions here and dont combine with expanding some root children (if option is
        # set), because do_predictions() will reorder the children according to the policy and thus
        # expand the highest probabilty moves.
        self.do_predictions()

        # expand and predict some of root children
        if self.conf.expand_root > 0:
            expand_root = 8
            for c in self.root.children[:expand_root]:
                if c.to_node is None:
                    self.expand_child(c)
                    self.nodes_to_predict.append(c.to_node)

            self.do_predictions()

        if self.conf.verbose:
            log.debug("time taken for root %.3f" % (time.time() - start_time))

        self.playout_loop(self.root, finish_time)

        choice = self.choose(finish_time)

        if self.conf.verbose:
            self.debug_output(choice)

        noop_res = self.noop()
        if noop_res is not None:
            return noop_res
        else:
            return choice.legal

    def dump_node(self, node, indent=0):
        indent_str = " " * indent
        role = self.match.sm.get_roles()[node.lead_role_index]
        print "%s>>> lead: %s, visits: %s, predict: %s" % (indent_str,
                                                           role,
                                                           node.mc_visits,
                                                           node.final_score[node.lead_role_index])

        rows = []
        for child, prob in self.get_probabilities(node):
            cols = []

            cols.append(child.move)
            cols.append(child.visits())
            cols.append(child.policy_prob * 100)
            cols.append(prob * 100)

            node_type = '?'
            if child.to_node is not None:
                node_type = "T" if child.to_node.is_terminal else "*"
            cols.append(node_type)

            if child.to_node is not None:
                n = child.to_node
                cols.append(n.mc_score[node.lead_role_index])
            else:
                cols.append(None)

            cols.append(child.debug_puct_score)
            cols.append(child.debug_node_score + child.debug_puct_score)

            rows.append(cols)

        headers = "move visits policy prob type score ~puct ~select".split()
        for line in tabulate(rows, headers, floatfmt=".2f", tablefmt="plain").splitlines():
            print indent_str + line

    def debug_output(self, choice):
        log.verbose("Debug @ depth %s" % self.match.game_depth)

        if self.match.game_info.game == "breakthrough":
            pretty_print_board(self.match.sm, self.root.state)
            print

        current = self.root

        dump_depth = 0
        while current is not None:
            assert not current.is_terminal

            if dump_depth == self.conf.max_dump_depth:
                break

            self.dump_node(current, indent=dump_depth * 4)
            current = current.sorted_children()[0].to_node

            if current.is_terminal:
                break

            dump_depth += 1

        print "Choice", choice

    def get_probabilities(self, node=None, temperature=1):
        if node is None:
            node = self.root

        total_visits = float(sum(c.visits() for c in node.children))

        temps = [((c.visits() + 1) / total_visits) ** temperature for c in node.children]
        temps_tot = sum(temps)

        probs = [(c, t / temps_tot) for c, t in zip(node.children, temps)]
        probs.sort(key=itemgetter(1), reverse=True)

        return probs

    def choose_converge_check(self):
        best_visit = self.root.sorted_children()[0]
        best_score = self.root.sorted_children(by_score=True)[0]
        if best_visit == best_score:
            if self.conf.verbose:
                log.info("Converged - breaking")
            return True
        return False

    def choose_converge(self, finish_time):
        if self.root.lead_role_index != self.match.our_role_index:
            return self.choose_top_visits(finish_time)

        best_visit = self.root.sorted_children()[0]

        score = best_visit.to_node.mc_score[self.root.lead_role_index]
        if score > 0.9 or score < 0.1:
            return best_visit

        best = best_visit
        best_score = self.root.sorted_children(by_score=True)[0]
        if best_visit != best_score:
            if self.conf.verbose:
                log.info("Conflicting between score and visits... visits : %s score : %s" % (best_visit,
                                                                                             best_score))

            store_current_alpha = self.conf.dirichlet_noise_alpha
            self.conf.dirichlet_noise_alpha = -1
            self.playout_loop(self.root, finish_time, self.choose_converge_check)
            self.conf.dirichlet_noise_alpha = store_current_alpha

            best_visit = self.root.sorted_children()[0]

            if self.conf.verbose:
                best_score = self.root.sorted_children(by_score=True)[0]
                if best_visit != best_score:
                    log.info("Failed to converge")

            if best != best_visit:
                if self.conf.verbose:
                    log.info("best visits now: %s -> %s" % (best, best_visit))
                best = best_visit

        if self.conf.verbose:
            log.info("BEST %s" % best)

        return best

    def choose_top_visits(self, finish_time):
        return self.root.sorted_children()[0]


##############################################################################

configs = dict(
    default=msgdefs.PUCTPlayerConf(name="default",
                                   verbose=True,
                                   playouts_per_iteration=42,
                                   playouts_per_iteration_noop=1,
                                   expand_root=100,
                                   dirichlet_noise_alpha=0.2,
                                   puct_before_expansions=3,
                                   puct_before_root_expansions=3,
                                   puct_constant_before=3.0,
                                   puct_constant_after=0.75,
                                   choose="choose_top_visits",
                                   max_dump_depth=2),

    two=msgdefs.PUCTPlayerConf(name="two-test",
                               verbose=True,
                               playouts_per_iteration=800,
                               playouts_per_iteration_noop=800,
                               expand_root=100,
                               dirichlet_noise_alpha=0.1,
                               puct_before_expansions=3,
                               puct_before_root_expansions=3,
                               puct_constant_before=3.0,
                               puct_constant_after=0.75,
                               puct_constant_tune=False,
                               choose="choose_converge",
                               max_dump_depth=2),

    rev=msgdefs.PUCTPlayerConf(name="rev2-test",
                               verbose=True,
                               playouts_per_iteration=42,
                               playouts_per_iteration_noop=1,
                               expand_root=100,
                               dirichlet_noise_alpha=-1,
                               puct_before_expansions=3,
                               puct_before_root_expansions=3,
                               puct_constant_before=0.75,
                               puct_constant_after=0.75,
                               puct_constant_tune=False,
                               choose="choose_top_visits",
                               max_dump_depth=2),

    three=msgdefs.PUCTPlayerConf(name="three-test",
                                 verbose=True,
                                 playouts_per_iteration=42,
                                 playouts_per_iteration_noop=1,
                                 expand_root=100,
                                 dirichlet_noise_alpha=-1,
                                 puct_before_expansions=3,
                                 puct_before_root_expansions=3,
                                 puct_constant_before=3.0,
                                 puct_constant_after=0.75,
                                 puct_constant_tune=False,
                                 choose="choose_converge",
                                 max_dump_depth=2),

    four=msgdefs.PUCTPlayerConf(name="four-test",
                                verbose=True,
                                playouts_per_iteration=42,
                                playouts_per_iteration_noop=1,
                                expand_root=100,
                                dirichlet_noise_alpha=0.1,
                                puct_before_expansions=3,
                                puct_before_root_expansions=8,
                                puct_constant_before=5.0,
                                puct_constant_after=1.5,
                                puct_constant_tune=True,

                                choose="choose_top_visits",
                                max_dump_depth=2))


def main():
    import sys
    from ggplib.play import play_runner
    from ggplearn.util.keras import constrain_resources

    constrain_resources()

    port = int(sys.argv[1])
    generation = sys.argv[2]

    config_name = "default"

    if len(sys.argv) > 3:
        config_name = sys.argv[3]

    conf = configs[config_name]
    conf.generation = generation
    player = PUCTPlayer(conf=conf)
    play_runner(player, port)


if __name__ == "__main__":
    main()

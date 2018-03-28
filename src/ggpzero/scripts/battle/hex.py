import os
import pdb
import sys
import time
import hashlib
import traceback

import tensorflow as tf

from ggplib.db.helper import get_gdl_for_game
# from ggplib.db import lookup
from ggplib.player.gamemaster import GameMaster

from ggpzero.defs import confs
from ggpzero.player.puctplayer import PUCTPlayer


def setup():
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)


def config(gen):
    return confs.PUCTPlayerConfig(name="hex_config",
                                  generation=gen,
                                  verbose=True,

                                  playouts_per_iteration=-1,
                                  playouts_per_iteration_noop=0,

                                  dirichlet_noise_alpha=-1,

                                  root_expansions_preset_visits=1,
                                  puct_before_expansions=3,
                                  puct_before_root_expansions=5,
                                  puct_constant_before=3.0,
                                  puct_constant_after=0.75,

                                  choose="choose_temperature",
                                  temperature=1.5,
                                  depth_temperature_max=3.0,
                                  depth_temperature_start=0,
                                  depth_temperature_increment=0.5,
                                  depth_temperature_stop=6,
                                  random_scale=1.00,

                                  max_dump_depth=2)


def play_game(gen0, gen1, move_time=10.0):
    # add players
    gm = GameMaster(get_gdl_for_game("hexLG11"))
    gm.add_player(PUCTPlayer(config(gen0)), "black")
    gm.add_player(PUCTPlayer(config(gen1)), "white")

    # play move via gamemaster:
    gm.reset()
    gm.start(meta_time=15, move_time=move_time)

    def swapaxis(s):
        mapping_x = {x1 : x0 for x0, x1 in zip('abcdefghi', '123456789')}
        mapping_y = {x0 : x1 for x0, x1 in zip('abcdefghi', '123456789')}
        return mapping_x[s[1]], mapping_y[s[0]]

    def remove_gdl(m):
        return m.replace("(place ", "").replace(")", "").strip().replace(' ', '')

    move = None
    sgf_moves = []
    while not gm.finished():
        move = gm.play_single_move(last_move=move)

        if move[0] == "noop":
            ri, str_move = 1, remove_gdl(move[1])
        else:
            ri, str_move = 0, remove_gdl(move[0])

        sgf_moves.append((ri, str_move))

        if str_move == "swap":
            sgf_moves[0][1] = swapaxis(sgf_moves[0][1])

    for ri, m in sgf_moves:
        print ri, m

    x = hashlib.md5(hashlib.sha1("%.5f" % time.time()).hexdigest()).hexdigest()[:6]
    with open("game_%s_%s_%s.sgf" % (gen0, gen1, x), "w") as f:
        f.write("(;FF[4]EV[null]PB[%s]PW[%s]SZ[11]GC[game#%s];" % (gen1, gen0, x))
        for ri, m in sgf_moves:
            f.write("%s[%s];" % ("W" if ri == 0 else "B", m))
        f.write(")\n")


if __name__ == "__main__":

    try:
        setup()
        gens = "x1_19", "x1_23"
        game_time = 5.0
        number_of_games = 1

        for i in range(4):
            play_game(gens[0], gens[1], game_time)
            #gens = gens[1], gens[0]

    except Exception as exc:
        print exc
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

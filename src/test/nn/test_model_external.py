'''
testing with games not defined in GDL
'''

import os
import random

import numpy as np

import tensorflow as tf

from ggplib.util.init import setup_once
from ggplib.db import lookup

from ggpzero.util import keras
from ggpzero.defs import templates

from ggpzero.nn.manager import get_manager

def setup():
    # set up ggplib
    setup_once()

    # ensure we have database with ggplib
    lookup.get_database()

    # initialise keras/tf
    keras.init()

    # just ensures we have the manager ready
    get_manager()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    np.set_printoptions(threshold=100000)

    from gzero_games.ggphack import addgame
    addgame.install_games()


games = ["draughts_bt_8x8"]


def advance_state(sm, basestate):
    sm.update_bases(basestate)

    # leaks, but who cares, it is a test
    joint_move = sm.get_joint_move()
    base_state = sm.new_base_state()

    for role_index in range(len(sm.get_roles())):
        ls = sm.get_legal_state(role_index)
        choice = ls.get_legal(random.randrange(0, ls.get_count()))
        joint_move.set(role_index, choice)

    # play move, the base_state will be new state
    sm.next_state(joint_move, base_state)
    return base_state


def test_basic_config():
    man = get_manager()

    for game in games:
        # look game from database
        game_info = lookup.by_name(game)
        assert game == game_info.game

        sm = game_info.get_sm()
        basestate = sm.get_initial_state()

        # lookup game in manager
        transformer = man.get_transformer(game)
        print transformer.x_cords

        print "rows x cols", transformer.num_rows, transformer.num_cols

        print
        print transformer.state_to_channels(basestate.to_list())
        basestate = advance_state(game_info.get_sm(), basestate)
        print
        print transformer.state_to_channels(basestate.to_list())



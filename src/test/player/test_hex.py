import os
from ggplib.player import get
from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup

from ggpzero.util import attrutil
from ggpzero.defs import confs, templates
from ggpzero.nn.manager import get_manager
from ggpzero.player.puctplayer import PUCTPlayer
from ggpzero.battle.hex2 import MatchInfo  # Import MatchInfo instead of print_board

BOARD_SIZE = 11
GAME = "hex_lg_11"
MODEL_WHITE = "b1_173"  # Model for white player
MODEL_BLACK = "h1_141"  # Model for black player

def setup():
    """Initialize the environment and prepare the neural network manager."""
    import tensorflow as tf

    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    import numpy as np
    np.set_printoptions(threshold=100000)

    man = get_manager()
    if not man.can_load(GAME, MODEL_WHITE):
        network = man.create_new_network(GAME)
        man.save_network(network, MODEL_WHITE)
    if not man.can_load(GAME, MODEL_BLACK):
        network = man.create_new_network(GAME)
        man.save_network(network, MODEL_BLACK)

def play(player_white, player_black, move_time=2.5):
    """Play a game between two players and export the game data."""
    gm = GameMaster(lookup.by_name(GAME), verbose=False)
    gm.add_player(player_white, "white")
    gm.add_player(player_black, "black")

    # Instantiate MatchInfo with the board size
    match_info = MatchInfo(BOARD_SIZE)

    gm.start(meta_time=15, move_time=move_time)

    move = None

    while not gm.finished():
        print("Current board =================================================")
        match_info.print_board(gm.sm) 
        move = gm.play_single_move(last_move=move) 

        player = None
        movement = None

        if move is not None:
            if move[0] == 'noop':
                player = "white"
                movement = move[1]
            else:
                player = "black"
                movement = move[0]

        print("Player:  "+ (str)(player) +", Movement: " +(str)(movement))
        print(type(movement))
        print(dir(movement))

        print("END ===============================================================")

    gm.finalise_match(move)
    match_info.print_board(gm.sm) 


def play_b1_vs_h1():
    """Set up and play a game between b1_173 and h1_141."""
    # Define evaluation configuration for b1_173
    eval_config_white = templates.base_puct_config(verbose=False, max_dump_depth=1)
    puct_config_white = confs.PUCTPlayerConfig(
        "gzero",
        True,
        800,
        0,
        MODEL_WHITE,
        eval_config_white
    )

    # Define evaluation configuration for h1_141
    eval_config_black = templates.base_puct_config(verbose=False, max_dump_depth=1)
    puct_config_black = confs.PUCTPlayerConfig(
        "gzero",
        True,
        800,
        0,
        MODEL_BLACK,
        eval_config_black
    )

    attrutil.pprint(puct_config_white)  # Print white player's configuration
    attrutil.pprint(puct_config_black)  # Print black player's configuration

    # Create players
    player_white = PUCTPlayer(puct_config_white)  # b1_173
    player_black = PUCTPlayer(puct_config_black)  # h1_141

    # Start the game
    play(player_white, player_black)

if __name__ == "__main__":
    setup()
    play_b1_vs_h1()

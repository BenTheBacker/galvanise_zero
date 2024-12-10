import os
import json
from flask import Flask, request, jsonify
from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup
from ggplib.util import log
from ggpzero.battle.hex import MatchInfo
from ggpzero.defs import confs
from ggpzero.player.puctplayer import PUCTPlayer

from ggplib.util import log

import tensorflow as tf
graph = tf.get_default_graph()

# Constants
BOARD_SIZE = 11
GAME = "hexLG11"
MODEL = "b1_173"

app = Flask(__name__)

def setup():
    """Initialize the environment once."""
    # Since we're using Python 2.7, no f-strings or print_function by default.
    import tensorflow as tf
    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    import numpy as np
    np.set_printoptions(threshold=100000)


def CreateConfig(model, displayLog):
    eval_config = confs.PUCTEvaluatorConfig(
        verbose=displayLog,
        puct_constant=0.85,
        puct_constant_root=3.0,
        dirichlet_noise_pct=-1,
        fpu_prior_discount=0.15,
        fpu_prior_discount_root=0.1,
        choose="choose_temperature",
        temperature=1.0,
        depth_temperature_max=10.0,
        depth_temperature_start=0,
        depth_temperature_increment=0.5,
        depth_temperature_stop=1,
        random_scale=1.0,
        batch_size=1,
        max_dump_depth=1,
    )

    puct_config = confs.PUCTPlayerConfig(
        name="gzero",
        verbose=displayLog,
        playouts_per_iteration=200,
        generation=model,
        evaluator_config=eval_config
    )
    
    return puct_config

def GetModels(displayLog):
    puct_config_white = CreateConfig(MODEL, displayLog)
    puct_config_black = CreateConfig(MODEL, displayLog)

    player_white = PUCTPlayer(puct_config_white)
    player_black = PUCTPlayer(puct_config_black)
    return player_white, player_black

def ParseMoves(moveString):
    moves = []
    moveSegments = moveString.split('.')
    for segment in moveSegments:
        if segment:
            parts = segment.split(':')
            if len(parts) == 3:
                player, x, y = parts
                # Swap move
                if y == 'z':
                    newMove = ('noop', 'swap')
                else:
                    newMove = ('noop', '(place ' + y + ' ' + x + ')')
                moves.append(newMove)
    return moves


# Default configuration
DEFAULT_DISPLAY_BOARD = False
DEFAULT_DISPLAY_LOGS = False
DEFAULT_MOVE_TIME = 5.0

# Global game state
player1 = None
player2 = None
gameMaster = None
matchInfo = None

def initialize_game(displayBoard=DEFAULT_DISPLAY_BOARD, displayLogs=DEFAULT_DISPLAY_LOGS, moveTime=DEFAULT_MOVE_TIME):
    global player1, player2, gameMaster, matchInfo
    player1, player2 = GetModels(displayLogs)
    gameMaster = GameMaster(lookup.by_name(GAME), verbose=displayBoard)
    gameMaster.add_player(player1, "white")
    gameMaster.add_player(player2, "black")
    gameMaster.start(meta_time=15, move_time=moveTime)
    matchInfo = MatchInfo(BOARD_SIZE)


@app.route('/next_move', methods=['POST'])
def next_move():
    data = request.get_json()
    log.debug("Received data: %s" % data)

    if data is None or "moves" not in data:
        log.debug("No moves provided in request data.")
        return jsonify({"error": "No moves provided"}), 400

    move_string = data["moves"]
    displayBoard = data.get("displayBoard", DEFAULT_DISPLAY_BOARD)
    displayLogs = data.get("displayLogs", DEFAULT_DISPLAY_LOGS)
    moveTime = float(data.get("moveTime", DEFAULT_MOVE_TIME))

    log.debug("Parsed request: moves=%s, displayBoard=%s, displayLogs=%s, moveTime=%s" %
              (move_string, displayBoard, displayLogs, moveTime))

    moves = ParseMoves(move_string)
    log.debug("Parsed moves: %s" % (moves,))

    with graph.as_default():
        lastMove = None
        roles = gameMaster.sm.get_roles()
        log.debug("Current roles: %s" % (roles,))

        for i, move in enumerate(moves):
            current_role = roles[i % len(roles)]
            log.debug("Applying move %s for role %s" % (move, current_role))
            gameMaster.set_forced_move(current_role, move[1])

            # Log whether the game is terminal before the move
            log.debug("Before move: terminal=%s" % gameMaster.finished())

            lastMove = gameMaster.play_single_move(lastMove)
            # Log whether the game is terminal after the move
            log.debug("After move: terminal=%s" % gameMaster.finished())

            gameMaster.clear_forced_move(current_role)

        if displayBoard:
            log.debug("Current board after applying moves:")
            matchInfo.print_board(gameMaster.sm)  # This prints to stdout, not the log
            
        # Now attempt to get the next move
        log.debug("Attempting to get next move.")
        log.debug("Before next move: terminal=%s" % gameMaster.finished())
        nextMove = gameMaster.play_single_move(lastMove)
        log.debug("Next move obtained: %s" % (nextMove,))
        log.debug("After next move: terminal=%s" % gameMaster.finished())

        if displayBoard:
            log.debug("Current board after next move:")
            matchInfo.print_board(gameMaster.sm)

    return jsonify({"move": str(nextMove)})


@app.route('/reset', methods=['POST'])
def reset_game():
    # This endpoint resets the game state to the initial position.
    with graph.as_default():
        initialize_game(DEFAULT_DISPLAY_BOARD, DEFAULT_DISPLAY_LOGS, DEFAULT_MOVE_TIME)

    return jsonify({"status": "game reset"})


if __name__ == "__main__":
    setup()
    initialize_game()
    # Run Flask server
    app.run(host='0.0.0.0', port=5000)

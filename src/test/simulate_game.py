import os
import sys
from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup
from ggplib.player import get
from ggpzero.battle.hex2 import MatchInfo  # Import MatchInfo to print the board

from ggpzero.nn.manager import get_manager

BOARD_SIZE = 11
GAME = "hex_lg_11"

def setup():
    """Initialize the environment."""
    import tensorflow as tf

    from ggplib.util.init import setup_once
    setup_once()

    from ggpzero.util.keras import init
    init()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    import numpy as np
    np.set_printoptions(threshold=100000)

def parse_moves(move_string):
    moves = []

    # Split the string by periods
    move_segments = move_string.split(':')

    for segment in move_segments:
        if segment:  # Skip empty segments

            parts = segment.split('.')
            if len(parts) == 3:
                player, x, y = parts
                print(segment)
                print(player, x, y)

                #Is black player
                if player == "RED":
                    newMove = ('(place ' + y + ' ' + (str)(x) + ')', 'noop')
                else:
                    newMove = ('noop', '(place ' + y + ' ' + (str)(x) + ')')

                moves.append(newMove)

    return moves

def sort_moves(moves, blackFirst=True):
    if not moves:
        return []

    # Determine the starting position based on blackFirst
    if blackFirst:
        pattern = [1, 0]  # Black (noop at index 1 first), then alternate
    else:
        pattern = [0, 1]  # Red (noop at index 0 first), then alternate

    # Create a function to determine the sorting key
    def sorting_key(index):
        # Use the alternation pattern to determine expected noop index
        return pattern[index % 2]

    # Sort the moves array according to the fixed pattern
    sorted_moves = sorted(enumerate(moves), key=lambda x: sorting_key(x[0]))
    return [move for _, move in sorted_moves]


def reconstruct_game(player_white, player_black, moves, board_size=BOARD_SIZE):
    # Initialize GameMaster
    gm = GameMaster(lookup.by_name(GAME), verbose=False)
    gm.add_player(player_white, "white")
    gm.add_player(player_black, "black")

    # Create MatchInfo to track and print the board
    match_info = MatchInfo(board_size)

    # Start the game
    gm.start(meta_time=15, move_time=0.5)

    # Replay the moves
    lastMove = None
    for move in moves:
        lastMove = gm.play_forced_move(move, lastMove)

    # Print the final reconstructed board
    match_info.print_board(gm.sm)

if __name__ == "__main__":
    # Ensure setup is called
    setup()

    # Parse the moves from the system argument
    if len(sys.argv) < 2:
        print("Usage: python reconstruct_game.py '<moves>'")
        print("Example: python reconstruct_game.py 'RED.1.a:BLUE.2.b:RED.3.c:...'")
        sys.exit(1)

    move_string = sys.argv[1]

    # Parse the move string into a list of moves
    print("Move string: ", move_string) 
    moves = parse_moves(move_string)
    moves = sort_moves(moves)

    print("Moves: ", moves)

    # Reconstruct and display the game state
    reconstruct_game(get.get_player("simplemcts"), get.get_player("simplemcts"), moves)

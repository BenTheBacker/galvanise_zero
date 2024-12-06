import os
import sys
from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup
from ggpzero.battle.hex2 import MatchInfo  # Import MatchInfo to print the board

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
                moves.append({"player": player, "x": x, "y": int(y)})

    return moves

def reconstruct_game(moves, board_size=BOARD_SIZE):
    """
    Reconstruct the game state from a list of moves and print the board.

    Args:
        moves (list of dict): List of moves, e.g., [{"player": "RED", "x": "a", "y": 1}, ...].
        board_size (int): The size of the Hex board.

    Returns:
        None
    """
    # Initialize GameMaster
    gm = GameMaster(lookup.by_name(GAME), verbose=False)

    # Create MatchInfo to track and print the board
    match_info = MatchInfo(board_size)

    # Start the game
    gm.start(meta_time=15, move_time=0.5)

    # Replay the moves
    for move in moves:
        role = "white" if move["player"] == "RED" else "black"
        move_str = f"{move['x']}{move['y']}"  # Format move as "a1", "b2", etc.
        gm.play_forced_move(move_str=move_str, role=role)

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
    moves = parse_moves(move_string)

    # Reconstruct and display the game state
    reconstruct_game(moves)

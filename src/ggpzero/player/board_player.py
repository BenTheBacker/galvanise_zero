import sys

# Decoding Functions
import os
import sys
from ggplib.player.gamemaster import GameMaster
from ggplib.db import lookup
from ggplib.player import get
from ggplib.util import log
from ggpzero.battle.hex import MatchInfo  # Import MatchInfo to print the board
from ggpzero.defs import confs, templates
from ggpzero.nn.manager import get_manager
from ggpzero.player.puctplayer import PUCTPlayer

BOARD_SIZE = 11
GAME = "hexLG11"
MODEL = "b1_173"


def TranslateByteToMove(byte):
    isVertical = bool((byte >> 7) & 0x01)
    num = byte & 0x7F  # 0x7F = 01111111
    
    if num == 0:
        # Special move
        x = 99
        y = 'z'

        return  ('noop', 'swap')
    else:
        x = (num // 11) + 1
        y_index = num % 11
        if y_index == 0:
            y_index = 11  # Adjust for zero-based indexing
        y = chr(ord('a') + y_index - 1)  # Convert back to character

        return ('noop', '(place ' + y + ' ' + x + ')')

def DecodeBoard(boardBytes):
    moves = []
    for byte in boardBytes:
        byte_val = ord(byte)  # Convert single character to integer
        move = TranslateByteToMove(byte_val)
        moves.append(move)
    return moves

def LoadBoardsFromFile(filename, movesPlayed):
    boards = []
    with open(filename, 'rb') as file:
        while True:
            board_bytes = file.read(movesPlayed)
            if not board_bytes:
                break
            if len(board_bytes) != movesPlayed:
                raise ValueError("Incomplete board data found in the file.")
            board = board_bytes  # In Python 2, this is a string
            decoded_board = DecodeBoard(board)
            boards.append(decoded_board)
    
    print "Successfully loaded {} boards from '{}'.".format(len(boards), filename)
    return boards

def GetNextMove(player_white, player_black, moves, moveTime = 5, board_size=BOARD_SIZE, displayBoard = False):
    # Initialize GameMaster
    gameMaster = GameMaster(lookup.by_name(GAME), verbose=displayBoard)
    gameMaster.add_player(player_white, "white")
    gameMaster.add_player(player_black, "black")

    # Create MatchInfo to track and print the board
    matchInfo = MatchInfo(board_size)

    # Start the game
    gameMaster.start(meta_time=15, move_time=moveTime)

    # Retrieve the roles in the order defined by the state machine
    roles = gameMaster.sm.get_roles()
    if len(roles) != 2:
        raise ValueError("This function is designed for two-player games.")

    # Determine the order of roles (assuming roles[0] is 'white' and roles[1] is 'black')
    role_order = roles  # Typically ['white', 'black']

    # Replay the moves
    lastMove = None
    for i, move in enumerate(moves):
        # Determine which role's turn it is
        current_role = role_order[i % len(role_order)]

        # Set the forced move for the current role
        gameMaster.set_forced_move(current_role, move[1])

        if gameMaster.verbose:
            log.info("Reconstructing move " + (str)(i + 1) + ": Role " + current_role + "plays " + move[1])

        # Execute the move by playing a single move
        lastMove = gameMaster.play_single_move(lastMove)

        # Clear the forced move after it's been used
        gameMaster.clear_forced_move(current_role)

    # Print the final reconstructed board
    if displayBoard:
        matchInfo.print_board(gameMaster.sm)
    
    #Predict the final move:
    lastMove = gameMaster.play_single_move(lastMove)
    if displayBoard:
        matchInfo.print_board(gameMaster.sm)

    return lastMove

def CreateConfig(model, displayLog):
    """Creates and returns a hardcoded PUCT configuration."""
    
    # Hardcoded PUCTEvaluatorConfig
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
    
    # Hardcoded PUCTPlayerConfig
    puct_config = confs.PUCTPlayerConfig(
        name="gzero",
        verbose=displayLog,
        playouts_per_iteration=200,  
        generation=model,
        evaluator_config=eval_config
    )
    
    return puct_config

def GetModels(displayLog):
    puct_config_white =  CreateConfig(MODEL, displayLog)
    puct_config_black = CreateConfig(MODEL, displayLog)

    # Create players
    player_white = PUCTPlayer(puct_config_white)
    player_black = PUCTPlayer(puct_config_black)  

    return player_white, player_black

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


if __name__ == "__main__":
    # Ensure setup is called
    setup()

    # Parse the move string into a list of moves
    #print("Move string: ", move_string) 
    moves = LoadBoardsFromFile("data//boardsTurn2.bin", 2)[0]

    #print("Moves: ", moves)

    # Reconstruct and display the game state
    player1, player2 = GetModels(False)
    move = GetNextMove(player1, player2, moves, 10, displayBoard=False)
        
    print("Next Move:", move)
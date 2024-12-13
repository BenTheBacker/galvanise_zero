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
    # Extract the isVertical flag (bit 7)
    isVertical = bool((byte >> 7) & 0x01)
    
    # Extract the num value (bits 0-6)
    num = byte & 0x7F  # 0x7F is 01111111 in binary
    
    if num == 0:
        # Special case where x is 99
        x = 99
        y = 'z'  # You can choose a default or placeholder
    else:
        # Reverse the calculation num = (x - 1) * 11 + y
        x = (num - 1) // 11 + 1
        y_num = (num - 1) % 11 + 1
        
        # Convert y_num back to a character
        y = chr(ord('a') + y_num - 1)
    
    move = (x, y)
    
    return isVertical, move

def DecodeBoard(boardBytes):
    moves = []
    for byte in boardBytes:
        byte_val = ord(byte)  # Convert single character to integer
        _, move = TranslateByteToMove(byte_val)
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
        if move[0] == 99:
            move = ('noop', 'swap')
        else:
            move = ('noop', '(place ' + (str)(move[1]) + ' ' + (str)(move[0]) + ')')

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

from __future__ import print_function  # Allows using print() as a function in Python 2.7
import sys

if __name__ == "__main__":
    inputFile = "data//boardsTurn1.bin"
    outputFile = "data//boardsTurn1Solved.bin"

    turns = 1

    # Ensure setup is called
    setup()

    # Load boards from the input file
    boards = LoadBoardsFromFile(inputFile, 1)
    total_boards = len(boards)

    # Read existing movesStr from the output file into a set
    existing_moves = set()
    try:
        with open(outputFile, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    moves_part = line.split(':')[0]
                    existing_moves.add(moves_part)
    except IOError:
        # If the output file doesn't exist, proceed without existing moves
        pass

    with open(outputFile, 'a') as output_file:
        processed_count = 0  # Counter for processed (written) boards
        for index, moves in enumerate(boards):
            index += 1  # Manually increment index to start from 1
            
            # Convert the current moves list to a string
            movesStr = ', '.join(['({}, {})'.format(m[0], m[1]) for m in moves])
            
            # Check if this movesStr is already in the output file
            if movesStr in existing_moves:
                print("Skipping {}/{}: {} already exists.".format(index, total_boards, movesStr))
                continue  # Skip to the next board
            
            # Get the models and determine the next move
            player1, player2 = GetModels(False)
            move = GetNextMove(player1, player2, moves, 10, displayBoard=True)
                
            # Prepare the output string
            outputStr = "{0}:{1}\n".format(movesStr, move)
            
            # Write the new move to the output file
            output_file.write(outputStr)
            
            # Add the new movesStr to the set to avoid future duplicates
            existing_moves.add(movesStr)
            
            # Increment the processed counter
            processed_count += 1
            
            # Print the progress
            print("{0}/{1} {2}".format(processed_count, total_boards, outputStr))
    

    
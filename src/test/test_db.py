from ggplib.db import lookup

import sys
import traceback


def main():
    # Retrieve all game names from the database
    game_names = lookup.get_all_game_names()
    
    # Check if there are any games
    if not game_names:
        print ("No games found in the database.")
        return
    
    # Print each game name
    print ("Available Games:")
    for name in sorted(game_names):
        print ("- {}".format(name))


if __name__ == "__main__":
    main()

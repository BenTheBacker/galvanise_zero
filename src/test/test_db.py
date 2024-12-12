from ggplib.db import lookup

import sys
import traceback

import pickle

def main():
    lookup.the_database = None
    lookup.get_database()

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

    game = lookup.by_name("hexLG11")
    print(game)


if __name__ == "__main__":
    main()

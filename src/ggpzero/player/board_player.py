import sys

# Decoding Functions

def TranslateByteToMove(byte):
    isVertical = bool((byte >> 7) & 0x01)
    num = byte & 0x7F  # 0x7F = 01111111
    
    if num == 0:
        # Special move
        x = 99
        y = 'z'
    else:
        x = (num // 11) + 1
        y_index = num % 11
        if y_index == 0:
            y_index = 11  # Adjust for zero-based indexing
        y = chr(ord('a') + y_index - 1)  # Convert back to character
    return (x, y)

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

if __name__ == "__main__":
    FILENAME = "boardsTurn2.bin"
    boards = LoadBoardsFromFile(FILENAME, 2)

    for i, board in enumerate(boards):
        print "Board {}: {}".format(i + 1, board)

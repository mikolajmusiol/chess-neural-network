import pygame as py
import chess
import chess.svg
import numpy as np
import tensorflow as tf

py.init()

WIDTH, HEIGHT = 512, 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)

PIECES = ['bP', 'bR', 'bN', 'bB', 'bQ', 'bK', 'wP', 'wR', 'wN', 'wB', 'wQ', 'wK']
IMAGES = {}


def predict_from_board(model, board, num_best=-1):
    one_hot_mapping = {".": np.array([0, 0, 0, 0, 0, 0]),
                       "P": np.array([1, 0, 0, 0, 0, 0]),
                       "N": np.array([0, 1, 0, 0, 0, 0]),
                       "B": np.array([0, 0, 1, 0, 0, 0]),
                       "R": np.array([0, 0, 0, 1, 0, 0]),
                       "Q": np.array([0, 0, 0, 0, 1, 0]),
                       "K": np.array([0, 0, 0, 0, 0, 1]),
                       "p": np.array([-1, 0, 0, 0, 0, 0]),
                       "n": np.array([0, -1, 0, 0, 0, 0]),
                       "b": np.array([0, 0, -1, 0, 0, 0]),
                       "r": np.array([0, 0, 0, -1, 0, 0]),
                       "q": np.array([0, 0, 0, 0, -1, 0]),
                       "k": np.array([0, 0, 0, 0, 0, -1])}

    board_data = np.array([board]).astype(str)
    data_split = np.array([row.split() for row in board_data[0].split("\n")])
    data_encoded_state = np.zeros([8, 8, 6])
    for char, encoding in one_hot_mapping.items():
        data_encoded_state[data_split == char] = encoding
    tensor_data = np.expand_dims(np.array(data_encoded_state), 0)
    tensor = tf.constant(tensor_data)
    test_pred = model.predict(tensor)
    move = chess.Move.null()
    move_made = False
    while not move_made:
        move_from_pred = np.argsort(np.max(test_pred[0], axis=0))[num_best]
        for possible_to_move in np.argsort(np.max(test_pred[1], axis=0))[::-1]:
            if possible_to_move != move_from_pred:
                move_to_check = chess.Move.from_uci(str(chess.square_name(move_from_pred)) + str(chess.square_name(possible_to_move)))
                if move_to_check in board.legal_moves and move not in board.move_stack:
                    move = move_to_check
                    move_made = True
                    break
        num_best-=1
    return move


def load_images():
    for piece in PIECES:
        IMAGES[piece] = py.transform.scale(py.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE))


def draw_board(screen, highlight_squares):
    colors = [LIGHT_BROWN, DARK_BROWN]
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = colors[(row + col) % 2]
            if chess.square(col, row) in highlight_squares:
                color = py.Color(0, 255, 0)
            py.draw.rect(screen, color, py.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def draw_pieces(screen, board):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board.piece_at(chess.square(col, row))
            if piece:
                piece_color = 'w' if piece.color == chess.WHITE else 'b'
                piece_type = piece.symbol().upper()
                piece_image_key = piece_color + piece_type
                piece_image = IMAGES[piece_image_key]
                screen.blit(piece_image, py.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def main():
    screen = py.display.set_mode((WIDTH, HEIGHT))
    clock = py.time.Clock()
    screen.fill(WHITE)

    model_name = 'model_1'
    model = tf.keras.models.load_model(f'./Models/{model_name}.keras')

    board = chess.Board()

    load_images()

    squares_to_highlight = []
    running = True
    human_turn = False
    square_selected = ()
    player_clicks = []
    game_over = False

    while running:
        for event in py.event.get():
            if event.type == py.QUIT or board.is_checkmate():
                running = False
            elif event.type == py.MOUSEBUTTONDOWN:
                if not game_over and human_turn:
                    location = py.mouse.get_pos()
                    column = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    if square_selected == (row, column) or column >= DIMENSION:
                        square_selected = ()
                        player_clicks = []
                        squares_to_highlight = []
                    else:
                        square_selected = (row, column)
                        player_clicks.append(square_selected)
                        squares_to_highlight.append(square_selected)
                        legal_moves = [move.to_square for move in board.legal_moves if move.from_square == chess.square(column, row)]
                        squares_to_highlight.extend(legal_moves)
                    if len(player_clicks) == 2:
                        move_from = chess.square_name(chess.square(player_clicks[0][1], player_clicks[0][0]))
                        move_to = chess.square_name(chess.square(player_clicks[1][1], player_clicks[1][0]))
                        move = chess.Move.from_uci(str(move_from) + str(move_to))
                        if move in board.legal_moves:
                            board.push(move)
                            square_selected = ()
                            player_clicks = []
                            human_turn = not human_turn
                        else:
                            player_clicks = [square_selected]
                        squares_to_highlight = []
            elif not game_over and not human_turn:
                move = predict_from_board(model, board, -1)
                py.time.delay(1000)
                board.push(move)
                human_turn = not human_turn

            draw_board(screen, squares_to_highlight)
            draw_pieces(screen, board)

            clock.tick(60)
            py.display.flip()

    py.quit()


if __name__ == "__main__":
    main()

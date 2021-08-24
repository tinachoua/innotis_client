import os

def print_title(text):
    # get terminal size
    columns, rows = os.get_terminal_size(0)
    print('-'*columns)
    print(text, '\n')
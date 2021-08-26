import os

def print_title(text):
    # get terminal size
    columns, rows = os.get_terminal_size()
    print('-'*columns)
    print(text, '\n')
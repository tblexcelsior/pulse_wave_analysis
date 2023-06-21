import os
import pathlib
import numpy as np
import tensorflow as tf

def get_single_col_by_input_type(input_type, column_definition):
    l = [tup[0] for tup in column_definition if tup[1] == input_type]
    if len(l) != 1:
        raise ValueError("Invalid number of columns for {}".format(input_type))

    return l[0]

def create_folder_if_not_exist(directory):
  """Creates folder if it doesn't exist.

  Args:
    directory: Folder path to create.
  """
  # Also creates directories recursively
  pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
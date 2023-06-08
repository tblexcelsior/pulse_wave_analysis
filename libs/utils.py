import os
import numpy as np

def get_single_col_by_input_type(input_type, column_definition):
    l = [tup[0] for tup in column_definition if tup[1] == input_type]
    if len(l) != 1:
        raise ValueError("Invalid number of columns for {}".format(input_type))

    return l[0]
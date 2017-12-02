# -*- coding: ascii -*-

import numpy as np


def scale(value, start_min, start_max, end_min, end_max):
    """Returns the result of scaling value from the range
    [start_min, start_max] to [end_min, end_max].
    """
    return (end_min + (end_max - end_min) * (value - start_min) /
            (start_max - start_min))

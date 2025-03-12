import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from pepNmemb.scripts.get_tilt_angle import find_consecutive_sublists, get_vector, vector_length, dot, angle_between_vectors, get_coords_spanning_memb, relative_tilt
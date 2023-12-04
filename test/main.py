import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'files')
sys.path.append(module_path)

from data_arrange import ArrangeData
from interpolation import ClimbInterp
import random

if __name__ == "__main__":
    random.seed(69)
    random_numbers_x = random.sample(range(1, 51), 50)
    random_numbers_y = random.sample(range(1, 51), 50)
    arrange_data = ArrangeData(random_numbers_x, random_numbers_y)
    random_numbers_x, random_numbers_y = arrange_data.get_arranged_points()
    climb_interp = ClimbInterp(random_numbers_x, random_numbers_y, show_graph=True)
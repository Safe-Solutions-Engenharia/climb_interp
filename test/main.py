# Linking the folder to the path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'climbinterp')
sys.path.append(module_path)

# Importing our modules
from data_arrange import ArrangeData
from interpolation import ClimbInterp
import random

if __name__ == "__main__":
    # Setting the seed for the random numbers
    SEED = 55

    random.seed(SEED)
    random_numbers_x = random.sample(range(1, 51), 50)
    random_numbers_y = random.sample(range(1, 51), 50)
    arrange_data = ArrangeData(random_numbers_x, random_numbers_y)
    random_numbers_x, random_numbers_y = arrange_data.get_arranged_points()
    climb_interp = ClimbInterp(random_numbers_x, random_numbers_y, show_graph=True)
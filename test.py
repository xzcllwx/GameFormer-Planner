import os
full_path = os.path.realpath(__file__)
import sys
to_add = os.path.abspath('/root/xzcllwx_ws/toy-example-of-iLQR/build/')
print("to_add: ", to_add)
sys.path.append(to_add)

import motion_planning
import numpy as np
print("motion_planning: init")
A = motion_planning.motion_planner('/root/xzcllwx_ws/toy-example-of-iLQR/config/scenario_two_borrow.yaml')
# A.demo('/root/xzcllwx_ws/toy-example-of-iLQR/config/scenario_two_borrow.yaml')

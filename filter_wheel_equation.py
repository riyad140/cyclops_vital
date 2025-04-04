# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 14:28:36 2025

@author: imrul
"""

filter_wheel_home_offset = 28
full_step_size_distance = 0.0075

filter_index_list = [0,1,2,3,4]

for filter_index in filter_index_list:

    filter_wheel_angle = filter_wheel_home_offset*(full_step_size_distance/0.6)+(360*(full_step_size_distance/0.6)/5)*filter_index
    
    
    print(f'Angle for filter wheel index {filter_index} is {filter_wheel_angle}')
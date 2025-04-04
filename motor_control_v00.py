# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 18:22:19 2021

@author: imrul
"""

#Run the program
from vital.phoenix import *
reads = []
temps = []
async with PhoenixClient("testbench-6.vital.company", on_alert=eng_on_alert_handler(reads, temps)) as client:
    
    
    programA = [
        #home_spindle(duration=0),
        position_spindle(duration=5, position=900, velocity_limit=20, acceleration=10, deceleration=10),
        set_spindle_velocity(duration=10, velocity=50, acceleration=25, deceleration=25),
        #power_off_spindle(duration=0)
    ]
    await client.run_custom_program(program=programA, title="motor_control_test")
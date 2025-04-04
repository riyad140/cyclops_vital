# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:31:12 2022

To create the angular postions map for different Assays for IM4 Discs

@author: imrul
"""

import numpy as np
import math

#%% PBS

theta_ref= 0 #159.85 # angular distance between notch and first reference of the assay
theta_total_0=8.46 # angular length of the frst structure of the assay


theta_safety=1 # overall safety margin from the structures
theta_eff=theta_total_0-theta_safety
nFOV=5 # Number 


theta_fov=theta_eff/nFOV  # angular displacement from fov to fov

home_deg_0=theta_ref+theta_safety/2

Angular_positions_0=[
                   home_deg_0+theta_fov/2+theta_fov*0,
                   home_deg_0+theta_fov/2+theta_fov*1,
                   home_deg_0+theta_fov/2+theta_fov*2,
                   home_deg_0+theta_fov/2+theta_fov*3,
                   home_deg_0+theta_fov/2+theta_fov*4     
                   ]

print(Angular_positions_0)

theta_structure_0=0.84
theta_jump_0=theta_structure_0+theta_total_0-Angular_positions_0[-1]

theta_total_1=8.91
theta_eff=theta_total_1-theta_safety
theta_fov=theta_eff/nFOV

home_deg_1=theta_jump_0+theta_safety/2+Angular_positions_0[-1]

Angular_positions_1=[
                   home_deg_1+theta_fov/2+theta_fov*0,
                   home_deg_1+theta_fov/2+theta_fov*1,
                   home_deg_1+theta_fov/2+theta_fov*2,
                   home_deg_1+theta_fov/2+theta_fov*3,
                   home_deg_1+theta_fov/2+theta_fov*4     
                   ]

print(Angular_positions_1)

# Sample

theta_structure_1=12.37
theta_jump_1=theta_total_0+theta_structure_0+theta_total_1+theta_structure_1-Angular_positions_1[-1]
theta_total_2=8.39
theta_eff=theta_total_2-theta_safety
theta_fov=theta_eff/nFOV

home_deg_2=theta_jump_1+theta_safety/2+Angular_positions_1[-1]

Angular_positions_2=[
                   home_deg_2+theta_fov/2+theta_fov*0,
                   home_deg_2+theta_fov/2+theta_fov*1,
                   home_deg_2+theta_fov/2+theta_fov*2,
                   home_deg_2+theta_fov/2+theta_fov*3,
                   home_deg_2+theta_fov/2+theta_fov*4     
                   ]

print(Angular_positions_2)

theta_structure_2= 1.02
theta_jump_2=theta_total_0+theta_structure_0+theta_total_1+theta_structure_1+theta_total_2+theta_structure_2-Angular_positions_2[-1]
theta_total_3=6.59
theta_eff=theta_total_3-theta_safety
theta_fov=theta_eff/nFOV

home_deg_3=theta_jump_2+theta_safety/2+Angular_positions_2[-1]

Angular_positions_3=[
                   home_deg_3+theta_fov/2+theta_fov*0,
                   home_deg_3+theta_fov/2+theta_fov*1,
                   home_deg_3+theta_fov/2+theta_fov*2,
                   home_deg_3+theta_fov/2+theta_fov*3,
                   home_deg_3+theta_fov/2+theta_fov*4     
                   ]

print(Angular_positions_3)

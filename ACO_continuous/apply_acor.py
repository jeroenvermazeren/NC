# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 18:16:01 2018
@author: Victor Costa
Simple example of how to apply the ACOr class.
"""

import ant_colony_for_continuous_domains
import continuous_benchmarks
from ctf.functions2d import eggholder, three_hump_camel
from scipy.optimize import rosen

colony = ant_colony_for_continuous_domains.ACOr()
ranges = [[-512,512],
          [-512,512]]

functions = [eggholder.Eggholder().cost, three_hump_camel.ThreeHumpCamel().cost, rosen]

for function in functions:

	colony.set_cost(function)
	colony.set_variables(2, ranges)
	colony.set_parameters(500, 50, 50, 0.0001, 0.5)
	# colony.set_parameters(100, 5, 50, 0.01, 0.85)

	solution = colony.optimize()

	print(solution[0:2])

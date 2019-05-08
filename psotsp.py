import psopy
import tsplib95

problem = tsplib95.load_problem('dsj1000.tsp')
print(list(problem.get_nodes()))
import math
import random
from PSO_system.Counting.particle import Particle
import numpy as np
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, pyqtSlot, QMutex
import shapely.geometry as sp
import descartes
from copy import deepcopy
import ACO_system.ant_colony_for_continuous_domains

class RunSignals(QObject):
    iteration = pyqtSignal(object)
    result = pyqtSignal(object)


class CarRunning(QRunnable):
    """ work thread, will execute "run" first """

    def __init__(self, data, filename, traindata, traindataname, para_list,save_file_index):
        super(CarRunning, self).__init__()
        # for return result
        self.signals = RunSignals()
        # read data file
        self._mutex = QMutex()
        self.data = data[filename]
        self.train_data = traindata[traindataname]
        # adjust parameters
        self.iteration_times = para_list[0]
        self.ant_colony_size = para_list[1]
        self.archive_size = int(para_list[2])
        # print("archive_size =  ",self.archive_size)
        self.locality = para_list[3]
        self.convergence_speed = para_list[4]
        self.neurl_num = para_list[5]
        self.v_max = para_list[6]
        self.sd_max = para_list[7]
        # self.file_name = para_list[-1]
        self.save_file = para_list[-1] + str(save_file_index)

        if traindataname.find('4') != -1:
            self.dim_i = 3
        elif traindataname.find('6') != -1:
            self.dim_i = 5
        else:
            self.dim_i = 3
        max_px = max_py = -math.inf
        min_px = min_py = math.inf
        for i, j in zip(self.data.x[2:], self.data.y[2:]):
            max_px = max(i, max_px)
            min_px = min(i, min_px)
            max_py = max(j, max_py)
            min_py = min(j, min_py)

        self.upbound_of_map = ((max_px - min_px)**2 +
                               (max_py - min_py)**2)**(0.5)
        # range for means
        maxp = -math.inf
        minp = math.inf
        for i in self.train_data.v_x:
            for j in range(self.dim_i):
                maxp = max(maxp, i[j])
                minp = min(minp, i[j])
        self.range = (maxp, minp)
        # initialize the particles
        self.particles = []
        self.swarm_size = 1 # using original PSO particle as datastructure for running ACO
        for _ in range(self.swarm_size):
            self.particles.append(Particle(self.neurl_num, self.dim_i, self.range, self.sd_max, self.v_max))


    # def run(self):
    #     """
    #     [the main function for caculation the GA]
    #     """
    #     for times in range(self.iteration_times):
    #         average_error = 0
    #         worst_error = math.inf
    #         better_particle_in_pocket = False
    #         #evaluate the fitness value for each particle
    #         for idx in range(self.swarm_size):
    #             self.particles[idx].fitness = self.adaptation_funct_pso(idx)
    #             average_error += self.particles[idx].fitness
    #             if self.particles[idx].fitness < worst_error:
    #                 worst_error = self.particles[idx].fitness
    #             if times == 0:
    #                 self.particles[idx].p_fitness = deepcopy(self.particles[idx].fitness)
    #             else:
    #                 if self.particles[idx].p_fitness < self.particles[idx].fitness:
    #                     self.particles[idx].update_p()
    #         average_error = average_error/self.swarm_size
    #         #arbitary
    #         if times == 0:
    #             self.pocket_particle = deepcopy(self.particles[0])
    #         self.bestneighbor = deepcopy(self.particles[0])
    #         # find the best neighbor
    #         for idx in range(self.swarm_size):
    #             if self.bestneighbor.fitness < self.particles[idx].fitness:
    #                 self.bestneighbor = deepcopy(self.particles[idx])
    #         if self.pocket_particle.fitness < self.bestneighbor.fitness:
    #             better_particle_in_pocket = True
    #             self.pocket_particle = deepcopy(self.bestneighbor)
    #         # update the v & x
    #         for p in self.particles:
    #             p.v_theta = (self.self_weight * p.v_theta) + self.exp_weight*random.uniform(0, 1)*(p.p_theta - p.theta) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.theta - p.theta)
    #             p.v_weight = (self.self_weight * p.v_weight) + self.exp_weight*random.uniform(0, 1)*(p.p_weight - p.weight) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.weight - p.weight)
    #             p.v_means = (self.self_weight * p.v_means) + self.exp_weight*random.uniform(0, 1)*(p.p_means - p.means) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.means - p.means)
    #             p.v_sd = (self.self_weight * p.v_sd) + self.exp_weight*random.uniform(0, 1)*(p.p_sd - p.sd) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.sd - p.sd)
    #             p.limit_v()
    #             p.update_location()
    #             p.limit_location_upbound()
    #         self.signals.iteration.emit("------------------------------------------------------")
    #         self.signals.iteration.emit("Iteration Times: {}".format(times+1))
    #         self.signals.iteration.emit("Average Error: {} \n  (Normalize: {})".format(1/average_error, 1/average_error/40))
    #         self.signals.iteration.emit("Worst Error: {} \n  (Normalize: {})".format(1/worst_error, 1/worst_error/40))
    #         self.signals.iteration.emit("Least Error: {} \n  (Normalize: {})".format(1/self.pocket_particle.fitness, 1/self.pocket_particle.fitness/40))
    #         if better_particle_in_pocket is True:
    #             self.signals.iteration.emit("Detail parameter:")
    #             self.signals.iteration.emit("Theta: {}".format(self.pocket_particle.theta))
    #             self.signals.iteration.emit("Means: {}".format(self.pocket_particle.means))
    #             self.signals.iteration.emit("Weight: {}".format(self.pocket_particle.weight))
    #             self.signals.iteration.emit("SD: {}".format(self.pocket_particle.sd))
    #     self.signals.iteration.emit("-------------PSO training finished---------------")
    #     self.signals.result.emit([self.pocket_particle, self.neurl_num, self.dim_i])

# DIT IS RUN_ACO BITCHES
    @pyqtSlot()
    def run(self):
        """
        [the main function for caculation the GA]
        """
        # for times in range(self.iteration_times):
        original_particle = deepcopy(self.particles[0])
        for times in range(1):
            average_error = 0
            worst_error = math.inf
            p = deepcopy(original_particle)
            colony = ACO_system.ant_colony_for_continuous_domains.ACOr(p)
            # ranges = []
            # form is [theta, v_max_1, ..., v_max_neurl_num, sd_max_1, ..., sd_max_neurl_num, weight_1, ..., weight_neurl_num]
            # ranges.append([-1.0, 1.0])
            # print(ranges)
            # for n in range(self.means):
            #     ranges.append([0.0, self.v_max])
            # for n in range(self.sd):
            #     ranges.append([0.0, self.sd_max])
            # for n in range(self.neurl_num):
            #     ranges.append([-1.0, 1.0])
            self._mutex.lock()
            colony.set_cost(self.adaptation_funct_aco)
            self._mutex.unlock()
            self._mutex.lock()
            print('set parameters')

            # max_iter = 20  # Maximum number of iterations
            # pop_size = 5  # Population size
            # k = 50  # Archive size
            # q = 0.1  # Locality of search
            # xi = 0.85  # Speed of convergence
            colony.set_parameters(self.iteration_times, self.ant_colony_size, self.archive_size, self.locality, self.convergence_speed)
            self._mutex.unlock()
            # colony.set_parameters(100, 5, 50, 0.01, 0.85)

            self._mutex.lock()
            solution, self.particles[0] = colony.optimize(self.signals,self.neurl_num,self.save_file)
            self._mutex.unlock()
            print('best found solution:')
            print(solution[0:10])
            print()
            self.pocket_particle = deepcopy(self.particles[0])
            self.pocket_particle.solution = solution
            # #evaluate the fitness value for each particle
            # for idx in range(self.swarm_size):
            #     self.particles[idx].fitness = self.adaptation_funct_pso(idx)
            #     average_error += self.particles[idx].fitness
            #     if self.particles[idx].fitness < worst_error:
            #         worst_error = self.particles[idx].fitness
            #     if times == 0:
            #         self.particles[idx].p_fitness = deepcopy(self.particles[idx].fitness)
            #     else:
            #         if self.particles[idx].p_fitness < self.particles[idx].fitness:
            #             self.particles[idx].update_p()
            # average_error = average_error/self.swarm_size
            #
            # #arbitary
            # if times == 0:
            #     self.pocket_particle = deepcopy(self.particles[0])
            # self.bestneighbor = deepcopy(self.particles[0])
            #
            # # find the best neighbor
            # for idx in range(self.swarm_size):
            #     if self.bestneighbor.fitness < self.particles[idx].fitness:
            #         self.bestneighbor = deepcopy(self.particles[idx])
            # if self.pocket_particle.fitness < self.bestneighbor.fitness:
            #     better_particle_in_pocket = True
            #     self.pocket_particle = deepcopy(self.bestneighbor)
            #
            # # update the v & x
            # for p in self.particles:
            #     p.v_theta = (self.self_weight * p.v_theta) + self.exp_weight*random.uniform(0, 1)*(p.p_theta - p.theta) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.theta - p.theta)
            #     p.v_weight = (self.self_weight * p.v_weight) + self.exp_weight*random.uniform(0, 1)*(p.p_weight - p.weight) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.weight - p.weight)
            #     p.v_means = (self.self_weight * p.v_means) + self.exp_weight*random.uniform(0, 1)*(p.p_means - p.means) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.means - p.means)
            #     p.v_sd = (self.self_weight * p.v_sd) + self.exp_weight*random.uniform(0, 1)*(p.p_sd - p.sd) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.sd - p.sd)
            #     p.limit_v()
            #     p.update_location()
            #     p.limit_location_upbound()
            self.signals.iteration.emit("------------------------------------------------------")
            self.signals.iteration.emit("Finished running ACO")
            # self.signals.iteration.emit("Average Error: {} \n  (Normalize: {})".format(1/average_error, 1/average_error/40))
            # self.signals.iteration.emit("Worst Error: {} \n  (Normalize: {})".format(1/worst_error, 1/worst_error/40))
            # self.signals.iteration.emit("Least Error: {} \n  (Normalize: {})".format(1/self.pocket_particle.fitness, 1/self.pocket_particle.fitness/40))
            # if better_particle_in_pocket is True:
            self.signals.iteration.emit("Detail parameter:")
            self.signals.iteration.emit("Theta: {}".format(solution[0]))
            self.signals.iteration.emit("Means: {}".format(solution[1:self.neurl_num * 3 + 1]))
            self.signals.iteration.emit("SD: {}".format(solution[self.neurl_num * 3 + 1:self.neurl_num * 4 + 1]))
            self.signals.iteration.emit("Weight: {}".format(solution[self.neurl_num * 4 + 1:]))

        self.signals.iteration.emit("-------------ACO training finished---------------")
        print(solution)
        self.signals.iteration.emit([solution, self.neurl_num, self.dim_i])
        self.signals.result.emit([solution, self.neurl_num, self.dim_i])

    # Original PSO implementation:

    def adaptation_funct_pso(self, index):
        e_n = 0
        # [y_n - F(x_n)]^2
        for idx, expected_output in enumerate(self.train_data.wheel_angle):
            rbfn_value = self.rbfn_funct_pso(self.train_data.v_x[idx], index)
            rbfn_value = max(-40, min(rbfn_value*40, 40))
            e_n += abs(expected_output - rbfn_value)
        # 1/N*(E_n)
        e_n = e_n/len(self.train_data.wheel_angle)
        # since the better parameters producing less e(n), we suppose let it
        # become bigger for reproduction.
        return 1 / e_n

    def rbfn_funct_pso(self, input_vector, idx):

        f_x = self.particles[idx].theta[0]  # theta
        for j in range(self.neurl_num):
            gaussian = self.gaussian_funct(
                j, input_vector, self.particles[idx].means, self.particles[idx].sd[j])
            f_x = f_x + self.particles[idx].weight[j] * gaussian
        return f_x


    # ACO implementation:

    def adaptation_funct_aco(self, solution):
        # e_n = 0
        # # [y_n - F(x_n)]^2
        # for idx, expected_output in enumerate(self.train_data.wheel_angle):
        #     rbfn_value = self.rbfn_funct_aco(self.train_data.v_x[idx], index)
        #     rbfn_value = max(-40, min(rbfn_value*40, 40))
        #     e_n += abs(expected_output - rbfn_value)
        # # 1/N*(E_n)
        # e_n = e_n/len(self.train_data.wheel_angle)
        # # since the better parameters producing less e(n), we suppose let it
        # # become bigger for reproduction.
        # return e_n
        e_n = 0
        # [y_n - F(x_n)]^2
        for idx, expected_output in enumerate(self.train_data.wheel_angle):
            rbfn_value = self.rbfn_funct_aco(self.train_data.v_x[idx], solution)
            rbfn_value = max(-40, min(rbfn_value * 40, 40))
            e_n += abs(expected_output - rbfn_value)
        # 1/N*(E_n)
        e_n = e_n / len(self.train_data.wheel_angle)
        # since the better parameters producing less e(n), we suppose let it
        # become bigger for reproduction.
        return e_n


    def rbfn_funct_aco(self, input_vector, solution):
        f_x = solution[0]  # theta
        for j in range(self.neurl_num):
            gaussian = self.gaussian_funct(
                j, input_vector, solution[j + 1:j + 4], solution[j + 3*self.neurl_num])
            f_x = f_x + solution[j + 4 * self.neurl_num + 1] * gaussian
        return f_x

    def rbfn_funct_pso(self, input_vector, idx):

        f_x = self.particles[idx].theta[0]  # theta
        for j in range(self.neurl_num):
            gaussian = self.gaussian_funct(
                j, input_vector, self.particles[idx].means, self.particles[idx].sd[j])
            f_x = f_x + self.particles[idx].weight[j] * gaussian
        return f_x

    def gaussian_funct(self, jth_neurl, v_x, v_m, o):
        temp = 0
        # means = np.array(
        #     v_m[len(v_x) * jth_neurl:len(v_x) * jth_neurl + self.dim_i])
        temp = (v_x - v_m).dot(v_x - v_m)
        return math.exp(-temp / (2 * o ** 2))

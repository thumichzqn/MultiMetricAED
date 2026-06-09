#%%
import numpy as np

class SHSE_Vec:
    def __init__(self, A, M, T, std = None, xi = None, num_of_iters = None):
        if std is None:
            std = np.ones((A + 1, M))
        if xi is None:
            xi = np.ones((A, M))
        if num_of_iters is None:
            num_of_iters = 100000

        self.action_size = A  # number of actions
        self.metric_size = M  # number of reward metrics of each arm
        self.time_horizon = T  # total budget
        self.num_of_iters = num_of_iters # number of iterations

        self.mean_est = np.zeros((A + 1, M, num_of_iters))  # reward empirical mean estimation
        self.std_est = np.zeros((A + 1, M, num_of_iters))  # std of each arm
        self.var_est = np.zeros((A + 1, M, num_of_iters))  # var of each arm

        self.xi_est = np.zeros((A, M, num_of_iters))  # constant in z-objective
        self.z_est = np.zeros((A, M, num_of_iters))  # z-objective of treatments

        self.pulls_prev_all = np.zeros((A + 1,num_of_iters))  # num of pulls for all prev days
        self.pulls_prev_batch = np.zeros((A + 1, num_of_iters))  # num of pulls for prev days in this batch

        self.active_treatments = np.ones((A, num_of_iters))

        self.traffic_allocation = np.zeros((A + 1, num_of_iters))  # traffic allocation probability for next day

        # Initialization
        self.num_of_batches = np.ceil(np.log2(self.action_size)).astype(int)  # how many arm elimination times
        self.batch_size = round(self.time_horizon / self.num_of_batches)  # num of traffic between eliminations

        self.std_est = np.tile(std[:, :, np.newaxis], (1, 1, num_of_iters))
        self.var_est = np.power(self.std_est, 2)
        self.xi_est = np.tile(xi[:, :, np.newaxis], (1, 1, num_of_iters))
        self.z_est = (self.mean_est[1:, :, :] - self.mean_est[0, :, :]) / np.sqrt(self.var_est[1:, :, :] + self.var_est[0, :, :]) + self.xi_est

        self.traffic_allocation = np.ones((A + 1, num_of_iters)) / self.action_size

    def update_mean_est(self, new_arm_stats, new_arm_pulls):
        """
        update the mean estimate with a new day's sample
        :param new_arm_stats: numpy array of (A+1, M, num_of_iter), the new day observed reward empirical mean for each arm
        :param new_arm_pulls: numpy array of (A+1, num_of_iter), the new day pull num for each arm
        :return:
        """
        pulls_total_new = self.pulls_prev_all + new_arm_pulls

        pre_pulls_fraction = self.pulls_prev_all / pulls_total_new
        weight_prev = np.tile(pre_pulls_fraction[:, np.newaxis, :], (1,self.metric_size, 1))

        self.mean_est = self.mean_est * weight_prev + (1 - weight_prev) * new_arm_stats

        self.pulls_prev_all += new_arm_pulls
        self.pulls_prev_batch += new_arm_pulls
        return
    def update_std_est(self, std):
        """
        update the standard deviation and variance of arms
        :param std: standard deviation, numpy array (A+1, M)
        :return:
        """
        self.std_est = np.tile(std[:, :, np.newaxis], (1, 1, self.num_of_iters))
        self.var_est = np.power(self.std_est, 2)
        return
    def update_xi_est(self, xi):
        """
        update the xi estimate of treatments
        :param xi: (A,M) numpy array, the known constant xi for each arm (determined by validation)
        :return:
        """
        self.xi_est = np.tile(xi[:, :, np.newaxis], (1, 1, self.num_of_iters))
        return
    def update_z_est(self):
        """
        update the z estimate of arms
        :return:
        """
        self.z_est = (self.mean_est[1:, :, :] - self.mean_est[0, :, :]) / np.sqrt(self.var_est[1:, :, :] + self.var_est[0, :, :]) + self.xi_est
        return
    def update_traffic_allocation(self):
        """
        update the traffic allocation based on the effective arm and reward set
        :return:
        """
        new_arm_pull_fraction_per_iter = 1 / (np.sum(self.active_treatments, axis = 0) + 1)
        self.traffic_allocation = np.tile(new_arm_pull_fraction_per_iter[np.newaxis, :], (self.action_size + 1, 1))
        self.traffic_allocation[1:, :] = self.traffic_allocation[1:, :] * self.active_treatments
        return
    def eliminate_treatment(self):
        """
        arm elimination and reward metric clear out
        :return:
        """
        z_est_min = np.min(self.z_est, axis = 1)
        z_est_min_active = np.where(self.active_treatments == 1, z_est_min, np.nan)
        active_median = np.nanmedian(z_est_min_active, axis = 0)
        self.active_treatments = self.active_treatments * (z_est_min >= np.tile(active_median[np.newaxis, :], (self.action_size, 1)))
        return
    def batch_sample_update(self, new_arm_stats, new_arm_pulls):
        """
        update the algorithm at the end of each batch data
        :param new_arm_stats: numpy array of (A+1, M, num_of_iters), the new day observed reward empirical mean for each arm
        :param new_arm_pulls: numpy array of (A+1, M, num_of_iters), the new day pull num for each arm
        :return:
        """
        # update the mean estimate and pull number
        self.update_mean_est(new_arm_stats, new_arm_pulls)

        # update the z estimate
        self.update_z_est()

        # check if stage is finished
        if np.max(np.sum(self.pulls_prev_batch, axis = 0)) >= self.batch_size:

            # eliminate action
            self.eliminate_treatment()
            self.pulls_prev_batch = np.zeros((self.action_size + 1, self.num_of_iters))

            # re-distribute traffic
            self.update_traffic_allocation()
        return self.traffic_allocation
    def recommend_treatment(self):
        """
        output the recommended arm based on the largest empirical z
        :return: arm: the arm index
        """
        z_est_min = np.min(self.z_est, axis=1)
        z_est_min_active = np.where(self.active_treatments == 1, z_est_min, -np.inf)
        best_arm = np.argmax(z_est_min_active, axis = 0) + 1
        return best_arm
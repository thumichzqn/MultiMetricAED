import numpy as np

class MOO:
    def __init__(self, A, M, T, std=None, xi=None):
        """
        initilize the algorithm class
        :param A: scalar, number of treatments
        :param M, number of reward metrics
        :param T: scalar, total number of traffic in MAB problem
        :param std: (A+1, M) numpy array, the known standard deviation of each arm
        :param xi: (A,M) numpy array, the known constant xi for each arm (determined by validation)
        """
        if std is None:
            std = np.ones((A+1,M))
        if xi is None:
            xi = np.ones((A,M))

        self.action_size = A            # number of actions
        self.metric_size = M            # number of reward metrics of each arm
        self.time_horizon = T           # total budget

        self.mean_est = np.zeros((A + 1, M))            # reward empirical mean estimation
        self.std_est = np.zeros((A + 1, M))             # std of each arm
        self.var_est = np.zeros((A + 1, M))             # var of each arm

        self.xi_est = np.zeros((A, M))              # constant in z-objective
        self.z_est = np.zeros((A, M))               # z-objective of treatments
        self.rho_est = np.zeros((A, M))             # fraction of treatment std rho
        self.rho_square_est = np.zeros((A, M))      # fraction of treatment variance rho_square
        self.lambda_est = np.zeros((A, M))          # fraction of control std
        self.lambda_square_est = np.zeros((A, M))   # fraction of control variance lambda_square

        self.pulls_prev_all = np.zeros((A + 1,))  # num of pulls for all prev days
        self.pulls_prev_batch = np.zeros((A + 1,))  # num of pulls for prev days in this batch

        self.active_treatments = list(range(1, A+1))  # active arms that has not been eliminated (A,)
        self.active_metrics = [list(range(M)) for i in self.active_treatments]  # active rewards that has not been eliminated (A, M)

        self.traffic_allocation = np.zeros((A + 1,))  # traffic allocation probability for next day

        # Initialization
        self.num_of_batches = np.ceil(np.log2(self.action_size)).astype(int)        # how many arm elimination times
        self.batch_size = round(self.time_horizon / self.num_of_batches)            # num of traffic between eliminations

        self.std_est = std
        self.var_est = np.power(self.std_est, 2)
        self.rho_square_est = self.var_est[1:, :] / (self.var_est[1:, :] + self.var_est[0])
        self.rho_est = np.sqrt(self.rho_square_est)
        self.lambda_square_est = 1 - self.rho_square_est
        self.lambda_est = np.sqrt(self.lambda_square_est)

        self.xi_est = xi
        self.z_est = (self.mean_est[1:, :] - self.mean_est[0]) / np.sqrt(self.var_est[1:, :] + self.var_est[0]) + self.xi_est

        self.traffic_allocation = np.ones((A+1,)) / self.action_size
        return

    def update_mean_est(self, new_arm_stats, new_arm_pulls):
        """
        update the mean estimate with a new day's sample
        :param new_arm_stats: numpy array of (A+1, M), the new day observed reward empirical mean for each arm
        :param new_arm_pulls: numpy array of (A+1,), the new day pull num for each arm
        :return:
        """
        pulls_total_new = self.pulls_prev_all + new_arm_pulls
        weight_prev = np.tile(self.pulls_prev_all / pulls_total_new, (self.metric_size, 1)).T
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
        self.std_est = std
        self.var_est = np.power(self.std_est, 2)
        self.rho_square_est = self.var_est[1:, :] / (self.var_est[1:, :] + self.var_est[0])
        self.lambda_square_est = 1 - self.rho_est
        self.rho_est = np.sqrt(self.rho_square_est)
        self.lambda_est = np.sqrt(self.lambda_square_est)
        return
    def update_xi_est(self, xi):
        """
        update the xi estimate of treatments
        :param xi: (A,M) numpy array, the known constant xi for each arm (determined by validation)
        :return:
        """
        self.xi_est = xi
        return
    def update_z_est(self):
        """
        update the z estimate of arms
        :return:
        """
        self.z_est = (self.mean_est[1:, :] - self.mean_est[0]) / np.sqrt(self.var_est[1:, :] + self.var_est[0]) + self.xi_est
        return
    def update_traffic_allocation(self):
        """
        update the traffic allocation based on the effective arm and reward set
        :return:
        """
        rho_effective_treatment = np.zeros((self.action_size, ))
        rho_effective_control = 0
        for arm in self.active_treatments:
            a = arm - 1
            rho_effective_treatment[a] = self.rho_est[a, self.active_metrics[a]].max()
            rho_effective_control = max([rho_effective_control, self.lambda_est[a, self.active_metrics[a]].max()])
        rho_square_effective_treatment = rho_effective_treatment ** 2
        rho_treatment_total = np.sqrt(np.sum([rho_square_effective_treatment[a-1] for a in self.active_treatments]))
        self.traffic_allocation[0] = rho_effective_control / (rho_treatment_total + rho_effective_control)
        self.traffic_allocation[1:] = rho_square_effective_treatment / (rho_treatment_total * (rho_treatment_total + rho_effective_control))
        return self.traffic_allocation
    def compute_confidence_interval(self, conf):
        """
        compute the confidence interval of the minimum of z given a certain treatment
        :param confidence: scalar, confidence level delta
        :return: LCB_A: a numpy array (A,) of lcbs
                 UCB_A: a numpy array (A,) of UCBs
        """
        pulls_prev_all_AM = np.tile(self.pulls_prev_all[1:], (self.metric_size,1)).T
        z_var = self.rho_square_est / pulls_prev_all_AM + self.lambda_square_est / self.pulls_prev_all[0]
        LCB_AM = self.z_est - np.sqrt(z_var) * conf
        UCB_AM = self.z_est + np.sqrt(z_var) * conf
        LCB_A = np.min(LCB_AM, axis=1)
        UCB_A = np.min(UCB_AM, axis=1)
        return LCB_A, UCB_A, LCB_AM, UCB_AM
    def index_treatment(self, num_of_iteration = 20):
        """
        compute the delta index of each arm used in elimination
        :param num_of_iteration: how many iterations to run for the binary search
        :return: index: the delta index of each arm used in elimination
        """
        index = np.inf * np.ones((self.action_size,))
        for arm in self.active_treatments:
            a = arm - 1
            # use binary search to find the best confidence level.
            conf_low = 0
            conf_high = 1
            LCB_high, UCB_high, LCB_high_AM, UCB_high_AM = self.compute_confidence_interval(conf_high)
            while UCB_high[a] < np.max(LCB_high):
                conf_low = conf_high
                conf_high = conf_high * 2
                LCB_high, UCB_high, LCB_high_AM, UCB_high_AM = self.compute_confidence_interval(conf_high)

            # use binary search
            conf_mid = conf_high / 2 + conf_low / 2
            for i in range(num_of_iteration):
                LCB_mid, UCB_mid, LCB_mid_AM, UCB_mid_AM = self.compute_confidence_interval(conf_mid)
                if np.max(LCB_mid) < UCB_mid[a]:
                    conf_high = conf_mid
                    conf_mid = conf_high / 2 + conf_low / 2
                else:
                    conf_low = conf_mid
                    conf_mid = conf_low / 2 + conf_high / 2
            index[a] = conf_mid
        return index
    def eliminate_treatment(self):
        """
        arm elimination and reward metric clear out
        :return: delta_total: the confidence of eliminating all arms.
        """

        # compute the delta index
        conf = self.index_treatment(num_of_iteration=100)

        # eliminate based on the median
        active_median = np.median([conf[i-1] for i in self.active_treatments])
        active_treatments_new = (np.where(conf <= active_median)[0] + 1).tolist()

        # clear out the reward metric for eliminated arm
        conf_min = np.inf
        for arm in self.active_treatments:
            a = arm - 1
            if arm not in active_treatments_new:
                self.active_metrics[a] = []
                conf_min = np.min([conf[a], conf_min])

        self.active_treatments = active_treatments_new.copy()
        return conf_min
    def batch_sample_update(self, new_arm_stats, new_arm_pulls):
        """
        update the algorithm at the end of each batch data
        :param new_arm_stats: numpy array of (A+1, M), the new day observed reward empirical mean for each arm
        :param new_arm_pulls: numpy array of (A+1,), the new day pull num for each arm
        :return:
        """
        # update the mean estimate and pull number
        self.update_mean_est(new_arm_stats, new_arm_pulls)

        # update the z estimate
        self.update_z_est()

        # check if stage is finished
        if np.sum(self.pulls_prev_batch) >= self.batch_size:

            # eliminate action
            conf = self.eliminate_treatment()
            self.pulls_prev_batch = np.zeros((self.action_size + 1,))

            # eliminate reward metric
            if conf > 1:
                LCB, UCB, LCB_AM, UCB_AM = self.compute_confidence_interval(conf)
                for arm in self.active_treatments:
                    a = arm - 1
                    for i in self.active_metrics[a]:
                        active_metrics_new = self.active_metrics[a].copy()
                        if LCB_AM[a,i] > UCB[a] and len(active_metrics_new) > 1:
                            active_metrics_new.remove(i)
                        self.active_metrics[a] = active_metrics_new.copy()
            # re-distribute traffic
            self.update_traffic_allocation()
        return self.traffic_allocation
    def recommend_treatment(self):
        """
        output the recommended arm based on the largest empirical mean
        :return: arm: the arm index
        """
        best_arm = 0
        best_z = - np.inf
        for a in self.active_treatments:
            if np.min(self.z_est[a-1, :]) > best_z:
                best_z = np.min(self.z_est[a-1, :])
                best_arm = a
        return best_arm

#%%
class MOO_Vec:
    def __init__(self, A, M, T, std=None, xi=None, num_of_iters=None):
        if std is None:
            std = np.ones((A + 1, M))
        if xi is None:
            xi = np.ones((A, M))
        if num_of_iters is None:
            num_of_iters = 100000

        self.action_size = A  # number of actions
        self.metric_size = M  # number of reward metrics of each arm
        self.time_horizon = T  # total budget
        self.num_of_iters = num_of_iters  # number of iterations

        self.mean_est = np.zeros((A + 1, M, num_of_iters))  # reward empirical mean estimation
        self.std_est = np.zeros((A + 1, M, num_of_iters))  # std of each arm
        self.var_est = np.zeros((A + 1, M, num_of_iters))  # var of each arm

        self.rho_est = np.zeros((A, M, num_of_iters))  # fraction of treatment std rho
        self.rho_square_est = np.zeros((A, M, num_of_iters))  # fraction of treatment variance rho_square
        self.lambda_est = np.zeros((A, M, num_of_iters))  # fraction of control std
        self.lambda_square_est = np.zeros((A, M, num_of_iters))  # fraction of control variance lambda_square

        self.xi_est = np.zeros((A, M, num_of_iters))  # constant in z-objective
        self.z_est = np.zeros((A, M, num_of_iters))  # z-objective of treatments

        self.pulls_prev_all = np.zeros((A + 1, num_of_iters))  # num of pulls for all prev days
        self.pulls_prev_batch = np.zeros((A + 1, num_of_iters))  # num of pulls for prev days in this batch

        self.active_treatments = np.ones((A, num_of_iters))
        self.active_metrics = np.ones((A, M, num_of_iters))

        self.traffic_allocation = np.zeros((A + 1, num_of_iters))  # traffic allocation probability for next day

        # Initialization
        self.num_of_batches = np.ceil(np.log2(self.action_size)).astype(int)        # how many arm elimination times
        self.batch_size = round(self.time_horizon / self.num_of_batches)            # num of traffic between eliminations

        self.std_est = np.tile(std[:, :, np.newaxis], (1, 1, num_of_iters))
        self.var_est = np.power(self.std_est, 2)

        self.rho_square_est = self.var_est[1:, :, :] / (self.var_est[1:, :, :] + self.var_est[0, :, :])
        self.rho_est = np.sqrt(self.rho_square_est)
        self.lambda_square_est = 1 - self.rho_square_est
        self.lambda_est = np.sqrt(self.lambda_square_est)

        self.xi_est = np.tile(xi[:, :, np.newaxis], (1, 1, num_of_iters))
        self.z_est = (self.mean_est[1:, :, :] - self.mean_est[0, :, :]) / np.sqrt(
            self.var_est[1:, :, :] + self.var_est[0, :, :]) + self.xi_est

        self.traffic_allocation = np.ones((A + 1, num_of_iters)) / self.action_size
        return

    def update_mean_est(self, new_arm_stats, new_arm_pulls):
        """
        update the mean estimate with a new day's sample
        :param new_arm_stats: numpy array of (A+1, M, num_of_iter), the new day observed reward empirical mean for each arm
        :param new_arm_pulls: numpy array of (A+1, num_of_iter), the new day pull num for each arm
        :return:
        """
        pulls_total_new = self.pulls_prev_all + new_arm_pulls

        pre_pulls_fraction = self.pulls_prev_all / pulls_total_new
        weight_prev = np.tile(pre_pulls_fraction[:, np.newaxis, :], (1, self.metric_size, 1))

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

        self.rho_square_est = self.var_est[1:, :, :] / (self.var_est[1:, :, :] + self.var_est[0, :, :])
        self.rho_est = np.sqrt(self.rho_square_est)
        self.lambda_square_est = 1 - self.rho_square_est
        self.lambda_est = np.sqrt(self.lambda_square_est)
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
        self.z_est = (self.mean_est[1:, :, :] - self.mean_est[0, :, :]) / np.sqrt(
            self.var_est[1:, :, :] + self.var_est[0, :, :]) + self.xi_est
        return

    def update_traffic_allocation(self):      # MOO traffic
        """
        update the traffic allocation based on the effective arm and reward set
        :return:
        """
        rho_effective_treatment = np.max(self.rho_est * self.active_metrics, axis=1)
        rho_effective_treatment_active = rho_effective_treatment * self.active_treatments
        rho_square_effective_treatment_active = rho_effective_treatment_active ** 2

        rho_square_total_active = np.sum(rho_square_effective_treatment_active, axis=0)
        rho_total_active = np.sqrt(rho_square_total_active)

        rho_effective_control = np.max(np.max(self.lambda_est, axis=1), axis=0)

        self.traffic_allocation[0, :] = rho_effective_control / (rho_total_active + rho_effective_control)
        numerator = (rho_total_active + rho_effective_control) * rho_total_active
        self.traffic_allocation[1:, :] = rho_square_effective_treatment_active / np.tile(numerator[np.newaxis, :],
                                                                                         (self.action_size, 1))
        return

    # def update_traffic_allocation(self):          # SH traffic
    #     """
    #     update the traffic allocation based on the effective arm and reward set
    #     :return:
    #     """
    #     new_arm_pull_fraction_per_iter = 1 / (np.sum(self.active_treatments, axis = 0) + 1)
    #     self.traffic_allocation = np.tile(new_arm_pull_fraction_per_iter[np.newaxis, :], (self.action_size + 1, 1))
    #     self.traffic_allocation[1:, :] = self.traffic_allocation[1:, :] * self.active_treatments
    #     return

    # def update_traffic_allocation(self):            # SHVar traffic
    #     """
    #     update the traffic allocation based on the effective arm and reward set
    #     :return:
    #     """
    #     var_est_max_traffic = np.max(self.var_est, axis = 1)
    #     var_est_max_traffic[1:, :] = var_est_max_traffic[1:, :] * self.active_treatments
    #     var_total_traffic = np.sum(var_est_max_traffic, axis = 0)
    #     self.traffic_allocation = var_est_max_traffic / np.tile(var_total_traffic[np.newaxis, :], (self.action_size + 1, 1))
    #     return

    def compute_confidence_interval(self, conf):
        """
        compute the confidence interval of the minimum of z given a certain treatment
        :param confidence: scalar, confidence level delta
        :return: LCB_A: a numpy array (A,) of lcbs
                 UCB_A: a numpy array (A,) of UCBs
        """
        pulls_prev_all_AMN = np.tile(self.pulls_prev_all[:, np.newaxis, :], (1, self.metric_size, 1))
        pulls_prev_treatment_AMN = pulls_prev_all_AMN[1:, :, :]
        pulls_prev_control_MN = pulls_prev_all_AMN[0, :, :]
        z_var = self.rho_square_est / pulls_prev_treatment_AMN + self.lambda_square_est / np.tile(pulls_prev_control_MN[np.newaxis, :, :], (self.action_size, 1, 1))

        LCB_AMN = self.z_est - np.sqrt(z_var) * np.tile(conf[:, np.newaxis,:], (1, self.metric_size, 1))
        UCB_AMN = self.z_est + np.sqrt(z_var) * np.tile(conf[:, np.newaxis,:], (1, self.metric_size, 1))
        LCB_AN = np.min(LCB_AMN, axis=1)
        UCB_AN = np.min(UCB_AMN, axis=1)
        return LCB_AN, UCB_AN, LCB_AMN, UCB_AMN

    def eliminate_treatment(self):
        """
        arm elimination and reward metric clear out
        :return: delta_total: the confidence of eliminating all arms.
        """
        eps = 1e-6

        conf_low = np.zeros((self.action_size, self.num_of_iters))
        conf_high = np.ones((self.action_size, self.num_of_iters))
        LCB_high, UCB_high, LCB_high_AM, UCB_high_AM = self.compute_confidence_interval(conf_high)
        index = (UCB_high < np.tile(np.max(LCB_high, axis = 0)[np.newaxis,:], (self.action_size, 1)))
        while np.any(index) == 1:
            conf_low[index == 1] = conf_high[index == 1]
            conf_high[index == 1] = conf_high[index == 1] * 2
            LCB_high, UCB_high, LCB_high_AM, UCB_high_AM = self.compute_confidence_interval(conf_high)
            index = (UCB_high < np.tile(np.max(LCB_high, axis=0)[np.newaxis, :], (self.action_size, 1)))

        conf_mid = conf_high / 2 + conf_low / 2
        converged = np.zeros((self.action_size, self.num_of_iters))

        while np.all(converged) == 0:
            LCB_mid, UCB_mid, LCB_mid_AM, UCB_mid_AM = self.compute_confidence_interval(conf_high)

            converged = converged + (1 - converged) * (np.abs(conf_high - conf_low) < eps)

            index_to_increase = (UCB_mid < np.tile(np.max(LCB_mid, axis = 0)[np.newaxis,:], (self.action_size, 1))) * (1 - converged)
            index_to_decrease = (UCB_mid > np.tile(np.max(LCB_mid, axis = 0)[np.newaxis,:], (self.action_size, 1))) * (1 - converged)

            conf_low[index_to_increase == 1] = conf_mid[index_to_increase == 1]
            conf_high[index_to_decrease == 1] = conf_mid[index_to_decrease == 1]
            conf_mid = conf_high / 2 + conf_low / 2

        conf = conf_mid
        conf_active = np.where(self.active_treatments == 1, conf, np.nan)

        # eliminate based on the median
        active_median = np.nanmedian(conf_active, axis=0)
        eliminated_treatments = (conf_active > np.tile(active_median[np.newaxis, :], (self.action_size, 1)))
        self.active_treatments = self.active_treatments * (1 - eliminated_treatments)

        self.active_metrics[np.tile(eliminated_treatments[:, np.newaxis, :], (1, self.metric_size, 1)) == 1] = 0

        conf_eliminated = conf * eliminated_treatments
        conf_eliminated[eliminated_treatments == 0] = np.inf
        conf_min = np.min(conf_eliminated, axis=0)
        return conf_min

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
        if np.max(np.sum(self.pulls_prev_batch, axis=0)) >= self.batch_size:
            # eliminate action
            conf = self.eliminate_treatment()
            self.pulls_prev_batch = np.zeros((self.action_size + 1, self.num_of_iters))

            # eliminate reward metric
            if np.all(conf > 1):
                LCB, UCB, LCB_AM, UCB_AM = self.compute_confidence_interval(np.tile(conf[np.newaxis, :], (self.action_size, 1)))
                index_to_eliminate_metric = (LCB_AM > np.tile(UCB[:, np.newaxis, :], (1, self.metric_size, 1)))
                self.active_metrics[index_to_eliminate_metric == 1] = 0
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
        best_arm = np.argmax(z_est_min_active, axis=0) + 1
        return best_arm

#%%
if 0:
    A = 6
    M = 2
    num_of_iter = 10
    np.random.seed(0)

    mean = np.zeros((A + 1, M))
    std = np.zeros((A + 1, M))
    xi = np.zeros((A, M))
    for i in range(A):
        xi[i, 0] = 0.43 / np.sqrt(1e8) * 0
        xi[i, 1] = 0.43 / np.sqrt(1e8) * 0

    z = np.zeros((A, M))
    for i in range(A):
        if i % 2 == 0:
            z[i, 0] = 0.1 - 0.01 * i
            z[i, 1] = 0.11
        else:
            z[i, 1] = 0.1 - 0.01 * i
            z[i, 0] = 0.11
    rho_square = np.zeros((A, M))
    for i in range(A):
        if i % 2 == 0:
            rho_square[i, 0] = 0.1 * (i + 1)
            rho_square[i, 1] = 1 - 0.1 * (i + 1)
        else:
            rho_square[i, 1] = 0.1 * (i + 1)
            rho_square[i, 0] = 1 - 0.1 * (i + 1)
    std[0, 0] = 10
    std[0, 1] = 30

    for i in range(A):
        std[i + 1, 0] = np.sqrt(rho_square[i, 0] / (1 - rho_square[i, 0]) * (std[0, 0] ** 2))
        std[i + 1, 1] = np.sqrt(rho_square[i, 1] / (1 - rho_square[i, 1]) * (std[0, 1] ** 2))

    mean[0, 0] = 0
    mean[0, 1] = 0

    mean[1:, :] = (z - xi) * np.sqrt(std[1:, :] ** 2 + std[0, :] ** 2)

    z_arm = np.min(z, axis=1)
    metric_arm = np.argmin(z, axis=1)

    best_treatment = np.argmax(np.min(z, axis=1)) + 1

    T = 1000000

    agent = MOO_Vec(A, M, T, std, xi, num_of_iter)
    agent.update_z_est()
    agent.update_traffic_allocation()

    num_of_batches = int(np.floor(np.log2(A)))
    batch_size = int(np.ceil(T / num_of_batches))
    for batch in range(num_of_batches):
        new_arm_pulls = np.ceil(agent.traffic_allocation * batch_size)
        new_arm_pulls = np.maximum(new_arm_pulls, 1)

        mean_for_rand = np.tile(mean[:, :, np.newaxis], (1, 1, num_of_iter))
        std_for_rand = np.tile(std[:, :, np.newaxis], (1, 1, num_of_iter)) / np.sqrt(
            np.tile(new_arm_pulls[:, np.newaxis, :], (1, M, 1)))
        new_arm_stats = np.random.normal(loc=mean_for_rand, scale=std_for_rand)

        new_arm_stats[1:, :, :] = new_arm_stats[1:, :, :] * np.tile(agent.active_treatments[:, np.newaxis, :],
                                                                    (1, M, 1))

        agent.batch_sample_update(new_arm_stats, new_arm_pulls)
    print(agent.recommend_treatment() == best_treatment)
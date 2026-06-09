import numpy as np
import matplotlib.pyplot as plt
import json

np.random.seed(42)

A = 27
Te = 500
Tv = 100
num_of_iter = 100000

mean = np.zeros((A + 1,))
std = np.zeros((A + 1,))

z = np.zeros((A,))
z = (3 - np.sqrt(np.arange(1, A+1))) * 0.1

heterogeneity_set = list(np.linspace(0,5, 11))

result_SHVar = []
rate_SHVar = []
result_MOO = []
rate_MOO = []
result_SH = []
rate_SH = []
result_SHZ = []
rate_SHZ = []
result_SHVarZ = []
rate_SHVarZ = []

for heterogeneity in heterogeneity_set:
    std[0] = 1
    std[1:] = np.sqrt(np.arange(1, A + 1) ** heterogeneity) + 1

    xi = np.zeros((A,))

    mean[0] = 0
    mean[1:] = (z - xi) * np.sqrt(std[1:] ** 2 + std[0] ** 2)

    z = (mean[1:] - mean[0]) / np.sqrt(std[1:] ** 2 + std[0] ** 2) + xi
    best_treatment = np.argmax(z) + 1
    var = std ** 2


    # compute the proportion of variance for treatments rho and lambda
    rho_square = var[1:] / (var[1:] + var[0])           # rho^2 numpy array (A,)
    lambd_square = 1 - rho_square                       # 1 - rho^2 numpy array (A,)
    rho = np.sqrt(rho_square)                           # rho numpy array (A,)
    lambd = np.sqrt(lambd_square)                       # lambda numpy array (A,)

    #%% SHRVar
    if 1:
        result = []
        for iter in range(num_of_iter):
            active_indicator = np.ones((A + 1,))                # index of active treatments numpy array (A+1,), control is always 0
            active_indicator[0] = 0

            mean_est = np.zeros((A + 1,))                       # estimate of empirical mean, numpy array (A+1,)
            num_of_pulls = np.zeros((A + 1,))                   # number of pulls before, numpy array (A+1,)

            num_of_batches = int(np.ceil(np.log2(A)))
            batch_size = Te / num_of_batches

            for t in range(num_of_batches):
                # compute the index of the treatment to calculate the total effective variance rho
                treatment_index = (active_indicator[1:] == 1)           # suitable for array (A,)
                arm_index = (active_indicator == 1)                     # suitable for array (A+1,)

                # compute the traffic allocation constant for treatment and conrtol, rho_treatment_total = sqrt(sum(rho^2))
                rho_control = max(lambd[treatment_index]) - min(lambd[treatment_index])         # rho_control = rho_0
                rho_treatment_total = np.sqrt(np.sum(rho_square[treatment_index]))              # rho_treatment = sum of rho(a) active

                # allocate traffic based on it
                traffic_allocation = np.zeros((A+1,))           # traffic allocation probability, numpy array (A+1,)
                traffic_allocation[0] = rho_control / (rho_treatment_total + rho_control)
                traffic_allocation[arm_index] = rho_square[treatment_index] / ((rho_treatment_total + rho_control) * rho_treatment_total)

                # sample new data based on traffic allocation
                num_of_new_pulls = traffic_allocation * batch_size      # the number of new impressions the next day after designing the traffic, numpy array (A+1,)
                arm_update_index = (num_of_new_pulls > 0.001)   # index of the arms which receive non-zero traffic (A+1,)

                new_sample_mean = np.zeros((A + 1,))            # the sample mean of new traffic numpy array (A+1,)
                new_sample_mean[arm_update_index] = (np.random.normal(0, 1, np.sum(arm_update_index))
                                                         * (std[arm_update_index] / np.sqrt(num_of_new_pulls[arm_update_index]))
                                                         + mean[arm_update_index])

                # update the model with mean estimation and the number of pulls each day
                mean_est[arm_update_index] = ((mean_est[arm_update_index] * num_of_pulls[arm_update_index] + new_sample_mean[arm_update_index] * num_of_new_pulls[arm_update_index])
                                              / (num_of_pulls[arm_update_index] + num_of_new_pulls[arm_update_index]))
                num_of_pulls = num_of_pulls + num_of_new_pulls

                # update the z estimation
                z_est = (mean_est[1:] - mean_est[0]) / np.sqrt(var[1:] + var[0]) + xi

                # arm elimination
                # compute the elimination threshold based on median of the z_estimate
                elimination_threshold = np.median(z_est[treatment_index])

                # compute the index of arms that will be eliminated:
                elimination_index = np.insert((z_est < elimination_threshold), 0, True) | (active_indicator == 0)

                # eliminate the arm in active_arm
                active_indicator[elimination_index] = False

            # winner selection based on the largest z estimate
            winner = np.argmax(- (1 - active_indicator[1:]) * 1e8 + z_est) + 1


            # Validation Phase
            output_treatment = winner

            sample_mean_control = np.random.normal(mean[0], std[0] / np.sqrt(Tv), 1)
            sample_mean_output = np.random.normal(mean[output_treatment], std[output_treatment] / np.sqrt(Tv), 1)

            val_success = (sample_mean_output - sample_mean_control) > 0

            result.append(val_success[0].item())
        result_MOO.append(result)
        rate_MOO.append(np.average(result))
        print('MOO', rate_MOO)
    #%% SH-Z
    if 1:
        result = []
        for iter in range(num_of_iter):
            active_indicator = np.ones((A + 1,))                # index of active treatments numpy array (A+1,), control is always 0
            active_indicator[0] = 0

            mean_est = np.zeros((A + 1,))                       # estimate of empirical mean, numpy array (A+1,)
            num_of_pulls = np.zeros((A + 1,))                   # number of pulls before, numpy array (A+1,)

            num_of_batches = int(np.ceil(np.log2(A)))
            batch_size = Te / num_of_batches

            for t in range(num_of_batches):
                # compute the index of the treatment to calculate the total effective variance rho
                treatment_index = (active_indicator[1:] == 1)           # suitable for array (A,)
                arm_index = (active_indicator == 1)                     # suitable for array (A+1,)

                # allocate traffic based on it
                traffic_allocation = np.zeros((A + 1,))  # traffic allocation probability, numpy array (A+1,)
                traffic_allocation[0] = 1 / (np.sum(arm_index) + 1)
                traffic_allocation[arm_index] = 1 / (np.sum(arm_index) + 1)

                # sample new data based on traffic allocation
                num_of_new_pulls = traffic_allocation * batch_size      # the number of new impressions the next day after designing the traffic, numpy array (A+1,)
                arm_update_index = (num_of_new_pulls > 0.001)   # index of the arms which receive non-zero traffic (A+1,)

                new_sample_mean = np.zeros((A + 1,))            # the sample mean of new traffic numpy array (A+1,)
                new_sample_mean[arm_update_index] = (np.random.normal(0, 1, np.sum(arm_update_index))
                                                         * (std[arm_update_index] / np.sqrt(num_of_new_pulls[arm_update_index]))
                                                         + mean[arm_update_index])

                # update the model with mean estimation and the number of pulls each day
                mean_est[arm_update_index] = ((mean_est[arm_update_index] * num_of_pulls[arm_update_index] + new_sample_mean[arm_update_index] * num_of_new_pulls[arm_update_index])
                                              / (num_of_pulls[arm_update_index] + num_of_new_pulls[arm_update_index]))
                num_of_pulls = num_of_pulls + num_of_new_pulls

                # update the z estimation
                z_est = (mean_est[1:] - mean_est[0]) / np.sqrt(var[1:] + var[0]) + xi

                # arm elimination
                # compute the elimination threshold based on median of the z_estimate
                elimination_threshold = np.median(z_est[treatment_index])

                # compute the index of arms that will be eliminated:
                elimination_index = np.insert((z_est < elimination_threshold), 0, True) | (active_indicator == 0)

                # eliminate the arm in active_arm
                active_indicator[elimination_index] = False

            # winner selection based on the largest z estimate
            winner = np.argmax(- (1 - active_indicator[1:]) * 1e8 + z_est) + 1


            # Validation Phase
            output_treatment = winner

            sample_mean_control = np.random.normal(mean[0], std[0] / np.sqrt(Tv), 1)
            sample_mean_output = np.random.normal(mean[output_treatment], std[output_treatment] / np.sqrt(Tv), 1)

            val_success = (sample_mean_output - sample_mean_control) > 0

            result.append(val_success[0].item())
        result_SHZ.append(result)
        rate_SHZ.append(np.average(result))
        print('SH-Z', rate_SHZ)
    #%% SH benchmark

    if 1:
        result = []
        for iter in range(num_of_iter):
            active_indicator = np.ones((A + 1,))                # index of active treatments numpy array (A+1,), control is always 0
            active_indicator[0] = 0

            mean_est = np.zeros((A + 1,))                       # estimate of empirical mean, numpy array (A+1,)
            num_of_pulls = np.zeros((A + 1,))                   # number of pulls before, numpy array (A+1,)

            num_of_batches = int(np.ceil(np.log2(A)))
            batch_size = Te / num_of_batches

            for t in range(num_of_batches):
                # compute the index of the treatment to calculate the total effective variance rho
                treatment_index = (active_indicator[1:] == 1)           # suitable for array (A,)
                arm_index = (active_indicator == 1)                     # suitable for array (A+1,)

                # allocate traffic based on it
                traffic_allocation = np.zeros((A+1,))           # traffic allocation probability, numpy array (A+1,)
                traffic_allocation[0] = 1 / (np.sum(arm_index) + 1)
                traffic_allocation[arm_index] = 1 / (np.sum(arm_index) + 1)

                # sample new data based on traffic allocation
                num_of_new_pulls = traffic_allocation * batch_size      # the number of new impressions the next day after designing the traffic, numpy array (A+1,)
                arm_update_index = (num_of_new_pulls > 0.001)   # index of the arms which receive non-zero traffic (A+1,)

                new_sample_mean = np.zeros((A + 1,))            # the sample mean of new traffic numpy array (A+1,)
                new_sample_mean[arm_update_index] = (np.random.normal(0, 1, np.sum(arm_update_index))
                                                         * (std[arm_update_index] / np.sqrt(num_of_new_pulls[arm_update_index]))
                                                         + mean[arm_update_index])

                # update the model with mean estimation and the number of pulls each day
                mean_est[arm_update_index] = ((mean_est[arm_update_index] * num_of_pulls[arm_update_index] + new_sample_mean[arm_update_index] * num_of_new_pulls[arm_update_index])
                                              / (num_of_pulls[arm_update_index] + num_of_new_pulls[arm_update_index]))
                num_of_pulls = num_of_pulls + num_of_new_pulls

                # arm elimination
                # compute the elimination threshold based on median of the mean estimate
                elimination_threshold = np.median(mean_est[arm_index])

                # compute the index of arms that will be eliminated:
                elimination_index = (mean_est < elimination_threshold) | (active_indicator == 0)

                # eliminate the arm in active_arm
                active_indicator[elimination_index] = False

            # winner selection based on the largest z estimate
            winner = np.argmax(- (1 - active_indicator[1:]) * 1e8 + mean_est[1:]) + 1


            # Validation Phase
            output_treatment = winner

            sample_mean_control = np.random.normal(mean[0], std[0] / np.sqrt(Tv), 1)
            sample_mean_output = np.random.normal(mean[output_treatment], std[output_treatment] / np.sqrt(Tv), 1)

            val_success = (sample_mean_output - sample_mean_control) > 0

            result.append(val_success[0].item())

        result_SH.append(result)
        rate_SH.append(np.average(result))
        print('SH', rate_SH)

    # %% SHVar-Z
    if 1:
        result = []
        for iter in range(num_of_iter):
            active_indicator = np.ones((A + 1,))  # index of active treatments numpy array (A+1,), control is always 0
            active_indicator[0] = 0

            mean_est = np.zeros((A + 1,))  # estimate of empirical mean, numpy array (A+1,)
            num_of_pulls = np.zeros((A + 1,))  # number of pulls before, numpy array (A+1,)

            num_of_batches = int(np.ceil(np.log2(A)))
            batch_size = Te / num_of_batches

            for t in range(num_of_batches):
                # compute the index of the treatment to calculate the total effective variance rho
                treatment_index = (active_indicator[1:] == 1)  # suitable for array (A,)
                arm_index = (active_indicator == 1)  # suitable for array (A+1,)

                # allocate traffic based on it
                traffic_allocation = np.zeros((A + 1,))  # traffic allocation probability, numpy array (A+1,)
                traffic_allocation[0] = var[0] / (np.sum(var[arm_index]) + var[0])
                traffic_allocation[arm_index] = var[arm_index] / (np.sum(var[arm_index]) + var[0])

                # sample new data based on traffic allocation
                num_of_new_pulls = traffic_allocation * batch_size  # the number of new impressions the next day after designing the traffic, numpy array (A+1,)
                arm_update_index = (num_of_new_pulls > 0.001)  # index of the arms which receive non-zero traffic (A+1,)

                new_sample_mean = np.zeros((A + 1,))  # the sample mean of new traffic numpy array (A+1,)
                new_sample_mean[arm_update_index] = (np.random.normal(0, 1, np.sum(arm_update_index))
                                                     * (std[arm_update_index] / np.sqrt(
                            num_of_new_pulls[arm_update_index]))
                                                     + mean[arm_update_index])

                # update the model with mean estimation and the number of pulls each day
                mean_est[arm_update_index] = ((mean_est[arm_update_index] * num_of_pulls[arm_update_index] +
                                               new_sample_mean[arm_update_index] * num_of_new_pulls[arm_update_index])
                                              / (num_of_pulls[arm_update_index] + num_of_new_pulls[arm_update_index]))
                num_of_pulls = num_of_pulls + num_of_new_pulls

                # update the z estimation
                z_est = (mean_est[1:] - mean_est[0]) / np.sqrt(var[1:] + var[0]) + xi

                # arm elimination
                # compute the elimination threshold based on median of the z_estimate
                elimination_threshold = np.median(z_est[treatment_index])

                # compute the index of arms that will be eliminated:
                elimination_index = np.insert((z_est < elimination_threshold), 0, True) | (active_indicator == 0)

                # eliminate the arm in active_arm
                active_indicator[elimination_index] = False

            # winner selection based on the largest z estimate
            winner = np.argmax(- (1 - active_indicator[1:]) * 1e8 + z_est) + 1

            # Validation Phase
            output_treatment = winner

            sample_mean_control = np.random.normal(mean[0], std[0] / np.sqrt(Tv), 1)
            sample_mean_output = np.random.normal(mean[output_treatment], std[output_treatment] / np.sqrt(Tv), 1)

            val_success = (sample_mean_output - sample_mean_control) > 0

            result.append(val_success[0].item())
        result_SHVarZ.append(result)
        rate_SHVarZ.append(np.average(result))
        print('SHVar-Z', rate_SHVarZ)

#%% SHVar benchmark
    if 1:
        result = []
        for iter in range(num_of_iter):
            active_indicator = np.ones((A + 1,))                # index of active treatments numpy array (A+1,), control is always 0
            active_indicator[0] = 0

            mean_est = np.zeros((A + 1,))                       # estimate of empirical mean, numpy array (A+1,)
            num_of_pulls = np.zeros((A + 1,))                   # number of pulls before, numpy array (A+1,)

            num_of_batches = int(np.ceil(np.log2(A)))
            batch_size = Te / num_of_batches

            for t in range(num_of_batches):
                # compute the index of the treatment to calculate the total effective variance rho
                treatment_index = (active_indicator[1:] == 1)           # suitable for array (A,)
                arm_index = (active_indicator == 1)                     # suitable for array (A+1,)

                # allocate traffic based on it
                traffic_allocation = np.zeros((A+1,))           # traffic allocation probability, numpy array (A+1,)
                traffic_allocation[0] = var[0] / (np.sum(var[arm_index]) + var[0])
                traffic_allocation[arm_index] = var[arm_index] / (np.sum(var[arm_index]) + var[0])

                # sample new data based on traffic allocation
                num_of_new_pulls = traffic_allocation * batch_size      # the number of new impressions the next day after designing the traffic, numpy array (A+1,)
                arm_update_index = (num_of_new_pulls > 0.001)   # index of the arms which receive non-zero traffic (A+1,)

                new_sample_mean = np.zeros((A + 1,))            # the sample mean of new traffic numpy array (A+1,)
                new_sample_mean[arm_update_index] = (np.random.normal(0, 1, np.sum(arm_update_index))
                                                         * (std[arm_update_index] / np.sqrt(num_of_new_pulls[arm_update_index]))
                                                         + mean[arm_update_index])

                # update the model with mean estimation and the number of pulls each day
                mean_est[arm_update_index] = ((mean_est[arm_update_index] * num_of_pulls[arm_update_index] + new_sample_mean[arm_update_index] * num_of_new_pulls[arm_update_index])
                                              / (num_of_pulls[arm_update_index] + num_of_new_pulls[arm_update_index]))
                num_of_pulls = num_of_pulls + num_of_new_pulls

                # arm elimination
                # compute the elimination threshold based on median of the mean estimate
                elimination_threshold = np.median(mean_est[arm_index])

                # compute the index of arms that will be eliminated:
                elimination_index = (mean_est < elimination_threshold) | (active_indicator == 0)

                # eliminate the arm in active_arm
                active_indicator[elimination_index] = False

            # winner selection based on the largest z estimate
            winner = np.argmax(- (1 - active_indicator[1:]) * 1e8 + mean_est[1:]) + 1


            # Validation Phase
            output_treatment = winner

            sample_mean_control = np.random.normal(mean[0], std[0] / np.sqrt(Tv), 1)
            sample_mean_output = np.random.normal(mean[output_treatment], std[output_treatment] / np.sqrt(Tv), 1)

            val_success = (sample_mean_output - sample_mean_control) > 0

            result.append(val_success[0].item())
        result_SHVar.append(result)
        rate_SHVar.append(np.average(result))
        print('SHVar', rate_SHVar)
#%%
print()
with open('Data/heterogeneity.json', 'w') as f:
    json.dump(heterogeneity_set, f)

with open('Data/MOO_end2end.json', 'w') as f:
    json.dump(result_MOO, f)

with open('Data/SH_end2end.json', 'w') as f:
    json.dump(result_SH, f)

with open('Data/SHZ_end2end.json', 'w') as f:
    json.dump(result_SHZ, f)

with open('Data/SHVar_end2end.json', 'w') as f:
    json.dump(result_SHVar, f)

with open('Data/SHVarZ_end2end.json', 'w') as f:
    json.dump(result_SHVarZ, f)


#%%
plt.plot(heterogeneity_set, rate_MOO, label = 'SHRVar')
plt.plot(heterogeneity_set, rate_SH, label = 'SH')
plt.plot(heterogeneity_set, rate_SHVar, label = 'SHVar')
plt.plot(heterogeneity_set, rate_SHZ, label = 'SHZ')
plt.plot(heterogeneity_set, rate_SHVarZ, label = 'SHVarZ')
plt.legend()
plt.xlabel('Variance Heterogeneity')
plt.ylabel('Validation Sucess Rate')
plt.show()

print()
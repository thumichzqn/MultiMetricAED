import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from Algorithms.MOOSE import MOOSE_Vec
from Algorithms.SHSE import SHSE_Vec
from Algorithms.SHVarSE import SHVarSE_Vec
from Algorithms.SHStdSE import SHStdSE_Vec
import json
#%%
if 1:
    A = 128
    M = 3
    num_of_iter = 100000
    np.random.seed(0)

    mean = np.zeros((A + 1, M))
    std = np.zeros((A + 1, M))
    xi = np.zeros((A, M))
    z = np.zeros((A, M))
    z[0, :] = np.array([0.15, 0.15, 0.05])
    for a in range(1, A):
        z[a, :] = np.array([-0.05, 0.25, 0.25])

    rho_square = np.zeros((A, M))
    rho_square[0, :] = np.array([0.8, 0.5, 0.2])
    for a in range(1, A):
        rho_square[a, :] = np.array([0.8, 0.5, 0.2]) + np.random.uniform(-0.1, 0.1, (1,M))

    std[0, :] = np.array([1, 1, 1])
    std[1:, :] = np.sqrt(rho_square[:, :] / (1 - rho_square[:, :]) * np.tile((std[0, :] ** 2), (A, 1)))

    mean[0, :] = np.array([0, 0, 0])
    mean[1:, :] = (z - xi) * np.sqrt(std[1:, :] ** 2 + std[0, :] ** 2)

    z_arm = np.min(z, axis=1)
    metric_arm = np.argmin(z, axis=1)

    best_treatment = np.argmax(np.min(z, axis=1)) + 1

    num_set = np.linspace(5, 14, 10)

    T_set = np.ceil(num_set * 1e4).tolist()
    Tv = 2000
    data = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'z': z.tolist(),
        'rho_square': rho_square.tolist(),
        'xi': xi.tolist(),
        'best_treatment': int(best_treatment),
        'T': T_set,
    }
    with open('./Data_128/setup_128.json', 'w') as f:
        json.dump(data, f)

#%% MOOSE
if 1:
    print('MOOSE')
    result = []
    success_rate = []
    type_one_error = []
    error_rate = []
    val_pass = []
    val_pass_rate = []
    for T in T_set:
        agent = MOOSE_Vec(A, M, T, std, xi, num_of_iter)
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
        recomended_treatment = agent.recommend_treatment()
        recomended_z = z_arm[recomended_treatment - 1]

        prob_of_pass = stats.norm.cdf(recomended_z * np.sqrt(Tv / 2))
        pass_or_not = np.random.binomial(1, prob_of_pass)
        error = pass_or_not * (recomended_z < 0)

        error_rate.append(np.average(error))
        type_one_error.append(error.tolist())

        val_pass_rate.append(np.average(pass_or_not))
        val_pass.append(pass_or_not.tolist())

        result.append((recomended_treatment == best_treatment).tolist())
        success_rate.append(np.average(recomended_treatment == best_treatment))

        print('Horizon:', T,
              'Success Rate:', np.average(recomended_treatment == best_treatment),
              'Pass:', np.average(pass_or_not),
              'Error:', np.average(error))

    success_result_MOOSE = result.copy()
    success_rate_MOOSE = success_rate.copy()
    pass_result_MOOSE = val_pass.copy()
    pass_rate_MOOSE = val_pass_rate.copy()
    error_result_MOOSE = type_one_error.copy()
    error_rate_MOOSE = error_rate.copy()

    with open('Data_128/MOOSE.json', 'w') as f:
        json.dump([success_result_MOOSE, pass_result_MOOSE, error_result_MOOSE], f)
#%% MOOSE Unknown Variance
if 1:
    print('MOOSE Unknown Variance')
    result = []
    success_rate = []
    type_one_error = []
    error_rate = []
    val_pass = []
    val_pass_rate = []
    for T in T_set:
        num_of_batches = int(np.floor(np.log2(A)))
        batch_size = int(np.ceil(T / num_of_batches))

        agent = MOOSE_Vec(A, M, T, std, xi, num_of_iter)
        agent.traffic_allocation = np.ones((A + 1, num_of_iter)) / agent.action_size

        degree_of_freedom = int(T / (np.log2(A) * A)) - 1
        samples = np.random.chisquare(df=degree_of_freedom, size=(A+1, M, num_of_iter))
        samples = samples / degree_of_freedom * (np.tile(std[:, :, np.newaxis], (1, 1, num_of_iter)) ** 2)

        agent.update_std_est(np.sqrt(samples))
        agent.update_z_est()
        agent.update_traffic_allocation()

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
        recomended_treatment = agent.recommend_treatment()
        recomended_z = z_arm[recomended_treatment - 1]

        prob_of_pass = stats.norm.cdf(recomended_z * np.sqrt(Tv / 2))
        pass_or_not = np.random.binomial(1, prob_of_pass)
        error = pass_or_not * (recomended_z < 0)

        error_rate.append(np.average(error))
        type_one_error.append(error.tolist())

        val_pass_rate.append(np.average(pass_or_not))
        val_pass.append(pass_or_not.tolist())

        result.append((recomended_treatment == best_treatment).tolist())
        success_rate.append(np.average(recomended_treatment == best_treatment))

        print('Horizon:', T,
              'Success Rate:', np.average(recomended_treatment == best_treatment),
              'Pass:', np.average(pass_or_not),
              'Error:', np.average(error))

    success_result_MOOSE_UN = result.copy()
    success_rate_MOOSE_UN = success_rate.copy()
    pass_result_MOOSE_UN = val_pass.copy()
    pass_rate_MOOSE_UN = val_pass_rate.copy()
    error_result_MOOSE_UN = type_one_error.copy()
    error_rate_MOOSE_UN = error_rate.copy()

    with open('Data_128/MOOSE_Unknown.json', 'w') as f:
        json.dump([success_result_MOOSE_UN, pass_result_MOOSE_UN, error_result_MOOSE_UN], f)

#%% SHSE
if 1:
    print('SHSE')
    result = []
    success_rate = []
    type_one_error = []
    error_rate = []
    val_pass = []
    val_pass_rate = []
    for T in T_set:
        agent = SHSE_Vec(A, M, T, std, xi, num_of_iter)
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

            new_arm_stats[1:, :, :] = new_arm_stats[1:, :, :] * np.tile(agent.active_treatments[:, np.newaxis, :],(1, M, 1))

            agent.batch_sample_update(new_arm_stats, new_arm_pulls)
        recomended_treatment = agent.recommend_treatment()
        recomended_z = z_arm[recomended_treatment - 1]

        prob_of_pass = stats.norm.cdf(recomended_z * np.sqrt(Tv / 2))
        pass_or_not = np.random.binomial(1, prob_of_pass)
        error = pass_or_not * (recomended_z < 0)

        error_rate.append(np.average(error))
        type_one_error.append(error.tolist())

        val_pass_rate.append(np.average(pass_or_not))
        val_pass.append(pass_or_not.tolist())

        result.append((recomended_treatment == best_treatment).tolist())
        success_rate.append(np.average(recomended_treatment == best_treatment))

        print('Horizon:', T,
              'Success Rate:', np.average(recomended_treatment == best_treatment),
              'Pass:', np.average(pass_or_not),
              'Error:', np.average(error))

    success_result_SHSE = result.copy()
    success_rate_SHSE = success_rate.copy()
    pass_result_SHSE = val_pass.copy()
    pass_rate_SHSE = val_pass_rate.copy()
    error_result_SHSE = type_one_error.copy()
    error_rate_SHSE = error_rate.copy()

    with open('Data_128/SHSE.json', 'w') as f:
        json.dump([success_result_SHSE, pass_result_SHSE, error_result_SHSE], f)
#%% SHVarSE
if 1:
    print('SHVarSE')
    result = []
    success_rate = []
    type_one_error = []
    error_rate = []
    val_pass = []
    val_pass_rate = []
    for T in T_set:
        agent = SHVarSE_Vec(A, M, T, std, xi, num_of_iter)
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
        recomended_treatment = agent.recommend_treatment()
        recomended_z = z_arm[recomended_treatment - 1]

        prob_of_pass = stats.norm.cdf(recomended_z * np.sqrt(Tv / 2))
        pass_or_not = np.random.binomial(1, prob_of_pass)
        error = pass_or_not * (recomended_z < 0)

        error_rate.append(np.average(error))
        type_one_error.append(error.tolist())

        val_pass_rate.append(np.average(pass_or_not))
        val_pass.append(pass_or_not.tolist())

        result.append((recomended_treatment == best_treatment).tolist())
        success_rate.append(np.average(recomended_treatment == best_treatment))

        print('Horizon:', T,
              'Success Rate:', np.average(recomended_treatment == best_treatment),
              'Pass:', np.average(pass_or_not),
              'Error:', np.average(error))

    success_result_SHVarSE = result.copy()
    success_rate_SHVarSE = success_rate.copy()
    pass_result_SHVarSE = val_pass.copy()
    pass_rate_SHVarSE = val_pass_rate.copy()
    error_result_SHVarSE = type_one_error.copy()
    error_rate_SHVarSE = error_rate.copy()

    with open('Data_128/SHVarSE.json', 'w') as f:
        json.dump([success_result_SHVarSE, pass_result_SHVarSE, error_result_SHVarSE], f)

#%% SHStdSE
if 1:
    print('SHStdSE')
    result = []
    success_rate = []
    type_one_error = []
    error_rate = []
    val_pass = []
    val_pass_rate = []
    for T in T_set:
        agent = SHStdSE_Vec(A, M, T, std, xi, num_of_iter)
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
        recomended_treatment = agent.recommend_treatment()
        recomended_z = z_arm[recomended_treatment - 1]

        prob_of_pass = stats.norm.cdf(recomended_z * np.sqrt(Tv / 2))
        pass_or_not = np.random.binomial(1, prob_of_pass)
        error = pass_or_not * (recomended_z < 0)

        error_rate.append(np.average(error))
        type_one_error.append(error.tolist())

        val_pass_rate.append(np.average(pass_or_not))
        val_pass.append(pass_or_not.tolist())

        result.append((recomended_treatment == best_treatment).tolist())
        success_rate.append(np.average(recomended_treatment == best_treatment))

        print('Horizon:', T,
              'Success Rate:', np.average(recomended_treatment == best_treatment),
              'Pass:', np.average(pass_or_not),
              'Error:', np.average(error))

    success_result_SHStdSE = result.copy()
    success_rate_SHStdSE = success_rate.copy()
    pass_result_SHStdSE = val_pass.copy()
    pass_rate_SHStdSE = val_pass_rate.copy()
    error_result_SHStdSE = type_one_error.copy()
    error_rate_SHStdSE = error_rate.copy()

    with open('Data_128/SHStdSE.json', 'w') as f:
        json.dump([success_result_SHStdSE, pass_result_SHStdSE, error_result_SHStdSE], f)

#%%
plt.plot(T_set, success_rate_MOOSE, label = 'MOO-single')
plt.plot(T_set, success_rate_MOOSE_UN, label = 'MOO-unknown')
plt.plot(T_set, success_rate_SHStdSE, label = 'Neyman')
plt.plot(T_set, success_rate_SHSE, label = 'SHSE')
plt.plot(T_set, success_rate_SHVarSE, label = 'SHVarSE')
plt.legend()
plt.xlabel('T')
plt.ylabel('Success Rate')
plt.show()

print()
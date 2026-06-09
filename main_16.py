import numpy as np
import matplotlib.pyplot as plt
from Algorithms.MOOSE import MOOSE_Vec
from Algorithms.MOO import MOO_Vec
from Algorithms.SH import SH_Vec
from Algorithms.SHVar import SHVar_Vec
from Algorithms.SHSE import SHSE_Vec
from Algorithms.SHVarSE import SHVarSE_Vec
import json
#%%
if 1:
    A = 16
    M = 3
    num_of_iter = 100000
    np.random.seed(0)

    mean = np.zeros((A + 1, M))
    std = np.zeros((A + 1, M))
    xi = np.zeros((A, M))
    z = np.zeros((A, M))
    z[0, :] = np.array([1.1, 1.1, 1])
    for a in range(1, A):
        z[a,:] = np.array([0.9, 1.2, 1.2])

    rho_square = np.zeros((A, M))
    rho_square[0, :] = np.array([0.8, 0.5, 0.2])
    for a in range(1, A):
        rho_square[a,:] = np.array([0.8, 0.5, 0.2])

    std[0, :] = np.array([1, 1, 1])
    std[1:, :] = np.sqrt(rho_square[:, :] / (1 - rho_square[:, :]) * np.tile((std[0, :] ** 2), (A, 1)))

    mean[0, :] = np.array([0, 0, 0])
    mean[1:, :] = (z - xi) * np.sqrt(std[1:, :] ** 2 + std[0, :] ** 2)

    z_arm = np.min(z, axis=1)
    metric_arm = np.argmin(z, axis=1)

    best_treatment = np.argmax(np.min(z, axis=1)) + 1

    num_set = np.linspace(5, 14, 10)

    T_set = np.ceil(num_set * 1e3).tolist()

    data = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'z': z.tolist(),
        'rho_square': rho_square.tolist(),
        'xi': xi.tolist(),
        'best_treatment': int(best_treatment),
        'T': T_set,
    }
    with open('./Data/setup.json', 'w') as f:
        json.dump(data, f)
#%%
if 1:
    print('MOOSE')
    result = []
    success_rate = []
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

        result.append((recomended_treatment == best_treatment).tolist())
        print('Horizon:', T, 'Success Rate:', np.average(recomended_treatment == best_treatment))
        success_rate.append(np.average(recomended_treatment == best_treatment))

    success_result_MOOSE = result.copy()
    success_rate_MOOSE = success_rate.copy()

    with open('Data/MOOSE.json', 'w') as f:
        json.dump(success_result_MOOSE, f)
#%% MOO
if 1:
    print('MOO')
    result = []
    success_rate = []
    for T in T_set:
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
        recomended_treatment = agent.recommend_treatment()

        result.append((recomended_treatment == best_treatment).tolist())
        print('Horizon:', T, 'Success Rate:', np.average(recomended_treatment == best_treatment))
        success_rate.append(np.average(recomended_treatment == best_treatment))

    success_result_MOO = result.copy()
    success_rate_MOO = success_rate.copy()

    with open('Data/MOO.json', 'w') as f:
        json.dump(success_result_MOO, f)
#%% SH
if 1:
    print('SH')
    result = []
    success_rate = []
    for T in T_set:
        agent = SH_Vec(A, M, T, std, xi, num_of_iter)
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

        result.append((recomended_treatment == best_treatment).tolist())
        print('Horizon:', T, 'Success Rate:', np.average(recomended_treatment == best_treatment))
        success_rate.append(np.average(recomended_treatment == best_treatment))

    success_result_SH = result.copy()
    success_rate_SH = success_rate.copy()

    with open('Data/SH.json', 'w') as f:
        json.dump(success_result_SH, f)
#%% SHSE
if 1:
    print('SHSE')
    result = []
    success_rate = []
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

        result.append((recomended_treatment == best_treatment).tolist())
        print('Horizon:', T, 'Success Rate:', np.average(recomended_treatment == best_treatment))
        success_rate.append(np.average(recomended_treatment == best_treatment))

    success_result_SHSE = result.copy()
    success_rate_SHSE = success_rate.copy()

    with open('Data/SHSE.json', 'w') as f:
        json.dump(success_result_SHSE, f)
#%% SHVar
if 1:
    print('SHVar')
    result = []
    success_rate = []
    for T in T_set:
        agent = SHVar_Vec(A, M, T, std, xi, num_of_iter)
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

        result.append((recomended_treatment == best_treatment).tolist())
        print('Horizon:', T, 'Success Rate:', np.average(recomended_treatment == best_treatment))
        success_rate.append(np.average(recomended_treatment == best_treatment))

    success_result_SHVar = result.copy()
    success_rate_SHVar = success_rate.copy()

    with open('Data/SHVar.json', 'w') as f:
        json.dump(success_result_SHVar, f)
#%% SHVarSE
if 1:
    print('SHVarSE')
    result = []
    success_rate = []
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

        result.append((recomended_treatment == best_treatment).tolist())
        print('Horizon:', T, 'Success Rate:', np.average(recomended_treatment == best_treatment))
        success_rate.append(np.average(recomended_treatment == best_treatment))

    success_result_SHVarSE = result.copy()
    success_rate_SHVarSE = success_rate.copy()

    with open('Data/SHVarSE.json', 'w') as f:
        json.dump(success_result_SHVarSE, f)
#%%
plt.plot(T_set, success_rate_MOOSE, label = 'MOO-single')
# plt.plot(T_set, success_rate_MOODE, label = 'MOO-double')
plt.plot(T_set, success_rate_MOO, label = 'MOO')
plt.plot(T_set, success_rate_SH, label = 'SH')
plt.plot(T_set, success_rate_SHSE, label = 'SHSE')
plt.plot(T_set, success_rate_SHVar, label = 'SHVar')
plt.plot(T_set, success_rate_SHVarSE, label = 'SHVarSE')
plt.legend()
plt.xlabel('T')
plt.ylabel('Success Rate')
plt.show()

print()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import adaptability_nonexploitability as laff
import numpy as np
import nashpy as nash
import matplotlib.pyplot as plt
import os
import pandas as pd


def read_round_rob(algo_list, game_name, start_num, end_num, expt_id):
    '''
    Computes summary data files from raw reward lists
    '''
    base_1, base_2 = laff.game_map[game_name]
    symmetric = np.all(base_1 == base_2.transpose())
    result_dict = {}
    result_array = np.zeros((end_num - start_num + 1, 2, len(algo_list), len(algo_list)))
    x = 'barg_results_pre/' + game_name + '/'
    for i in range(len(algo_list)):
        alg1 = algo_list[i]
        name1 = alg1(0, 0, None, None, name=True)
        result_dict[name1] = {}
        for j in range(len(algo_list)): 
            alg2 = algo_list[j]
            name2 = alg2(0, 0, None, None, name=True)
            results = [[], []]
            y = name1 + '/' + name2 + '/'
            if symmetric and j < i:
                res = result_dict[name2][name1]
                results = np.array([res[1], res[0]])
            else:
                for k in range(start_num, end_num + 1):
                    rew1 = np.array(pd.read_csv(x + y + 'P1/' + expt_id + '/' + str(k) + '.txt',
                                                header=None, squeeze=True))
                    rew2 = np.array(pd.read_csv(x + y + 'P2/' + expt_id + '/' + str(k) + '.txt',
                                                header=None, squeeze=True))
                    results[0].append(rew1)
                    results[1].append(rew2)
            result_dict[name1][name2] = results
            result_array[:,0,i,j] = np.mean(np.array(results[0]), axis=1)
            result_array[:,1,j,i] = np.mean(np.array(results[1]), axis=1)
    for n in range(end_num - start_num + 1):
        np.savetxt('../summaries/' + game_name + 'summary_P1_' + str(n + start_num) + '.txt', result_array[n,0])
        np.savetxt('../summaries/' + game_name + 'summary_P2_' + str(n + start_num) + '.txt', result_array[n,1])
    return result_dict, result_array


def read_exp_summaries(algo_list, start_num, end_num, num_gens=500, num_reps=100):
    '''
    From summary matrices, computes empirical game results and replicator dynamic
    '''
    roundrobins_arr = []
    J = len(algo_list)
    G = len(laff.game_map)
    init_pop = np.ones(J)/J
    num_rounds = end_num - start_num + 1
    for name in laff.game_map.keys():
        result_array = np.zeros((num_rounds, 2, len(algo_list), len(algo_list)))
        for n in range(num_rounds):
            result_array[n,0] = np.array(pd.read_csv('summaries/' + name + 'summary_P1_' + str(n + start_num) + '.txt', header=None, sep=' '))
            result_array[n,1] = np.array(pd.read_csv('summaries/' + name + 'summary_P2_' + str(n + start_num) + '.txt', header=None, sep=' '))
        roundrobins_arr.append(result_array)

    population_min = np.tile(init_pop, (num_reps, num_gens, 1))
    for i in range(num_gens-1):
        tot_min = np.zeros((num_reps, J, J))
        for j in range(G):
            rr_arr = roundrobins_arr[j][np.random.choice(num_rounds, num_reps)]
            tot_min += np.min(rr_arr, axis=1)
        tot_min /= G
        fit_min = np.einsum('ijk,ik->ij', tot_min, population_min[:,i,:])
        phi_min = np.einsum('ij,ij->i', fit_min, population_min[:,i,:])
        population_min[:,i+1,:] = population_min[:,i,:]*(1 + fit_min - phi_min[:,None])
    bim = np.mean(np.mean(np.array(roundrobins_arr), axis=0), axis=0)
    g1 = bim[0]
    g2 = bim[1].T
    emp_game = nash.Game(g1, g2)
    nes = list(emp_game.support_enumeration())
    mean_pop_min = np.mean(population_min, axis=0)
    sd_pop_min = np.std(population_min, axis=0)
    for j in range(J):
        plt.plot(mean_pop_min[:,j])
        plt.fill_between(np.arange(num_gens),
          mean_pop_min[:,j] - sd_pop_min[:,j], mean_pop_min[:,j] + sd_pop_min[:,j],
          alpha=0.2)
    plt.ylim(top=1, bottom=0)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Population Share', fontsize=16)
    new_names = ['S++', 'Manipulator', 'M-Qubed', 'Bully', 'Q-Learning',
                 'LAFF', 'FTFT', 'FP']
    plt.legend(new_names, loc='upper left', prop={'size': 9}, frameon=False)
    plt.savefig('rep.png', bbox_inches='tight')
    plt.show()
    return population_min, roundrobins_arr, emp_game, nes



def three_values(base_1, base_2):
    '''
    EBS, Bully values, and maximin values for given game
    '''
    rg = laff.RepeatedGame(base_1, base_2, int(1e5), eps=0.05, K=1)
    return rg.vebs, rg.bul, rg.vsec, rg.probs_ebs, rg.probs_bul



def gtft_mu(base_1, base_2, p_ix=0):
    '''
    Computes the mu* of player p_ix against a Generous Godfather
    '''
    rg = laff.RepeatedGame(base_1, base_2, int(1e5), eps=0.05, K=1)
    rg.tPs[(1, rg.probs_ebs[1][0])] = laff.transition(rg.S, rg.A, rg.geom_states,
        rg.K, (rg.probs_ebs[1][0], 1-rg.probs_ebs[1][0]), (rg.probs_ebs[1][0], 1-rg.probs_ebs[1][0]))
    rg.tPs[(rg.probs_ebs[0][0], 1)] = laff.transition(rg.S, rg.A, rg.geom_states,
        rg.K, (rg.probs_ebs[0][0], 1-rg.probs_ebs[0][0]), (rg.probs_ebs[0][0], 1-rg.probs_ebs[0][0]))

    tP = rg.tPs[(rg.probs_ebs[0][0], 1)] if p_ix else rg.tPs[(1, rg.probs_ebs[1][0])]
    pol2 = rg.generous[1-p_ix]
    S=rg.S
    A=rg.A
    rewards_1=rg.rewards_1
    rewards_2=rg.rewards_2
    pol1 = laff.bestresp(pol2, tP, S, A, rewards_1, rewards_2, base_1, base_2, delt=1e-5,
            p_ix=p_ix)
    return laff.emp_avg_return(pol2 if p_ix else pol1, rewards_2 if p_ix else rewards_1, pol1 if p_ix else pol2, int(1e5), tP, 0, S, A)


def plot_regret(rewards, baseline):
    plt.plot(np.cumsum(np.array(baseline) - np.array(rewards)))


def regret_results(result_directories, tee=int(2e5)):
    '''
    * LAFF vs LAFF (Cond Follow)
    * LAFF vs Eps-Q (Uncond Follow)
    * LAFF vs Gen Godfather (BM)
    * LAFF vs Bul Godf (Expl BM)
    * LAFF vs Manipulator (Adv)
    '''
    uncon_baselines = {}
    con_baselines = {}
    bm_baselines = {}
    expl_baselines = {}
    safe_baselines = {}
    regs = {}
    for game_name, mats in laff.game_map.items():
        values = three_values(mats[0], mats[1])
        follow_val = gtft_mu(mats[0], mats[1])
        uncon_baselines[game_name] = values[1][1]
        con_baselines[game_name] = values[0][0]
        safe_baselines[game_name] = values[2][1]
        expl_baselines[game_name] = values[0][0]
        bm_baselines[game_name] = follow_val
    official_names = ['Unfair S', 'Unfair A', 'Win-Win S',
                      'Win-Win A', 'Biased S', 'Biased A',
                      'Second-Best S', 'Second-Best A',
                      'Inferior S', 'Inferior A',
                      'Cyclic A']
    bul_names = ['Unfair S', 'Unfair A', 'Biased S', 'Biased A',
                      'Inferior S', 'Inferior A',
                      'Cyclic A']

    codes = ['/Eps-Greedy Q/LAFF/P2/', '/LAFF/LAFF/P1/',
             '/LAFF/Generous Godfather/P1/', '/Bully Manipulator/LAFF/P2/',
             '/Bully Godfather/LAFF/P1/']

    j=0
    for code, dct in zip(codes, [uncon_baselines, con_baselines,
      bm_baselines, safe_baselines, expl_baselines]):
        j += 1
        for game_name in laff.game_map.keys():
            if code != '/Bully Godfather/LAFF/P1/' or game_name not in ['ses', 'sea', 'wws', 'wwa']:
                rew_array = np.zeros(tee)
                ct = 0
                for direc in result_directories:
                    path = direc + '/barg_results_pre/' + game_name + code + 'test_B'
                    for file in os.listdir(path):
                        rew_array += np.array(pd.read_csv(path + '/' + file,
                                    header=None, squeeze=True))
                        ct += 1
                rew_array /= ct
                plot_regret(rew_array, dct[game_name])
        if code == '/Eps-Greedy Q/LAFF/P2/':
            plt.legend(official_names if code != '/Bully Godfather/LAFF/P1/' else bul_names,
              prop={'size': 9}, frameon=False)
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('P1 Regret' if code != '/Bully Godfather/LAFF/P1/' else 'P2 Regret', fontsize=16)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.savefig(str(j) + '.png', bbox_inches='tight')
        plt.show()

    '''
    NOTE: [ses, sea, wws, wwa] either have Maximin value = EBS for P1, or have Bully = EBS,
           so we omit those from non-expl
    '''

    '''
    1) For each game in the game_list, do num_rounds iterations of
    the round_robin tournament among algo_list.
      * Can present these as standalone results, seeing each algo's
        average-case + worst-case performance
      * Can compute the algos' rewards against their best-responses,
        and an EGTA Nash eq of the learning game
    2) Compute replicator dynamic based on samples bootstrapped from
    the round robin results
    '''


# %%

if __name__ == '__main__':
    dirs = []
    done = False
    while not done:
        dir_name = input('Name of next results directory you want to add: ')
        dirs.append(dir_name)
        done = bool(input('Done adding results directories? Enter nothing if not done: '))
    # Generate summary stats
    for dir in dirs:
        os.chdir(dir)
        for name in laff.game_names:
            read_round_rob(laff.final_algo_list, name, 0, 49,
              'test_B')
        os.chdir('..')
    # Generate Fig. 4
    np.random.seed(1)
    population_min, roundrobins_arr, emp_game, nes = read_exp_summaries(laff.final_algo_list, 0, 49, num_reps=1000)
    '''
    Need to be in same directory as the "summaries" subdirectory
    '''
    # Generate Fig. 3
    regret_results(dirs)

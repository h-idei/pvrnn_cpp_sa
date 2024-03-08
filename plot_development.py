import numpy as np
import os.path
import sys
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from optparse import OptionParser
import glob
SEED_NUM = 1
Q_size_sen = 1
Q_size_ass = 3
T_MIN = 0
T_MAX = 200
SEQ_NUM = 48
EACH_SELF_OTHER = 24
FOLDERNAME_GENERATE = "learning_generation"
FOLDERNAME_MODEL = "learning_model"
EPOCH_SAVE_POINT = 5000
EPOCH_POINT_NUM = 40
if __name__ == '__main__':
    
    sen_each_seed_final_epoch = np.zeros((SEED_NUM, 2)) #total_self-produced, total_externally-produced
    sen_each_seed_final_epoch_prior_sigma = np.zeros((SEED_NUM, 2)) 
    
    prop_epoch_change_self = np.zeros((EPOCH_POINT_NUM, 1))
    prop_epoch_change_others = np.zeros((EPOCH_POINT_NUM, 1))
    extero_epoch_change_self = np.zeros((EPOCH_POINT_NUM, 1))
    extero_epoch_change_others = np.zeros((EPOCH_POINT_NUM, 1))
    
    prop_epoch_sigma_self = np.zeros((EPOCH_POINT_NUM, 1))
    prop_epoch_sigma_others = np.zeros((EPOCH_POINT_NUM, 1))
    extero_epoch_sigma_self = np.zeros((EPOCH_POINT_NUM, 1))
    extero_epoch_sigma_others = np.zeros((EPOCH_POINT_NUM, 1))
    
    
    for seed in range(SEED_NUM):
        for k in range(EPOCH_POINT_NUM): #k: epoch
            epoch = k*EPOCH_SAVE_POINT
            prop_epoch_change_seq = np.zeros(SEQ_NUM)
            extero_epoch_change_seq = np.zeros(SEQ_NUM)
            
            prop_epoch_sigma_seq = np.zeros(SEQ_NUM)
            extero_epoch_sigma_seq = np.zeros(SEQ_NUM)
            
            for i in range(SEQ_NUM):
                #proprioceptive area
                filename_proprioceptive_in_p_mu = "./" + FOLDERNAME_GENERATE + "/proprioceptive" + "/in_p_mu_{:0>7}".format(i) + "_{:0>7}".format(epoch) + ".txt"
                filename_proprioceptive_in_p_sigma = "./" + FOLDERNAME_GENERATE + "/proprioceptive" + "/in_p_sigma_{:0>7}".format(i) + "_{:0>7}".format(epoch) + ".txt"
                filename_proprioceptive_a_mu = "./" + FOLDERNAME_MODEL + "/proprioceptive" + "/a_mu_{:0>7}".format(i) + "_{:0>7}".format(epoch) + ".txt"
                filename_proprioceptive_a_sigma = "./" + FOLDERNAME_MODEL + "/proprioceptive" + "/a_sigma_{:0>7}".format(i) + "_{:0>7}".format(epoch) + ".txt"
                #exteroceptive area
                filename_exteroceptive_in_p_mu = "./" + FOLDERNAME_GENERATE + "/exteroceptive" + "/in_p_mu_{:0>7}".format(i) + "_{:0>7}".format(epoch) + ".txt"
                filename_exteroceptive_in_p_sigma = "./" + FOLDERNAME_GENERATE + "/exteroceptive" + "/in_p_sigma_{:0>7}".format(i) + "_{:0>7}".format(epoch) + ".txt"
                filename_exteroceptive_a_mu = "./" + FOLDERNAME_MODEL + "/exteroceptive" + "/a_mu_{:0>7}".format(i) + "_{:0>7}".format(epoch) + ".txt"
                filename_exteroceptive_a_sigma = "./" + FOLDERNAME_MODEL + "/exteroceptive" + "/a_sigma_{:0>7}".format(i) + "_{:0>7}".format(epoch) + ".txt"
                
                proprioceptive_in_p_mu = np.loadtxt(filename_proprioceptive_in_p_mu)
                proprioceptive_in_p_sigma = np.loadtxt(filename_proprioceptive_in_p_sigma)
                proprioceptive_a_mu = np.loadtxt(filename_proprioceptive_a_mu)
                proprioceptive_a_sigma = np.loadtxt(filename_proprioceptive_a_sigma)
                
                exteroceptive_in_p_mu = np.loadtxt(filename_exteroceptive_in_p_mu)
                exteroceptive_in_p_sigma = np.loadtxt(filename_exteroceptive_in_p_sigma)
                exteroceptive_a_mu = np.loadtxt(filename_exteroceptive_a_mu)
                exteroceptive_a_sigma = np.loadtxt(filename_exteroceptive_a_sigma)
                
                prop_q_mu = np.tanh(proprioceptive_a_mu)
                prop_p_sigma = np.exp(proprioceptive_in_p_sigma)
                extero_q_mu = np.tanh(exteroceptive_a_mu)
                extero_p_sigma = np.exp(exteroceptive_in_p_sigma)
                
                prop_change = 0
                extero_change = 0
                
                for t in range(T_MAX-1):
                    prop_change += np.sqrt( np.sum ( (prop_q_mu[t+1] - prop_q_mu[t]) ** 2)) / T_MAX / Q_size_sen
                    extero_change += np.sqrt( np.sum ( (extero_q_mu[t+1] - extero_q_mu[t]) ** 2)) / T_MAX / Q_size_sen
                
                prop_epoch_change_seq[i] = prop_change #/ SEQ_NUM
                extero_epoch_change_seq[i] = extero_change #/ SEQ_NUM
                
                prop_epoch_sigma_seq[i] = np.average(prop_p_sigma)
                extero_epoch_sigma_seq[i] = np.average(extero_p_sigma)
                
            #average over all seeds
            prop_epoch_change_self[k] += np.average(prop_epoch_change_seq[:EACH_SELF_OTHER]) / SEED_NUM
            prop_epoch_change_others[k] += np.average(prop_epoch_change_seq[EACH_SELF_OTHER:]) / SEED_NUM
            extero_epoch_change_self[k] += np.average(extero_epoch_change_seq[:EACH_SELF_OTHER]) / SEED_NUM
            extero_epoch_change_others[k] += np.average(extero_epoch_change_seq[EACH_SELF_OTHER:]) / SEED_NUM
            
            prop_epoch_sigma_self[k] += np.average(prop_epoch_sigma_seq[:EACH_SELF_OTHER]) / SEED_NUM
            prop_epoch_sigma_others[k] += np.average(prop_epoch_sigma_seq[EACH_SELF_OTHER:]) / SEED_NUM
            extero_epoch_sigma_self[k] += np.average(extero_epoch_sigma_seq[:EACH_SELF_OTHER]) / SEED_NUM
            extero_epoch_sigma_others[k] += np.average(extero_epoch_sigma_seq[EACH_SELF_OTHER:]) / SEED_NUM
            
            #total response in each seed
            if k == EPOCH_POINT_NUM - 1:
                #self-produced condition
                sen_each_seed_final_epoch[seed, 0] = np.average(prop_epoch_change_seq[:EACH_SELF_OTHER] + extero_epoch_change_seq[:EACH_SELF_OTHER])
                sen_each_seed_final_epoch_prior_sigma[seed, 0] = np.average(prop_epoch_sigma_seq[:EACH_SELF_OTHER] + extero_epoch_sigma_seq[:EACH_SELF_OTHER])
                #externally produced condition
                sen_each_seed_final_epoch[seed, 1] = np.average(prop_epoch_change_seq[EACH_SELF_OTHER:] + extero_epoch_change_seq[EACH_SELF_OTHER:])
                sen_each_seed_final_epoch_prior_sigma[seed, 1] = np.average(prop_epoch_sigma_seq[EACH_SELF_OTHER:] + extero_epoch_sigma_seq[EACH_SELF_OTHER:])

    fig = plt.figure(figsize=(6.4, 4.8), dpi=300, facecolor='w', linewidth=0, edgecolor='w')
    ax = fig.add_subplot(111)
    ax.plot(extero_epoch_change_self, color='r', linestyle='solid')
    ax.plot(prop_epoch_change_self, color='r', linestyle='dotted')
    ax.plot(extero_epoch_change_others, color='b', linestyle='solid')
    ax.plot(prop_epoch_change_others, color='b', linestyle='dotted')
    ax.set_xlim((0, 40))
    ax.set_xticks((0, 10, 20, 30, 40))
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.5, 1.0))
    filename_fig = "./development_sensory_posterior_response.pdf"
    fig.savefig(filename_fig, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    
    
    fig = plt.figure(figsize=(6.4, 4.8), dpi=300, facecolor='w', linewidth=0, edgecolor='w')
    ax = fig.add_subplot(111)
    ax.plot(extero_epoch_sigma_self, color='r', linestyle='solid')
    ax.plot(prop_epoch_sigma_self, color='r', linestyle='dotted')
    ax.plot(extero_epoch_sigma_others, color='b', linestyle='solid')
    ax.plot(prop_epoch_sigma_others, color='b', linestyle='dotted')
    ax.set_xlim((0, 40))
    ax.set_xticks((0, 10, 20, 30, 40))
    ax.set_ylim((0, 2.0))
    ax.set_yticks((0, 1.0, 2.0))
    filename_fig = "./development_sensory_prior_sigma.pdf"
    fig.savefig(filename_fig, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    
    

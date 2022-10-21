#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pylab
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from optparse import OptionParser

SEQUENCE_INDEX = "0"
EPOCH = "0"
T_MIN = 0
T_MAX = 200
C_MAP = plt.cm.seismic
params = {"backend": "pdf",
          "font.family": "Arial",
          "axes.titlesize": 20,
          "axes.labelsize": 20,
          "font.size": 20,
          "legend.fontsize":20,
          "xtick.labelsize": 20,
          "ytick.labelsize": 20,
          "text.usetex": False,
          "savefig.facecolor": "1.0"}
pylab.rcParams.update(params)

def read_parameter(f):
    r = {}
    r["c_state_size"] = re.compile(r"^# c_state_size")
    r["out_state_size"] = re.compile(r"^# out_state_size")
    r_comment = re.compile(r'^#')
    params = {}
    for line in f:
        for k,v in r.iteritems():
            if (v.match(line)):
                x = line.split('=')[1]
                if k == 'target':
                    m = int(v.match(line).group(1))
                    if (k in params):
                        params[k][m] = x
                    else:
                        params[k] = {m:x}
                else:
                    params[k] = x

        if (r_comment.match(line) == None):
            break
    f.seek(0)
    return params

def fk(ms, vs):
        l1, l2, l3 = [0.1, 0.3, 0.5] # lengths of links
        th = (ms + 0.8) * np.pi / 1.6
        tht1 = th[0]
        tht2 = th[1]
        tht3 = th[2]#joint angles
        arm_pos = np.zeros_like(vs)

        #position of link 1
        x1 = - (l1 * np.cos(tht1))
        y1 = l1 * np.sin(tht1)
        
        #position of link 2
        x2 = x1 - (l2 * np.cos((tht1+tht2)))
        y2 = y1 + (l2 * np.sin((tht1+tht2)))
        
        #position of link 3
        x3 = x2 - (l3 * np.cos((tht1+tht2+tht3)))
        y3 = y2 + (l3 * np.sin((tht1+tht2+tht3)))
        
        arm_pos[0] = x3
        arm_pos[1] = y3
        
        #return arm position
        return arm_pos

class plot_rnn(object):
    def __init__(self,
                 filename, filename_executive_a_mu,  filename_executive_a_sigma, filename_associative_in_p_mu, filename_associative_in_p_sigma, filename_associative_a_mu, filename_associative_a_sigma, filename_proprioceptive_in_p_mu, filename_proprioceptive_in_p_sigma,  filename_proprioceptive_a_mu,  filename_proprioceptive_a_sigma, filename_exteroceptive_in_p_mu, filename_exteroceptive_in_p_sigma, filename_exteroceptive_a_mu, filename_exteroceptive_a_sigma, filename_output_proprio, filename_output_extero, filename_target_proprio, filename_target_extero):
        self.figure_name = filename
        self.state_filename_executive_a_mu = filename_executive_a_mu
        self.state_filename_executive_a_sigma = filename_executive_a_sigma
        self.state_filename_associative_in_p_mu = filename_associative_in_p_mu
        self.state_filename_associative_in_p_sigma = filename_associative_in_p_sigma
        self.state_filename_associative_a_mu = filename_associative_a_mu
        self.state_filename_associative_a_sigma = filename_associative_a_sigma
        
        self.state_filename_proprioceptive_in_p_mu = filename_proprioceptive_in_p_mu
        self.state_filename_proprioceptive_in_p_sigma = filename_proprioceptive_in_p_sigma
        self.state_filename_proprioceptive_a_mu = filename_proprioceptive_a_mu
        self.state_filename_proprioceptive_a_sigma = filename_proprioceptive_a_sigma
        
        self.state_filename_exteroceptive_in_p_mu = filename_exteroceptive_in_p_mu
        self.state_filename_exteroceptive_in_p_sigma = filename_exteroceptive_in_p_sigma
        self.state_filename_exteroceptive_a_mu = filename_exteroceptive_a_mu
        self.state_filename_exteroceptive_a_sigma = filename_exteroceptive_a_sigma
        
        self.state_filename_output_proprio = filename_output_proprio
        self.state_filename_output_extero = filename_output_extero
        
        self.state_filename_target_proprio = filename_target_proprio
        self.state_filename_target_extero = filename_target_extero

    def add_info(self, ax, title, xlim, ylim, xlabel, ylabel):
        if title != None:
            ax.set_title(title)

        if xlim != None:
            ax.set_xlim(xlim)
            #ax.set_xticks((xlim[0], (xlim[0] + xlim[1]) / 2.0, xlim[1]))
            ax.set_xticks([0, 100, 200])
        else:
            ax.set_xticks([])

        if xlabel != None:
            ax.set_xlabel(xlabel)

        if ylim != None:
            ax.set_ylim(ylim)
            ax.set_yticks((ylim[0], (ylim[0] + ylim[1]) / 2.0, ylim[1]))
        if ylabel != None:
            ax.set_ylabel(ylabel)

        ax.grid(True) 
        
    def set_no_yticks(self, ax):
        ax.set_yticks([])

    def configure(self, fig_matrix, width, height):
        fig = plt.figure(figsize = (1.5*width * fig_matrix[1], height * fig_matrix[0]))
        gs = gridspec.GridSpec(fig_matrix[0], fig_matrix[1])
        axes = [fig.add_subplot(gs[i, j]) for i in range(fig_matrix[0]) for j in range(fig_matrix[1])]
        return fig, gs, axes

    def plot_colormap(self, ax, state, range):
        im = ax.imshow(state.T, vmin = range[0], vmax = range[1], aspect = "auto", interpolation = "nearest", cmap = C_MAP)
        if state.shape[0] is not 1:
            ax.set_xlim(0, state.shape[0])
            ax.set_ylim(-0.5, state.shape[1] - 0.5)
            ax.set_yticks((0, state.shape[1] -1))
        return im

    def state(self, tmin, tmax, width, height):
        fig_matrix = [8, 1]
        fig, gs, axes = self.configure(fig_matrix, width, height)
        
        executive_a_mu = np.loadtxt(self.state_filename_executive_a_mu)
        executive_a_sigma = np.loadtxt(self.state_filename_executive_a_sigma)
        
        associative_in_p_mu = np.loadtxt(self.state_filename_associative_in_p_mu)
        associative_in_p_sigma = np.loadtxt(self.state_filename_associative_in_p_sigma)
        associative_a_mu = np.loadtxt(self.state_filename_associative_a_mu)
        associative_a_sigma = np.loadtxt(self.state_filename_associative_a_sigma)
        
        proprioceptive_in_p_mu = np.loadtxt(self.state_filename_proprioceptive_in_p_mu)
        proprioceptive_in_p_sigma = np.loadtxt(self.state_filename_proprioceptive_in_p_sigma)
        proprioceptive_a_mu = np.loadtxt(self.state_filename_proprioceptive_a_mu)
        proprioceptive_a_sigma = np.loadtxt(self.state_filename_proprioceptive_a_sigma)
        
        exteroceptive_in_p_mu = np.loadtxt(self.state_filename_exteroceptive_in_p_mu)
        exteroceptive_in_p_sigma = np.loadtxt(self.state_filename_exteroceptive_in_p_sigma)
        exteroceptive_a_mu = np.loadtxt(self.state_filename_exteroceptive_a_mu)
        exteroceptive_a_sigma = np.loadtxt(self.state_filename_exteroceptive_a_sigma)
        
        output_proprio = np.loadtxt(self.state_filename_output_proprio)
        output_hand_pos = np.zeros((output_proprio.shape[0], 2))
        output_extero = np.loadtxt(self.state_filename_output_extero)
        
        target_proprio = np.loadtxt(self.state_filename_target_proprio)
        target_hand_pos = np.zeros((output_proprio.shape[0], 2))
        target_extero = np.loadtxt(self.state_filename_target_extero)
        
        for t in range(output_proprio.shape[0]):
            output_hand_pos[t] = fk(output_proprio[t, :], output_extero[t, :])
            target_hand_pos[t] = fk(target_proprio[t, :], target_extero[t, :])

        axes[0].plot(np.tanh(executive_a_mu), linestyle="solid", linewidth="1")
        self.add_info(axes[0], None, None, (-1.2,1.2), None, "Exe. mu")
        axes[1].plot(np.exp(executive_a_sigma), linestyle="solid", linewidth="1")
        self.add_info(axes[1], None, None, (0.0, 2.0), None, "Exe. sigma")
        
        #Association area
        axes[2].plot(np.tanh(associative_a_mu[:, 0]), linestyle="solid", linewidth="1", color = "c")
        axes[2].plot(np.tanh(associative_a_mu[:, 1]), linestyle="solid", linewidth="1", color = "m")
        axes[2].plot(np.tanh(associative_a_mu[:, 2]), linestyle="solid", linewidth="1", color = "y")
        axes[2].plot(np.tanh(associative_in_p_mu[:, 0]), linestyle="dashed", linewidth="1", color = "c")
        axes[2].plot(np.tanh(associative_in_p_mu[:, 1]), linestyle="dashed", linewidth="1", color = "m")
        axes[2].plot(np.tanh(associative_in_p_mu[:, 2]), linestyle="dashed", linewidth="1", color = "y")
        self.add_info(axes[2], None, None, (-1.2,1.2), None, "Ass. mu")
        
        axes[3].plot(np.exp(associative_a_sigma[:, 0]), linestyle="solid", linewidth="1", color = "c")
        axes[3].plot(np.exp(associative_a_sigma[:, 1]), linestyle="solid", linewidth="1", color = "m")
        axes[3].plot(np.exp(associative_a_sigma[:, 2]), linestyle="solid", linewidth="1", color = "y")
        
        axes[3].plot(np.exp(associative_in_p_sigma[:, 0]), linestyle="dashed", linewidth="1", color = "c")
        axes[3].plot(np.exp(associative_in_p_sigma[:, 1]), linestyle="dashed", linewidth="1", color = "m")
        axes[3].plot(np.exp(associative_in_p_sigma[:, 2]), linestyle="dashed", linewidth="1", color = "y")
        self.add_info(axes[3], None, None, (0.0,2.0), None, "Ass. sigma")

        #Sensory areas
        axes[4].plot(np.tanh(exteroceptive_a_mu), linestyle="solid", linewidth="1", color="r")
        axes[4].plot(np.tanh(proprioceptive_a_mu), linestyle="solid", linewidth="1", color="b")
        axes[4].plot(np.tanh(exteroceptive_in_p_mu), linestyle="dashed", linewidth="1", color="r")
        axes[4].plot(np.tanh(proprioceptive_in_p_mu), linestyle="dashed", linewidth="1", color="b")
        self.add_info(axes[4], None, None, (-1.2,1.2), None, "Sen. mu")
        
        axes[5].plot(np.exp(exteroceptive_a_sigma), linestyle="solid", linewidth="1", color="r")
        axes[5].plot(np.exp(proprioceptive_a_sigma), linestyle="solid", linewidth="1", color="b")
        axes[5].plot(np.exp(exteroceptive_in_p_sigma), linestyle="dashed", linewidth="1", color="r")
        axes[5].plot(np.exp(proprioceptive_in_p_sigma), linestyle="dashed", linewidth="1", color="b")
        self.add_info(axes[5], None, None, (0.0,2.0), None, "Sen. sigma")
       
        #output
        axes[6].plot(output_extero, linestyle="solid", linewidth="1", color="r")
        axes[6].plot(output_hand_pos, linestyle="solid", linewidth="1", color="b")
        self.add_info(axes[6], None, None, (-1.2,1.2), None, "Out.")
        
        #target
        axes[7].plot(target_extero, linestyle="solid", linewidth="1", color="r") #extero
        axes[7].plot(target_hand_pos, linestyle="solid", linewidth="1", color="b") #prop
        self.add_info(axes[7], None, (0,200), (-1.2,1.2), "Time", "Real")
        
        for ax in axes:
            ax.set_xlim(tmin, tmax)
            
        fig.savefig(self.figure_name, format="pdf",dpi=600)
        fig.show()

def main():
    parser = OptionParser()
    parser.add_option("-s", "--sequence", dest="sequence",
                      help="sequence index", metavar=SEQUENCE_INDEX, default=SEQUENCE_INDEX)
    parser.add_option("-e", "--epoch", dest="epoch",
                      help="epoch", metavar="EPOCH", default= EPOCH)
    (options, args) = parser.parse_args()
    
    foldername_list_window = sorted(glob.glob("./test_generation/window_*/"))
    print(foldername_list_window)
    for w, foldername_window in enumerate(foldername_list_window):
        foldername_list_epoch = sorted(glob.glob(foldername_window + "epoch_*/"))
        print(foldername_list_epoch)
        for e, foldername_epoch in enumerate(foldername_list_epoch):
            foldername_list_lr = sorted(glob.glob(foldername_epoch + "lr_*/"))
            print(foldername_list_lr)
            for l, foldername_lr in enumerate(foldername_list_lr):
                #executive area
                filename_executive_a_mu = foldername_lr + "executive" + "/a_mu_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_executive_a_sigma = foldername_lr + "executive" + "/a_sigma_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                #associative area
                filename_associative_in_p_mu = foldername_lr + "associative" + "/in_p_mu_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_associative_in_p_sigma = foldername_lr + "associative" + "/in_p_sigma_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_associative_a_mu = foldername_lr + "associative" + "/a_mu_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_associative_a_sigma = foldername_lr + "associative" + "/a_sigma_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                #proprioceptive area
                filename_proprioceptive_in_p_mu = foldername_lr + "proprioceptive" + "/in_p_mu_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_proprioceptive_in_p_sigma = foldername_lr + "proprioceptive" + "/in_p_sigma_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_proprioceptive_a_mu = foldername_lr + "proprioceptive" + "/a_mu_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_proprioceptive_a_sigma = foldername_lr + "proprioceptive" + "/a_sigma_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                #exteroceptive area
                filename_exteroceptive_in_p_mu = foldername_lr + "exteroceptive" + "/in_p_mu_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_exteroceptive_in_p_sigma = foldername_lr + "exteroceptive" + "/in_p_sigma_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_exteroceptive_a_mu = foldername_lr + "exteroceptive" + "/a_mu_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_exteroceptive_a_sigma = foldername_lr + "exteroceptive" + "/a_sigma_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                #output
                filename_output_proprio = foldername_lr + "out_proprio" + "/output_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_output_extero = foldername_lr + "out_extero" + "/output_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                #target
                filename_target_proprio = foldername_lr + "out_proprio" + "/target_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_target_extero = foldername_lr + "out_extero" + "/target_{:0>7}".format(options.sequence) + "_{:0>7}".format(options.epoch) + ".txt"
                filename_fig = foldername_lr + "/generate.pdf"
                
                plot = plot_rnn(filename_fig, filename_executive_a_mu,  filename_executive_a_sigma, filename_associative_in_p_mu, filename_associative_in_p_sigma, filename_associative_a_mu, filename_associative_a_sigma, filename_proprioceptive_in_p_mu, filename_proprioceptive_in_p_sigma,  filename_proprioceptive_a_mu,  filename_proprioceptive_a_sigma, filename_exteroceptive_in_p_mu, filename_exteroceptive_in_p_sigma, filename_exteroceptive_a_mu, filename_exteroceptive_a_sigma, filename_output_proprio, filename_output_extero, filename_target_proprio, filename_target_extero)
                plot.state(T_MIN, T_MAX, 8, 2)


if __name__ == "__main__":
    main()

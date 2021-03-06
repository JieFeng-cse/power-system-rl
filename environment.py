import numpy as np
import gym
from gym import spaces
import os

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import torch
import matplotlib.pyplot as plt
from numpy import array, linalg as LA

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda:0" if use_cuda else "cpu")

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class VoltageCtrl_nonlinear(gym.Env):
    def __init__(self, pp_net, injection_bus, obs_dim=1, action_dim=1, 
                 v0=1, vmax=1.05, vmin=0.95): #1.05, 0.95
        
        self.network =  pp_net
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.action_space = spaces.Box(-100, 100, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-400.0, 400.0, (1,), dtype=np.float32)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])
 
        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        
        self.state = np.ones(self.agentnum, )
        self.last_action = 0.0
    
    def step(self, action):
        
        done = False 
        # action-transition dynamics                
        reward = float(-10*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)        
        action = self.last_action + action
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i+1, 'q_mvar'] = action[i] 
        
        if not self.network['Converged_in_100_r']:
            print("Failed to converge in 100 rounds")
            print(action)
            reward -= 1000
            self.network['Converged_in_100_r'] = True
            done = True
        
        self.last_action = action

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, reward, done, {None:None}
    
    def reset(self, seed=1):
        seed = np.random.randint(0,200)
        # np.random.seed(seed)
        self.last_action = 0.0
        senario = np.random.choice([0, 1])
        # print(senario)
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -0.5*np.random.uniform(2, 5)
            self.network.sgen.at[2, 'p_mw'] = -0.6*np.random.uniform(10, 30)
            self.network.sgen.at[3, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[4, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[5, 'p_mw'] = -0.4*np.random.uniform(2, 8)

        elif(senario == 1): #high voltage 
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = 0.5*np.random.uniform(2, 10)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(5, 40)
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 14)
            self.network.sgen.at[4, 'p_mw'] = 0.4*np.random.uniform(2, 14) 
            self.network.sgen.at[5, 'p_mw'] = 0.4*np.random.uniform(2, 14) 
            
        else: #mixture
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -2*np.random.uniform(2, 3)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(15, 35)
            self.network.sgen.at[2, 'q_mvar'] = 0.1*self.network.sgen.at[2, 'p_mw']
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 12)
            self.network.sgen.at[4, 'p_mw'] = -2*np.random.uniform(2, 8) 
            self.network.sgen.at[5, 'p_mw'] = 0.2*np.random.uniform(2, 12) 
            self.network.sgen.at[5, 'q_mvar'] = 0.2*self.network.sgen.at[5, 'p_mw']
            
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state

def create_56bus():
    pp_net = pp.converter.from_mpc('SCE_56bus.mat', casename_mpc_file='case_mpc')
    
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 17, p_mw = 1.5, q_mvar=0)
    pp.create_sgen(pp_net, 20, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 29, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 44, p_mw = 2, q_mvar=0)
    pp.create_sgen(pp_net, 52, p_mw = 2, q_mvar=0)
    
    return pp_net
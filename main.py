import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter
from environment import VoltageCtrl_nonlinear,create_56bus

import os
import gym
import numpy as np
from gym import wrappers

import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='discount factor for model (default: 0.01)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=0.2, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.1, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=200, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=13, metavar='N',
                    help='random seed (default: 13)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=600, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=100, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
args = parser.parse_args()

writer = SummaryWriter()

pp_net = create_56bus()
injection_bus = np.array([17, 20, 29, 44, 52])
env = VoltageCtrl_nonlinear(pp_net, injection_bus)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent1 = DDPG(args.gamma, args.tau, args.hidden_size,
                    env.observation_space.shape[0], env.action_space)
agent2 = DDPG(args.gamma, args.tau, args.hidden_size,
                    env.observation_space.shape[0], env.action_space)
agent3 = DDPG(args.gamma, args.tau, args.hidden_size,
                    env.observation_space.shape[0], env.action_space)
agent4 = DDPG(args.gamma, args.tau, args.hidden_size,
                    env.observation_space.shape[0], env.action_space)                                        
agent5 = DDPG(args.gamma, args.tau, args.hidden_size,
                    env.observation_space.shape[0], env.action_space)

memory1 = ReplayMemory(args.replay_size)
memory2 = ReplayMemory(args.replay_size)
memory3 = ReplayMemory(args.replay_size)
memory4 = ReplayMemory(args.replay_size)
memory5 = ReplayMemory(args.replay_size)

ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, 
    desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None

rewards = []
total_numsteps = 0
updates = 0
best_val_reward = -100000

for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])

    if args.ou_noise: 
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0.05, args.exploration_end -
                                                                      i_episode) / args.exploration_end + args.final_noise_scale
        # ounoise.scale = args.noise_scale
        ounoise.reset()

    # if args.param_noise and args.algo == "DDPG":
    #     agent.perturb_actor_parameters(param_noise)

    episode_reward = 0
    episode_len = 0
    log = []
    while True: #state:[1,1]
        state1 = state[:,0].unsqueeze(-1)
        state2 = state[:,1].unsqueeze(-1)
        state3 = state[:,2].unsqueeze(-1)
        state4 = state[:,3].unsqueeze(-1)
        state5 = state[:,4].unsqueeze(-1)
        action1 = agent1.select_action(state1, ounoise, param_noise)
        action2 = agent2.select_action(state2, ounoise, param_noise)
        action3 = agent3.select_action(state3, ounoise, param_noise)
        action4 = agent4.select_action(state4, ounoise, param_noise)
        action5 = agent5.select_action(state5, ounoise, param_noise)
        action = torch.cat([action1, action2, action3, action4, action5], dim=1)
        log.append(torch.cat([state,action],dim=1).detach().cpu().numpy())
        next_state, reward, done, _ = env.step(action.numpy()[0])
        total_numsteps += 1
        episode_reward += np.sum(reward)
        episode_len += 1

        action = torch.Tensor(action)
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        memory1.push(state1, action1, mask, next_state[:,0].unsqueeze(-1), reward[:,0])
        memory2.push(state2, action2, mask, next_state[:,1].unsqueeze(-1), reward[:,1])
        memory3.push(state3, action3, mask, next_state[:,2].unsqueeze(-1), reward[:,2])
        memory4.push(state4, action4, mask, next_state[:,3].unsqueeze(-1), reward[:,3])
        memory5.push(state5, action5, mask, next_state[:,4].unsqueeze(-1), reward[:,4])

        state = next_state

        if len(memory1) > args.batch_size:
            for _ in range(args.updates_per_step):
                transitions1 = memory1.sample(args.batch_size)
                batch1 = Transition(*zip(*transitions1))

                transitions2 = memory2.sample(args.batch_size)
                batch2 = Transition(*zip(*transitions2))

                transitions3 = memory3.sample(args.batch_size)
                batch3 = Transition(*zip(*transitions3))

                transitions4 = memory4.sample(args.batch_size)
                batch4 = Transition(*zip(*transitions4))

                transitions5 = memory5.sample(args.batch_size)
                batch5 = Transition(*zip(*transitions5))

                value_loss, policy_loss = agent1.update_parameters(batch1)
                value_loss, policy_loss = agent2.update_parameters(batch2)
                value_loss, policy_loss = agent3.update_parameters(batch3)
                value_loss, policy_loss = agent4.update_parameters(batch4)
                value_loss, policy_loss = agent5.update_parameters(batch5)

                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)

                updates += 1
        if done or episode_len==30:
            log = np.vstack(log)
            np.savetxt('train_log2.txt',log, fmt='%1.4e')
            break

    writer.add_scalar('reward/train', episode_reward, i_episode)

    # Update param_noise based on distance metric
    # if args.param_noise:
    #     episode_transitions = memory.memory[memory.position-t:memory.position]
    #     states = torch.cat([transition[0] for transition in episode_transitions], 0)
    #     unperturbed_actions = agent.select_action(states, None, None)
    #     perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

    #     ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
    #     param_noise.adapt(ddpg_dist)

    rewards.append(episode_reward)
    log = []
    if i_episode % 10 == 0:
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        test_len=0
        while True:
            state1 = state[:,0].unsqueeze(-1)
            state2 = state[:,1].unsqueeze(-1)
            state3 = state[:,2].unsqueeze(-1)
            state4 = state[:,3].unsqueeze(-1)
            state5 = state[:,4].unsqueeze(-1)
            action1 = agent1.select_action(state1, ounoise, param_noise)
            action2 = agent2.select_action(state2, ounoise, param_noise)
            action3 = agent3.select_action(state3, ounoise, param_noise)
            action4 = agent4.select_action(state4, ounoise, param_noise)
            action5 = agent5.select_action(state5, ounoise, param_noise)
            action = torch.cat([action1, action2, action3, action4, action5], dim=1)
            log.append(torch.cat([state,action],dim=1).detach().cpu().numpy())
            test_len+=1

            next_state, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += np.sum(reward)

            next_state = torch.Tensor([next_state])

            state = next_state
            if done or test_len==60:
                log = np.vstack(log)
                np.savetxt('test_log2.txt',log, fmt='%1.4e')
                break

        writer.add_scalar('reward/test', episode_reward, i_episode)
        if episode_reward > best_val_reward:
            best_val_reward = episode_reward
            model1_pth = './models/agent1.pt'
            model2_pth = './models/agent2.pt'
            model3_pth = './models/agent3.pt'
            model4_pth = './models/agent4.pt'
            model5_pth = './models/agent5.pt'
            torch.save(agent1,model1_pth)
            torch.save(agent2,model2_pth)
            torch.save(agent3,model3_pth)
            torch.save(agent4,model4_pth)
            torch.save(agent5,model5_pth)


        rewards.append(episode_reward)
        print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))
    
env.close()

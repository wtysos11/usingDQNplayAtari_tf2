import torch
import gym

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging
import math

from replay_buffer import ReplayBuffer
from wrappers import *

from gym import spaces
import torch.nn as nn
import torch.nn.functional as F

# Class structure loosely inspired by https://towardsdatascience.com/beating-video-games-with-deep-q-networks-7f73320b9592
class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(
            observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form (channels,width,height), and this should be 4*84*84'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        
        # Based on paper "Human-level control through deep reinforcement learning"
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7*7 , out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_space.n)
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0],-1)
        return self.fc(conv_out)

class DQNplayer:
    '''
    然后还有一个函数，能够根据给定的神经网络运行这个游戏，自动进行。

    Attributes:
        Q_main，神经网络，主要训练函数。输入为Atari状态，输出为大小等于动作空间的向量
        Q_target，Q_main的旧有拷贝，负责切断因果性进行最大动作估计。
    '''
    def __init__(self,hyper_param,networkName="conv2d"):
        '''
        初始化，这里应该声明网络和超参数
        Args:
            name，表示所选择的Atari游戏，要求必须为RGB模式的输入
        '''
        # 声明环境
        self.hyper_param = hyper_param

        self.env = gym.make(hyper_param["env"])
        self.env = NoopResetEnv(self.env, noop_max=30)
        self.env = MaxAndSkipEnv(self.env, skip=4)
        self.env = EpisodicLifeEnv(self.env)
        self.env = FireResetEnv(self.env)
        self.env = WarpFrame(self.env)
        self.env = PyTorchFrame(self.env)
        self.env = ClipRewardEnv(self.env)
        self.env = FrameStack(self.env, 4)
        self.env = gym.wrappers.Monitor(
        self.env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

        self.gameName = hyper_param["env"]
        self.networkName = networkName
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 构建网络
        self.learning_rate = hyper_param["learning-rate"]

        assert self.env.observation_space.shape == (4,84,84)
        self.policy_network = DQN(self.env.observation_space,self.env.action_space).to(self.device)
        self.target_network = DQN(self.env.observation_space,self.env.action_space).to(self.device)
        self.update_target_network()
        self.target_network.eval()
        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters()
            , lr=self.learning_rate)
        # 基础设施
        # 弃用简易buffer，改用priority_buffer(from baseline.dqn)
        # self.memoryBuffer = MemoryBuffer(max_length = max_memory_length)
        # 使用Priority_buffer，需要修改三个地方
        # 第一，开头声明。这里需要加上各种超参数
        # 第二，与普通的buffer相比，拿变量的时候不是五个值而是七个值，最后会有一个weights和一个batch_idxes
        # 第三，计算完成之后要将td_errors用来更新权重。关键在于td_error怎么计算，按论文上的说法，TD-error = q-target - q-eval
        # new_priorities = abs(td_error) + epsilon(1e-6)，然后调用replay_buffer.update_priorities方法
        self.buffer_size = hyper_param["replay-buffer-size"]
        self.memoryBuffer = ReplayBuffer(hyper_param["replay-buffer-size"])

        self.learning_starts = hyper_param["learning-starts"]
        # 声明超参数
        # epsilon控制系统
        # epsilon min max decay rate...
        self.eps_start = hyper_param["eps-start"]
        self.eps_end = hyper_param["eps-end"]
        self.eps_fraction = hyper_param["eps-fraction"]

        # 学习率
        self.discount_factor = hyper_param["discount-factor"]
        # Q_target更新速率
        self.target_update_rate = hyper_param["target-update-freq"]

        # 记录系统
        self.actionRandomRecorder = deque(maxlen = 10000)

    def main_process(self):
        '''
        主过程，详情见上面的部分
        '''
        self.global_counting = 0 #额外超参数，每次训练重置，负责提供各个超参数的衰变
        num_steps = self.hyper_param["num-steps"]
        batch_size = self.hyper_param["batch-size"]
        step_train = self.hyper_param["learning-freq"]
        rewardFileWriter = open("episodeReward","w")
        
        # 声明规划器
        episode_total_reward = [] # 用于存放每一轮的奖励，并绘制最终曲线
        # exploration
         # 目的是填满缓冲区。我想了一下还是不用多步了，因为计算出来的肯定是错的，所以用不用是没区别的。
         # 最好的方式应该是保留下缓冲区的数据，供之后的训练使用。
        episode_num_counter = 0

        while self.global_counting < num_steps:
        #对于每一个episode
            logging.warning("episode num {} start".format(episode_num_counter))
            
            currentEpisodeReward = 0
            #初始化环境，主要是运行obs = env.reset()
            observation = self.env.reset()
            gameDone = False
            # 奖励记录，供multi_step计算使用
            rewardList = []
            lossList = []       
            # 用来维护，保存最新的若干个处理后的帧
            # 每次游戏时应该清空之前存储的状态信息

            #当游戏没有结束
            while not gameDone and self.global_counting < num_steps:
                assert not gameDone #理论上不应该有gameDone为true的情况
                
                #epsilon-greedy，拿取动作
                action = self.epsilon_greedy(observation)
                #执行动作,获取新环境
                next_observation, reward, done, info = self.env.step(action)
                gameDone = done
                currentEpisodeReward += reward # 更新episode奖励
                rewardList.append(reward)

                self.memoryBuffer.add(observation,action,reward,next_observation,float(done))
                observation = next_observation
                # reward应当为rewardRecorder的和，作为前面的和式计算。此时的reward为未来的奖励
                # 具体来说，状态reward_0*gammma^0+reward_1*gammma^1+...+reward_{n-1}*gammma^{n-1}
                # 然后最后的估计值与gamma^n相乘
                
                #如果运行了指定次数且有足够多的训练数据，则开始训练
                if self.global_counting > self.learning_starts and self.global_counting % step_train == 0:
                    #self.global_counting > self.buffer_size:
                    #从记忆库拿取数据，转换成数组形式
                    device = self.device
                    experience = self.memoryBuffer.sample(batch_size)
                    (obs_array,action_array,reward_array,next_obs_array,done_array) = experience
                    # preprocessing
                    obs_array = np.array(obs_array)/255.
                    next_obs_array = np.array(next_obs_array)/255.
                    # load torch data
                    states = torch.from_numpy(obs_array).float().to(device)
                    actions = torch.from_numpy(action_array).long().to(device)
                    rewards = torch.from_numpy(reward_array).float().to(device)
                    next_states = torch.from_numpy(next_obs_array).float().to(device)
                    dones = torch.from_numpy(done_array).float().to(device)
                    # 计算最大值
                    with torch.no_grad():
                        _, policyNetworkFutureAction = self.policy_network(next_states).max(1) #max接受参数为计算的轴；第一个是最大的值，第二个是表情
                        targetNetworkFutureActionval = self.target_network(next_states).gather(1, policyNetworkFutureAction.unsqueeze(1)).squeeze() #沿1轴检索，并展开
                        q_target = rewards + (1 - dones) * self.discount_factor * targetNetworkFutureActionval
                    q_eval = self.policy_network(states)
                    q_eval = q_eval.gather(1, actions.unsqueeze(1)).squeeze()#拿到原动作对应的q_eval
                    # Huber
                    loss = F.smooth_l1_loss(q_eval, q_target)
                    # 常规操作
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()
                    # 清空，减少爆内存可能性
                    del states
                    del next_states
                    lossList.append(loss.item())
                    
                # 如果又经过了指定的时间
                if self.global_counting > self.learning_starts and self.global_counting % self.target_update_rate == 0:
                    #Q_target更新
                    self.update_target_network()
                
                # 计数器更新
                self.global_counting += 1
                if self.global_counting > self.learning_starts:
                    #记录episode总奖励
                    #logging.warning("mean loss:{}",np.mean(lossList))
                    if self.global_counting % self.hyper_param["backup_record_step"] == 0:
                        self.savemodel(str(self.global_counting))
            
            episode_num_counter += 1
            episode_total_reward.append(currentEpisodeReward)
            # 记录在文件中
            rewardFileWriter.write("{}\n".format(currentEpisodeReward))
            rewardFileWriter.flush()
            logging.warning("episode {}/frame {} 's reward {}".format(episode_num_counter,self.global_counting,currentEpisodeReward))
            #logging.warning("reward distribute: max reward {}/ minreward {}".format(max(rewardList),min(rewardList)))
            if len(lossList) > 0:
                logging.warning("mean of loss is {}".format(np.mean(lossList)))
            else:
                logging.warning("Still not start")

            # 输出平均值
            if len(episode_total_reward)>0:
                logging.warning("avg reward of 10 episode {}".format(np.mean(episode_total_reward[-10:])))
                logging.warning("avg random action times for 1w frame {}:".format(np.mean(self.actionRandomRecorder)))
                logging.warning("Current Epsilon val : {}".format(self.currentEpsilonVal))

            if episode_num_counter % self.hyper_param["print-freq"] == 0:
                self.savemodel()

        # 训练完毕
        rewardFileWriter.close()
        plt.cla()
        plt.plot(episode_total_reward)
        plt.show()

    def epsilon_greedy(self,state):
        '''
            接受神经网络的输出值，返回期望进行的动作。
            函数会产生一个随机数，如果随机数的值小于epsilon，则返回随机动作；
                否则，则返回action数组中值最大的动作。
        Args:
            actionVal，大小为action_space_n的输出向量，为Q_main的输出向量
            epsilon，一个0~1的float值。
        Returns:
            actionNum，一个0~action_space_n-1的标量，表示执行的动作序号
        Raises:
            None
        '''
        eps_timesteps = self.eps_fraction * \
            float(self.hyper_param["num-steps"]) #总共跑1百万次，其中前10%用来做探索
        fraction = min(1.0, float(self.global_counting) / eps_timesteps)
        currentEpsilonVal = self.eps_start + fraction * \
            (self.eps_end - self.eps_start)
        self.currentEpsilonVal = currentEpsilonVal
        randomNum = np.random.rand() # 获取一个0~1的随机数
        if randomNum < currentEpsilonVal:
            # 返回随机动作
            self.actionRandomRecorder.append(1)
            return self.env.action_space.sample()
        else:
            # 返回当前最大动作
            self.actionRandomRecorder.append(0)
            device = self.device
            state = np.array(state)/255.
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_network(state)
                _, action = q_values.max(1)
                return action.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def savemodel(self,name="Q_main"):
        torch.save(self.policy_network.state_dict(), f'checkpoint.pth')


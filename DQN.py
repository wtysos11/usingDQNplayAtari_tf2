import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Flatten,Dense,Conv2D,LeakyReLU,Multiply,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
import gym

import cv2
from skimage.color import rgb2gray

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging
import math

from replay_buffer import ReplayBuffer
from wrappers import *

class NeuralNetworkBuilder:
    '''
        负责构建Q神经网络
    '''
    def __init__(self,learning_rate = 0.0001):
        '''
            初始化
        '''
        self.learning_rate = learning_rate
    
    def build_duel(self,n_input,n_output):
        '''
        Args:
            n_input 输入数据的维数，这里应该是图像的大小
            n_output 输出数据的维数，这里应该是动作空间的大小
        Returns:
            model 返回一个Keras类型的model
        Reference:
            参考了https://github.com/keras-rl/keras-rl 的网络架构
        '''
        # Game-specific 游戏特化代码，如果后续要做泛化的话需要修改此处
        def lambda_out_shape(input_shape): # lambda层使用的计算后续shape的函数
            shape = list(input_shape)
            shape[-1] = 1
            return tuple(shape)
        # 第一层应该是一个图像预处理层。使用Pong-v0时的输入为210*160*3
        model_input = Input(shape=n_input)
        action_one_hot = Input(shape=(n_output,))# 这里默认n_output为动作空间大小
        # 初始化完毕后是三层Conv2D+BN(可能可以不加？先不加看一下效果)
        conv1 = Conv2D(32,(8,8),strides=(4,4),activation='relu')(model_input)
        conv2 = Conv2D(64,(4,4),strides=(2,2),activation='relu')(conv1)
        conv3 = Conv2D(64,(3,3),strides=(1,1),activation='relu')(conv2)
        # 再之后是两层全连接层，确保最后的输出向量维数为action_space的大小
        flatten = Flatten()(conv3)
        # Dueling DQN
        fc1 = Dense(512)(flatten)
        leakyrelu = LeakyReLU()(fc1)
        advantage = Dense(n_output)(leakyrelu)
        # V(s)
        fc2 = Dense(512)(flatten)
        value = Dense(1)(fc2)
        # 最终的值Q=V+A
        # 或者说，Q = A-mean(A) + V
        policy = Lambda(lambda x: x[0] - K.mean(x[0]) + x[1])([advantage, value])

        # 声明模型，并返回它
        model = Model(inputs=[model_input], outputs=[policy])
        model.compile(loss=keras.losses.Huber(),optimizer=Adam(self.learning_rate))
        return model

    def build_conv2d(self,n_input,n_output):
        '''
        Args:
            n_input 输入数据的维数，这里应该是图像的大小
            n_output 输出数据的维数，这里应该是动作空间的大小
        Returns:
            model 返回一个Keras类型的model
        Reference:
            参考了https://github.com/keras-rl/keras-rl 的网络架构
        '''
        # Game-specific 游戏特化代码，如果后续要做泛化的话需要修改此处
        model_input = Input(shape=n_input)
        # 初始化完毕后是三层Conv2D+BN(可能可以不加？先不加看一下效果)
        conv1 = Conv2D(32,(8,8),strides=(4,4),activation='relu')(model_input)
        conv2 = Conv2D(64,(4,4),strides=(2,2),activation='relu')(conv1)
        conv3 = Conv2D(64,(3,3),strides=(1,1),activation='relu')(conv2)
        # 再之后是两层全连接层，确保最后的输出向量维数为action_space的大小
        flatten = Flatten()(conv3)
        fc1 = Dense(512)(flatten)
        model_output = Dense(n_output)(fc1)

        # 声明模型，并返回它
        model = Model(inputs=[model_input], outputs=[model_output])
        model.compile(loss=keras.losses.Huber(),optimizer=RMSprop(self.learning_rate))
        return model

    def build_network(self,n_input,n_output,name="conv2d"):
        '''
            使用默认值构建神经网络
        Args:
            name 表示生成的神经网络类型
            n_input 输入数据的维数，这里应该是图像的大小
            n_output 输出数据的维数，这里应该是动作空间的大小
        Returns:
            model 返回一个Keras类型的model
        '''
        if name=="duel":
            return self.build_duel(n_input,n_output)
        elif name == "conv2d":
            return self.build_conv2d(n_input,n_output)

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
        # 构建网络
        self.learning_rate = hyper_param["learning-rate"]
        Builder = NeuralNetworkBuilder(self.learning_rate)
        network_input = (84,84,4) # 根据skip frame得到的形状
        self.policy_network = Builder.build_network(network_input,self.env.action_space.n,name = networkName)
        self.target_network = Builder.build_network(network_input,self.env.action_space.n,name = networkName)
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
            gameDone = False
            currentEpisodeReward = 0
            #初始化环境，主要是运行obs = env.reset()
            observation = self.env.reset()
            observation = self.preprocessing(observation)
            # 奖励记录，供multi_step计算使用
            
            # 用来维护，保存最新的若干个处理后的帧
            # 每次游戏时应该清空之前存储的状态信息
            rewardList = []
            lossList = []
            #当游戏没有结束
            while not gameDone:
                if self.global_counting % 1000 == 0:
                    logging.info("episode :{}/ frame : {}".format(episode_num_counter,self.global_counting))
                #拿取当前环境并初始化图像（上一时刻的下一记录即为当前记录）
                #epsilon-greedy，拿取动作
                action_val = self.policy_network.predict(np.array([observation])) # 这里需要思考一下，我觉得应该是要上升为数组再开回来
                action = self.epsilon_greedy(action_val.ravel())
                #执行动作,获取新环境
                next_observation, reward, done, _ = self.env.step(action)
                gameDone = done
                currentEpisodeReward += reward # 更新episode奖励
                rewardList.append(reward)

                self.memoryBuffer.add(observation,action,reward,self.preprocessing(next_observation),float(done))
                observation = self.preprocessing(next_observation)
                # reward应当为rewardRecorder的和，作为前面的和式计算。此时的reward为未来的奖励
                # 具体来说，状态reward_0*gammma^0+reward_1*gammma^1+...+reward_{n-1}*gammma^{n-1}
                # 然后最后的估计值与gamma^n相乘
                
                #如果运行了指定次数且有足够多的训练数据，则开始训练
                if self.global_counting > self.learning_starts and self.global_counting % step_train == 0:
                    #self.global_counting > self.buffer_size:
                    #从记忆库拿取数据，转换成数组形式
                    experience = self.memoryBuffer.sample(batch_size)
                    (obs_array,action_array,reward_array,next_obs_array,done_array) = experience
                    # weights是在duel网络中用来作为选择最佳动作的输入，这里应该是不用的
                    # 进行预处理
                    policyNetworkFutureActionval = self.policy_network.predict(np.array(next_obs_array))
                    # 此处一定要使用Q_target进行计算
                    targetNetworkFutureActionval = self.target_network.predict(np.array(next_obs_array)) #得到Q(s')的所有动作的向量
                    # 计算Q表
                    policyNetworkCurrentActionval = self.policy_network.predict(np.array(obs_array))
                    # Double Q优化
                    maxFutureAction = np.argmax(policyNetworkFutureActionval,axis=1)
                    maxActionVal = targetNetworkFutureActionval[np.arange(len(targetNetworkFutureActionval)),maxFutureAction]
                    #q_target = reward_array + (1-done_array)*self.discount_factor * maxActionVal
                    # 对于每一个状态，对Q表进行更新。顺便计算td_errors
                    policyNetworkCurrentActionval[np.arange(len(targetNetworkFutureActionval)),action_array] = maxActionVal
                        
                    #喂给神经网络，使用MSE作为损失函数。进行训练
                    loss = self.policy_network.train_on_batch(np.array(obs_array),policyNetworkCurrentActionval)
                    lossList.append(loss)
                    del obs_array
                    del next_obs_array
                    
                # 如果又经过了指定的时间
                if self.global_counting > self.learning_starts and self.global_counting % self.target_update_rate == 0:
                    #Q_target更新
                    self.target_network.set_weights(self.policy_network.get_weights())
                
                # 计数器更新
                self.global_counting += 1
            
            episode_num_counter += 1
            episode_total_reward.append(currentEpisodeReward)
            # 记录在文件中
            rewardFileWriter.write("{}\n".format(currentEpisodeReward))
            rewardFileWriter.flush()
            logging.warning("episode {} 's reward {}".format(episode_num_counter,currentEpisodeReward))
            logging.warning("reward distribute: max reward {}/ minreward {}".format(max(rewardList),min(rewardList)))
            if self.global_counting > self.learning_starts:
                #记录episode总奖励
                #logging.warning("mean loss:{}",np.mean(lossList))
                if episode_num_counter % 500 == 0 and episode_num_counter > 0:
                    self.savemodel(str(episode_num_counter))
        # 训练完毕
        rewardFileWriter.close()
        plt.cla()
        plt.plot(episode_total_reward)
        plt.show()

    def epsilon_greedy(self,actionVal):
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
            float(self.hyper_param["num-steps"])
        fraction = min(1.0, float(self.global_counting) / eps_timesteps)
        currentEpsilonVal = self.eps_start + fraction * \
            (self.eps_end - self.eps_start)
        randomNum = np.random.rand() # 获取一个0~1的随机数
        if randomNum < currentEpsilonVal:
            # 返回随机动作
            return np.random.randint(len(actionVal))
        else:
            # 返回当前最大动作
            return np.argmax(actionVal)

    def preprocessing(self,observation):
        '''
        预处理函数，负责对数据进行预处理。
        分为两种情况，如果是RGB数据，则将其映射到[0,1]上，即除以255
        如果是RAM数据，则直接返回
        '''
        #if self.networkName == "conv2d" or self.networkName == "duel":
        #    observation = observation[34:-16, :, :] # 裁剪掉上下的无用部分
        #    resized_frame = cv2.resize(observation, (84, 84), interpolation = cv2.INTER_AREA)
        #    frame_gray = rgb2gray(resized_frame)
        #    return frame_gray
        #else:
        return np.moveaxis(np.array(observation),0,2)/255.

    def savemodel(self,name="Q_main"):
        self.policy_network.save(name)


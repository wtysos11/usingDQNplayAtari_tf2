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

from replay_buffer import PrioritizedReplayBuffer,LinearSchedule

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


class MemoryBuffer:
    '''
    DQNplayer中的缓存管理系统
    需要记录元组(s_t,a_t,r,s_{t+1},done)，供之后的训练使用。
    '''
    def __init__(self,max_length = 10000):
        self.memory = deque(maxlen=max_length)

    def getLength(self):
        return len(self.memory)

    def addItem(self,item):
        self.memory.append(item)

    def sampleBatch(self,batch_size = 32):
        '''
        输入一个batch所需要的数据量，输出这个batch的数据。
        数据由记忆库乱序输出得到
        Args:
            batch_size 一个batch所需要的数据量
        Returns:
            data元组 这个batch的数据，是一个np.array的形式，由5个列组成
        '''
        order = np.random.permutation(len(self.memory))[:batch_size]
        data = np.array(self.memory)[order]
        return data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]

class DQNplayer:
    '''
    然后还有一个函数，能够根据给定的神经网络运行这个游戏，自动进行。

    Attributes:
        Q_main，神经网络，主要训练函数。输入为Atari状态，输出为大小等于动作空间的向量
        Q_target，Q_main的旧有拷贝，负责切断因果性进行最大动作估计。
    '''
    def __init__(self,name="Pong-v0",networkName="conv2d",max_memory_length = 20000):
        '''
        初始化，这里应该声明网络和超参数
        Args:
            name，表示所选择的Atari游戏，要求必须为RGB模式的输入
        '''
        # 声明环境
        self.env = gym.make(name)
        self.gameName = name
        self.networkName = networkName
        # 构建网络
        self.learning_rate = 0.0001
        Builder = NeuralNetworkBuilder(self.learning_rate)
        network_input = (84,84,4) # 根据skip frame得到的形状
        self.Q_main = Builder.build_network(network_input,self.env.action_space.n,name = networkName)
        self.Q_target = Builder.build_network(network_input,self.env.action_space.n,name = networkName)
        # 基础设施
        # 弃用简易buffer，改用priority_buffer(from baseline.dqn)
        # self.memoryBuffer = MemoryBuffer(max_length = max_memory_length)
        # 使用Priority_buffer，需要修改三个地方
        # 第一，开头声明。这里需要加上各种超参数
        # 第二，与普通的buffer相比，拿变量的时候不是五个值而是七个值，最后会有一个weights和一个batch_idxes
        # 第三，计算完成之后要将td_errors用来更新权重。关键在于td_error怎么计算，按论文上的说法，TD-error = q-target - q-eval
        # new_priorities = abs(td_error) + epsilon(1e-6)，然后调用replay_buffer.update_priorities方法
        self.buffer_size = max_memory_length
        self.prioritized_replay_alpha = 0.6 # 用于构建priority_replay部分
        self.prioritized_replay_beta0 = 0.4 # 用于在sample中充当参数
        self.prioritized_replay_eps = 1e-6 #用于更新priority
        self.memoryBuffer = PrioritizedReplayBuffer(self.buffer_size,self.prioritized_replay_alpha)
        # 声明超参数
        # epsilon控制系统
        # epsilon min max decay rate...
        self.epsilon_min = 0.02
        self.epsilon_max = 1.0
        self.epsilon_decay_steps = 100000

        # 学习率
        self.trajectory_length = 4 # multi-step的轨迹长度, Rainbow原文用了3，我觉得4也差不多。
        self.discount_factor = 0.99
        # Q_target更新速率
        self.target_update_rate = 1000

    def main_process(self,episode_num = 10000,batch_size = 32):
        '''
        主过程，详情见上面的部分
        '''
        self.global_counting = 0 #额外超参数，每次训练重置，负责提供各个超参数的衰变
        step_train = 4 # 每4次行动训练一次神经网络
        frameNum_perState = 4 # 4个环境样本组合在一起作为一个状态
        rewardFileWriter = open("episodeReward","w")
        
        # 声明规划器
        self.prioritized_replay_beta_iters = episode_num * 1500
        self.beta_schedule = LinearSchedule(self.prioritized_replay_beta_iters,
                                       initial_p=self.prioritized_replay_beta0,
                                       final_p=1.0)

        episode_total_reward = [] # 用于存放每一轮的奖励，并绘制最终曲线
        # exploration
         # 目的是填满缓冲区。我想了一下还是不用多步了，因为计算出来的肯定是错的，所以用不用是没区别的。
         # 最好的方式应该是保留下缓冲区的数据，供之后的训练使用。

        for episode_num_counter in range(episode_num):
        #对于每一个episode
            logging.warning("episode num {} start".format(episode_num_counter))
            gameDone = False
            currentEpisodeReward = 0
            #初始化环境，主要是运行obs = env.reset()
            observation = self.env.reset()
            # 奖励记录，供multi_step计算使用
            rewardRecorder = deque(maxlen=self.trajectory_length)
            observationRecorder = deque(maxlen=self.trajectory_length) #里面存放的是84*84*4的处理后数据，与神经网络的输入数据相同
            actionRecorder = deque(maxlen=self.trajectory_length)
            
            # 用来维护，保存最新的若干个处理后的帧
            # 每次游戏时应该清空之前存储的状态信息
            preprocessFrameStack = deque(maxlen = frameNum_perState)
            rewardList = []
            lossList = []
            # 预处理，直接填满
            while len(preprocessFrameStack) < frameNum_perState:
                preprocessFrameStack.append(self.preprocessing(observation)) #仅在此处添加状态，将新的处理后的状态加入
            while len(observationRecorder) < self.trajectory_length:
                observationRecorder.append(np.stack(preprocessFrameStack,axis=2))
            while len(actionRecorder) < self.trajectory_length:
                actionRecorder.append(self.env.action_space.sample())
            #当游戏没有结束
            while not gameDone:
                if self.global_counting % 1000 == 0:
                    print("episode :{}/ frame : {}".format(episode_num_counter,self.global_counting))
                #拿取当前环境并初始化图像（上一时刻的下一记录即为当前记录）
                observation = self.preprocessing(observation)
                #epsilon-greedy，拿取动作
                action_val = self.Q_main.predict(np.array([np.stack(preprocessFrameStack,axis=2)])) # 这里需要思考一下，我觉得应该是要上升为数组再开回来
                action = self.epsilon_greedy(action_val)
                #执行动作,获取新环境
                next_observation, reward, done, _ = self.env.step(action)
                gameDone = done
                currentEpisodeReward += reward # 更新episode奖励
                rewardList.append(reward)

                # 考虑在此处执行Multi-step。进行全面的更新，以当前为0，压入S_0,a_0,reward_0
                # multi-step的本质实质上就是用下n步的奖励来代替这一步的奖励，从而更好的完成估计。因此需要维护一个未来的奖励数组
                observationRecorder.append(np.stack(preprocessFrameStack,axis=2))
                actionRecorder.append(action) #状态所对应的动作应该加入
                rewardRecorder.append(reward) #动作所对应的奖励应该加入
                preprocessFrameStack.append(self.preprocessing(next_observation))
                # reward应当为rewardRecorder的和，作为前面的和式计算。此时的reward为未来的奖励
                # 具体来说，状态reward_0*gammma^0+reward_1*gammma^1+...+reward_{n-1}*gammma^{n-1}
                # 然后最后的估计值与gamma^n相乘
                for beginPoint in range(len(observationRecorder)):
                    originReward = 0.
                    discountRecord = 1
                    for rewardIndex in range(beginPoint,len(rewardRecorder)):
                        originReward += rewardRecorder[rewardIndex] * discountRecord
                        discountRecord *= self.discount_factor
                    originState = observationRecorder[0]
                    originAction = actionRecorder[0]
                    self.memoryBuffer.add(originState,originAction,reward,np.stack(preprocessFrameStack,axis=2),done,self.trajectory_length-beginPoint)

                #如果运行了指定次数且有足够多的训练数据，则开始训练
                if self.global_counting % step_train == 0 and \
                    len(self.memoryBuffer) >= batch_size and \
                    self.global_counting > self.buffer_size:
                    #从记忆库拿取数据，转换成数组形式
                    experience = self.memoryBuffer.sample(batch_size,beta = self.beta_schedule.value(self.global_counting))
                    (obs_array,action_array,reward_array,next_obs_array,done_array,weights,batch_idxes) = experience
                    # weights是在duel网络中用来作为选择最佳动作的输入，这里应该是不用的
                    # 进行预处理
                    obs_array = [x for x in obs_array]
                    next_obs_array = [x for x in next_obs_array]
                    current_actionVal = self.Q_main.predict(np.array(next_obs_array))
                    # 此处一定要使用Q_target进行计算
                    next_actionVal = self.Q_target.predict(np.array(next_obs_array)) #得到Q(s')的所有动作的向量
                    # 计算Q表
                    QTable = self.Q_main.predict(np.array(obs_array))
                    # 对于每一个状态，对Q表进行更新。顺便计算td_errors
                    td_errors = np.zeros(batch_size)
                    for i in range(len(obs_array)):
                        replay_action = action_array[i]
                        # Q*(s,a) = r + \gamma * Q'(s',a')
                        # 此处的思路是这样的，最终要求的损失值是(y-Q(s_t,a_t))^2
                        # 如果将Q(s_t)的动作a_t部分更换成新的y，其他地方不变
                        # 在MSE的loss计算模式下，其他地方为0，数组编号为a_t时有(y-Q(s_t,a_t))^2
                        # Double Q优化处
                        maxFutureAction = np.argmax(current_actionVal[i],axis=-1)#使用最新的网络来选择动作
                        maxActionVal = next_actionVal[i][maxFutureAction] #拿到目标网络中的值
                        # multi_step公式中，此处为self.discount_factor^self.trajectory_length
                        q_target = reward_array[i] + (1-done_array[i])*math.pow(self.discount_factor,self.trajectory_length) * maxActionVal
                        # TD_error = abs(q_target - q_eval),abs放在后面计算
                        td_errors[i] = QTable[i][replay_action] - q_target
                        # 更新，这样keras中就可以直接计算损失函数
                        QTable[i][replay_action] = q_target
                        
                    #喂给神经网络，使用MSE作为损失函数。进行训练
                    loss = self.Q_main.train_on_batch(np.array(obs_array),QTable)
                    lossList.append(loss)
                    # priority_buffer 更新权重
                    new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                    self.memoryBuffer.update_priorities(batch_idxes, new_priorities)
                    
                # 如果又经过了指定的时间
                if self.global_counting % self.target_update_rate == 0:
                    #Q_target更新
                    self.Q_target.set_weights(self.Q_main.get_weights())
                
                # 计数器更新
                observation = next_observation
                self.global_counting += 1
            #记录episode总奖励
            episode_total_reward.append(currentEpisodeReward)
            # 记录在文件中
            rewardFileWriter.write("{}\n".format(currentEpisodeReward))
            rewardFileWriter.flush()
            logging.warning("episode {} 's reward {}, loss {}".format(episode_num_counter,currentEpisodeReward,np.mean(lossList)))
            logging.warning("reward distribute: max reward {}/ minreward {}".format(max(rewardList),min(rewardList)))
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
        randomNum = np.random.rand() # 获取一个0~1的随机数
        currentEpsilonVal = max(self.epsilon_min,self.epsilon_max-(self.epsilon_max - self.epsilon_min) * self.global_counting/self.epsilon_decay_steps)
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
        if self.networkName == "conv2d" or self.networkName == "duel":
            observation = observation[34:-16, :, :] # 裁剪掉上下的无用部分
            resized_frame = cv2.resize(observation, (84, 84), interpolation = cv2.INTER_AREA)
            frame_gray = rgb2gray(resized_frame)
            return frame_gray
        else:
            return observation

    def savemodel(self,name="Q_main"):
        self.Q_main.save(name)


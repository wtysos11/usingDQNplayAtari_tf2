import tensorflow.keras as keras
import gym

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class NeuralNetworkBuilder:
    '''
        负责构建Q神经网络
    '''
    def __init__(self):
        '''
            初始化
        '''
    def build_conv2d(self,n_input,n_output):
        '''
        Args:
            n_input 输入数据的维数，这里应该是图像的大小
            n_output 输出数据的维数，这里应该是动作空间的大小
        Returns:
            model 返回一个Keras类型的model
        '''
        # Game-specific 游戏特化代码，如果后续要做泛化的话需要修改此处
        # 第一层应该是一个图像预处理层。使用Pong-v0时的输入为210*160*3

        # 初始化完毕后是三层Conv2D+BN(可能可以不加？)

        # 再之后是两层全连接层，确保最后的输出向量维数为action_space的大小

        # 声明模型，并返回它

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
        if name=="conv2d":
            return build_conv2d(n_input,n_output)


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

    def sampleBatch(self,batch_size = 100):
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
    def __init__(self,name="Pong-v0",networkName="conv2d",max_memory_length = 10000):
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
        Builder = NeuralNetworkBuilder()
        self.Q_main = Builder.build_network(env.observation_space.shape,env.action_space.n,name = networkName)
        self.Q_target = Builder.build_network(env.observation_space.shape,env.action_space.n,name = networkName)
        # 基础设施
        self.memoryBuffer = MemoryBuffer(max_lnegth = max_memory_length)
        # 声明超参数
        # epsilon控制系统
        # epsilon min max decay rate...
        self.epsilon_init = 0.05
        self.epsilon_min = 0.05
        self.epsilon_max = 1.0
        self.epsilon_decay_steps = 500000

        # 学习率
        self.learning_rate = 0.001
        self.discount_factor = 0.97
        # Q_target更新速率
        self.target_update_rate = 10000

    def main_process(self,episode_num = 10000,batch_size = 100):
        '''
        主过程，详情见上面的部分
        '''
        self.global_counting = 0 #额外超参数，每次训练重置，负责提供各个超参数的衰变
        step_train = 4 # 每4次行动训练一次神经网络

        episode_total_reward = [] # 用于存放每一轮的奖励，并绘制最终曲线
        
        for episode_num_counter in range(episode_num):
        #对于每一个episode
            gameDone = False
            currentEpisodeReward = 0
            #初始化环境，主要是运行obs = env.reset()
            observation = self.env.reset()
            #当游戏没有结束
            while not gameDone:
                #拿取当前环境并初始化图像（上一时刻的下一记录即为当前记录）
                observation = self.preprocessing(observation)
                #epsilon-greedy，拿取动作
                action_val = self.Q_main.predict([observation])[0] # 这里需要思考一下，我觉得应该是要上升为数组再开回来
                action = self.epsilon_greedy(action_val)
                #执行动作,获取新环境
                next_observation, reward, done, _ = self.env.step(action)
                currentEpisodeReward += reward # 更新episode奖励

                #制作元组存入记忆库中
                self.memoryBuffer.addItem([observation,action,reward,self.preprocessing(next_observation),done])

                #如果运行了指定次数且有足够多的训练数据，则开始训练
                if self.global_counting % step_train == 0 and self.memoryBuffer.getLength() >= batch_size:
                    #从记忆库拿取数据，转换成数组形式
                    obs_array,action_array,reward_array,next_obs_array,done_array = self.memoryBuffer.sampleBatch(batch_size=batch_size)
                    # 进行预处理
                    obs_array = [x for x in obs_array]
                    next_obs_array = [x for x in next_obs_array]
                    # 此处一定要使用Q_target进行计算
                    next_actionVal = self.Q_target.predict(next_obs_array) #得到Q(s')的所有动作的向量
                    # 计算Q表
                    QTable = self.Q_main.predict(obs_array)
                    # 对于每一个状态，对Q表进行更新
                    for i in range(len(obs_array)):
                        replay_action = action_array[i]
                        # Q*(s,a) = r + \gamma * Q'(s',a')
                        QTable[i][replay_action] =reward_array[i] + self.discount_factor * max(next_actionVal[i])
                    #喂给神经网络，使用MSE作为损失函数
                    Q_main.fit(obs_array,QTable)
                    #训练
                # 如果又经过了指定的时间
                if self.global_counting % self.target_update_rate == 0:
                    #Q_target更新
                    self.Q_target.set_weights(self.Q_main.get_weights())
                
                # 计数器更新
                self.global_counting += 1
            #记录episode总奖励
            episode_total_reward.append(currentEpisodeReward)
        # 训练完毕

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
        if self.networkName == "conv2d":
            return np.divide(observation,255.)
        else:
            return observation


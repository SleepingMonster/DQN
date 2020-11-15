# agent：板
from typing import (  #即有些设了默认值的可以不传参数
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):   #就是那个反弹平板模型

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,   #restore默认设为None
    ) -> None:   
        self.__action_dim = action_dim   #设置模型初始参数
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = DQN(action_dim, device).to(device)   #可以设置运行在cpu还是gpu上
        self.__target = DQN(action_dim, device).to(device)
        if restore is None:
            self.__policy.apply(DQN.init_weights)     #采用dqn初始权重
        else:
            self.__policy.load_state_dict(torch.load(restore))   #否则更新为restore中的权重
        self.__target.load_state_dict(self.__policy.state_dict())
        self.__optimizer = optim.Adam(    #优化器
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()
    
    # 采用epsilon-greedy方法选择action
    def run(self, state: TensorStack4, training: bool = False) -> int:   #运行
        """run suggests an action for the given state."""
        if training:
            #利用eps_start/eps_final这些动态衰减epsilon
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay   #这是啥意思？？貌似是为了调整学习率
            self.__eps = max(self.__eps, self.__eps_final)
        
        #有一定的概率选择使函数值最大的action，在当前网络policy_network中得到
        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        #否则随机选择action
        return self.__r.randint(0, self.__action_dim - 1)
    
    #从memory中提取state action reward next来训练网络
    def learn(self, memory: ReplayMemory, batch_size: int) -> float:     
        """learn trains the value network via TD-learning."""
        # 从replay buffer当中采样！！用于更新policy网络
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            memory.sample(batch_size)   #将所有变量转为张量。
        #使用行为网络计算值函数 Q_j
        values = self.__policy(state_batch.float()).gather(1, action_batch)     # Q_j对应有state_batch和action_batch
        
        #使用目标网络计算 Q_{j+1}并计算 expected = r_{j+1} + max(a') Q_{j+1}
        #其中(1-done_batch)用于判断是否terminal，如果是就退化到expected = r_{j+1}
        #这里相当于q-learning中的更新公式的一部分。在target网络中计算Q值。
        values_next = self.__target(next_batch.float()).max(1).values.detach()
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch
        
        #根据目标函数 （Q_j - expected)^2来梯度下降
        loss = F.smooth_l1_loss(values, expected)

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()
    
    #同步target网络和policy网络，即目标和行为网络
    def sync(self) -> None:              #
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())
    
    #保存policy network
    def save(self, path: str) -> None:         #保存结果
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)

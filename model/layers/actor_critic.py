import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        """Actor that picks a fragment for the summary in every iteration"""
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, self.action_size)

    def forward(self, state):
        """
        Args:
            state: [num_fragments, 1]
        Return:
            distribution: categorical distribution of pytorch
        """
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.linear4(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        """Critic that evaluates the Actor's choices"""
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 1)

    def forward(self, state):
        """
        Args:
            state: [num_fragments, 1]
        Return:
            value: scalar
        """
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = F.relu(self.linear4(output))
        value = self.linear5(output)
        return value

if __name__ == '__main__':

    pass

import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn import functional as F
from torch.distributions import Normal

Transition = namedtuple('Transition',
                        ('state', 'reward', 'done', 'action', 'next_state'))

class Agent:
    def __init__(self, env_specs):
        self.env_name = "Hopper-v2"
        self.n_states = 11
        self.n_actions = 3
        self.memory_size = 1e+6
        self.batch_size = 256
        self.gamma = 0.99
        self.alpha = 1
        self.lr = 3e-4
        self.action_bounds = [-1., 1]
        self.reward_scale = 5
        self.memory = Memory(memory_size=self.memory_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hypernet= None
        self.hypernetwork = WeightNet(11, 256).to(self.device)
        
        self.policy_network = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions,
                                        action_bounds=self.action_bounds, hypernet=self.hypernet).to(self.device)
        self.q_value_network1 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.q_value_network2 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.value_network = ValueNetwork(n_states=self.n_states, hypernet=self.hypernet).to(self.device)
        self.value_target_network = ValueNetwork(n_states=self.n_states).to(self.device)
        self.value_target_network.load_state_dict(self.value_network.state_dict(), strict=False)
        self.value_target_network.eval()

        self.value_loss = torch.nn.MSELoss()
        self.q_value_loss = torch.nn.MSELoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.lr)
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, reward, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.n_actions).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)

        return states, rewards, dones, actions, next_states
    
    def update(self, state, action, reward, next_state, done, timestep):
        self.store(state, reward, done, action, next_state)
        self.train()
        return
        
    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0, 0
        else:
            batch = self.memory.sample(self.batch_size)
            states, rewards, dones, actions, next_states = self.unpack(batch)

            # Calculating the value target
            hyper_weights = self.hypernetwork(states)
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states, hyper_weights)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.alpha * log_probs.detach()

            value = self.value_network(states, hyper_weights)
            value_loss = self.value_loss(value, target_value)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.reward_scale * rewards + \
                           self.gamma * self.value_target_network(next_states, hyper_weights) * (1 - dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.q_value_loss(q1, target_q)
            q2_loss = self.q_value_loss(q2, target_q)

            policy_loss = (self.alpha * log_probs - q).mean()
            
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()


            self.soft_update_target_network(self.value_network, self.value_target_network)

            return value_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()

    def act(self, states, mode="train"):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states, self.hypernetwork(states))
        return action.detach().cpu().numpy()[0]

    @staticmethod
    def soft_update_target_network(local_network, target_network, tau=0.005):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def save_weights(self, root_path):
        torch.save(self.policy_network.state_dict(), root_path + "weights.pth")

    def load_weights(self, root_path):
        self.policy_network.load_state_dict(torch.load(root_path + "weights.pth"))

    def set_to_eval_mode(self):
        self.policy_network.eval()

class Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []

    def add(self, *transition):
        self.memory.append(Transition(*transition))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        assert len(self.memory) <= self.memory_size

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)
    


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)

class LinearHyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearHyperNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.net = nn.Linear(in_features=self.input_dim, out_features=self.output_dim)
        
        init_weight(self.net)
        self.net.bias.data.zero_()
    
    def forward(self, states):
        return self.net(x)
    
class WeightNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WeightNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.left_net = nn.Linear(in_features=self.input_dim, out_features = self.output_dim)
        self.right_net = nn.Linear(in_features=self.input_dim, out_features = self.output_dim)
        init_weight(self.left_net)
        self.left_net.bias.data.zero_()
        init_weight(self.right_net)
        self.right_net.bias.data.zero_()
    def forward(self, states):
        left = self.left_net(states)
        right = self.right_net(states)
        
        return torch.einsum('bp, bq->bpq', left, right)[0]

class MultiLinearHyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLinearHyperNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden1 = nn.Linear(in_features=self.input_dim, out_features=4*self.output_dim)
        self.hidden2 = nn.Linear(in_features=4*self.output_dim, out_features=2*self.output_dim)
        self.out = nn.Linear(in_features=2*self.output_dim, out_features=self.output_dim)
        
        init_weight(self.hidden1)
        init_weight(self.hidden2)
        init_weight(self.out)
        self.hidden1.bias.data.zero_()
        self.hidden2.bias.data.zero_()
        self.out.bias.data.zero_()
    
    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.out(x)

class ValueNetwork(nn.Module):
    def __init__(self, n_states, hypernet=None, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        if hypernet != None:
            self.hidden2 = hypernet
        else:
            self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            init_weight(self.hidden2)
            self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states, weight):
        x = F.relu(self.hidden1(states))
        x = F.relu(x@weight)
        return self.value(x)


class QvalueNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, action_bounds, hypernet, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        if hypernet != None:
            self.hidden2 = hypernet
        else:
            self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            init_weight(self.hidden2)
            self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states, weight):
        x = F.relu(self.hidden1(states))
        x = F.relu(x@weight)

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states, weight):
        dist = self(states, weight)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob

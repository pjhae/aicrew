# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    
    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        return policy, value

class openai_critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n+obs_shape_n, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, action_bound , args):
        super(openai_actor, self).__init__()
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai)
        self.linear_a2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_a = nn.Linear(args.num_units_openai, action_size)

        # action rescaling
        self.action_scale = torch.FloatTensor((action_bound[1] - action_bound[0]) / 2.)
        self.action_bias = torch.FloatTensor((action_bound[1] + action_bound[0]) / 2.)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))
    
    def forward(self, input, model_original_out=False, eval=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        policy = torch.tanh(model_out)*self.action_scale + self.action_bias

        # add gaussian noise if 'NOT' evaluation mode!
        policy_noise = self.action_scale * torch.randn_like(policy, dtype=torch.float32)/100
        if eval == False:
            policy += policy_noise

        if model_original_out == True:   return model_out, policy # for model_out criterion
        return policy
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(openai_actor, self).to(device)
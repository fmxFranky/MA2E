import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

from components.masked_transformer import Base_Transformer as MT
from modules.layer.self_without_atten import Self_Without_Attention

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args,mt_pretraining = False):
        self.mt_pretraining = mt_pretraining
        super(RNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        if self.args.use_MT_mode and self.mt_pretraining is not True :
            self.fc0 = nn.Linear(self.input_shape,int(args.rnn_hidden_dim/2))
            self.fc1 = nn.Linear(self.input_shape, int(args.rnn_hidden_dim/2))
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
            
            self.mae = MT(input_shape, 1, self.args,"cuda" if self.args.use_cuda else "cpu", positional_type = self.args.positional_encoding_target)
            for param in self.mae.parameters() :
                param.requires_grad= False
            
            self.self_obs_attention = Self_Without_Attention(input_shape,1,input_shape)
            self.self_action_attention = Self_Without_Attention(1,1,1)
            
            
            self.input_mae_obs_base = th.zeros((3000,self.args.n_agents*self.args.MT_traj_length,input_shape)).to("cuda" if self.args.use_cuda else "cpu")
            self.input_mae_action_base = th.zeros((3000,self.args.n_agents*self.args.MT_traj_length,1)).to("cuda" if self.args.use_cuda else "cpu")
            #self.seq_num = th.arange(3000)
            #self.agent_location = th.arange(self.args.n_agents).repeat(1000)    
        else :
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, all_info, hidden_state=None):
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)

        if self.args.use_MT_mode is not True or self.mt_pretraining :

            x = F.relu(self.fc1(inputs), inplace=True)
            if hidden_state is not None:
                hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, hidden_state)
            q = self.fc2(h)
            return q.view(b, a, -1), h.view(b, a, -1)
        else :
            input_mae_obs, input_mae_action = self.generate_mae_input(all_info)
            mae_obs,mae_action = self.mae(1,input_mae_obs, input_mae_action, train=False)
            start = (self.args.MT_traj_length-1)*self.args.n_agents
            end = self.args.MT_traj_length*self.args.n_agents
            #index = self.seq_num[:mae_obs.shape[0]]
            #col = start+self.agent_location[:mae_obs.shape[0]]
            #mae_obs[index,col] = inputs
            #col = col-start
            mae_obs[:,start,:] = inputs
            #obs_memory = self.self_obs_attention(mae_obs[:,start:end,:],col,index)
            obs_attention_score, obs_memory = self.self_obs_attention(mae_obs[:,start:end,:])
            
            #x1 = F.relu(self.fc0(obs_memory[index,col,:].reshape((-1,self.input_shape))))
            x1 = F.relu(self.fc0(obs_memory[:,0,:].reshape((-1,self.input_shape))))
            x2 = F.relu(self.fc1(inputs.flatten(1)))
            x = th.cat((x1,x2),dim=1)
            h_in = hidden_state.reshape(-1,self.args.rnn_hidden_dim)
            h = self.rnn(x,h_in)
            if getattr(self.args, "use_layer_norm", False):
                q = self.fc2(self.layer_norm(h))
            else:
                q = self.fc2(h)

            return q.view(b, a, -1), h.view(b, a, -1)
        
    def generate_mae_input(self, all_info) :
        b,a,t,e = all_info[0].shape
        obs_all = all_info[0].reshape(-1,t,e)
        b,a,t,e = all_info[1].shape
        action_all = all_info[1] .reshape(-1,t,e)
        
        input_mae_obs = self.input_mae_obs_base[0:obs_all.shape[0],:,:].clone().detach()
        input_mae_action = self.input_mae_action_base[0:action_all.shape[0],:,:].clone().detach()
        
        input_mae_obs[:,::self.args.n_agents,:] = obs_all[:,::1,:]
        input_mae_action[:,::self.args.n_agents,:] = action_all[:,::1,:]/self.args.n_actions
    
        return input_mae_obs, input_mae_action
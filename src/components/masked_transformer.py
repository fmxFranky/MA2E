import torch
import torch.nn as nn
#from torch.masked import masked_tensor, as_masked_tensor
import random 
import math
import time 


class Positional_Encoding(nn.Module) :
    def __init__(self, max_seq_len, d_embed, device) :
        super().__init__()
        self.device = device
        self.d_embed = d_embed
        self.pe = torch.zeros(max_seq_len, self.d_embed, device = self.device)
        self.pe2 = torch.zeros(max_seq_len, self.d_embed, device = self.device)
        self.pe.requires_grad = False
        self.pe2.requires_grad = False        
        position = torch.arange(0,max_seq_len, dtype=torch.float, device = self.device).unsqueeze(1)
        
        self._2i = torch.arange(0, self.d_embed, step=2, dtype=torch.float, device = self.device)
        
        self.pe[:,0::2] = torch.sin(position/10000 **(self._2i/self.d_embed))#, device = self.device)
        self.pe[:,1::2] = torch.cos(position/10000 **(self._2i/self.d_embed))#, device = self.device)
        self.pe2[:,0::2] = torch.sin(position/10000 **(self._2i/self.d_embed))#, device = self.device)
        self.pe2[:,1::2] = torch.cos(position/10000 **(self._2i/self.d_embed))#, device = self.device)
    
    def get_pe(self, new_position,one_position=True) :        
        if one_position :
            self.pe[:,0::2] = torch.sin(new_position/10000 **(self._2i/self.d_embed))#, device = self.device)
            self.pe[:,1::2] = torch.cos(new_position/10000 **(self._2i/self.d_embed))#, device = self.device)
        else :
            self.pe2[:,0::2] = torch.sin(new_position/10000 **(self._2i/self.d_embed))#, device = self.device)
            self.pe2[:,1::2] = torch.cos(new_position/10000 **(self._2i/self.d_embed))#, device = self.device)

        
    def forward(self, x, positional_type) :
        batch_size, seq_len,_ = x.size()
        if positional_type == 'agent' or positional_type == 'time' :
            PE = self.pe[:seq_len,:]
        elif positional_type == 'both' :
            PE = torch.cat((self.pe[:int(seq_len/2),:],self.pe2[int(seq_len/2):seq_len,:]),0)
        elif positional_type == 'both2' :
            PE = self.pe[:seq_len,:] + self.pe[:seq_len,:]
        out = x + PE
        return out 
        
        

class Base_Transformer(nn.Module) :
    def __init__(self, obs_dim, action_dim, config,device,positional_type='agent') :
        super().__init__()
        self.batchsize = 6
        self.n_embd = config.MT_n_embd
        self.n_heads = config.MT_n_heads
        self.n_encoder = config.MT_n_enc_layer
        self.n_decoder = config.MT_n_dec_layer
        self.traj_length = config.MT_traj_length
        self.max_len = config.MT_traj_length * 2
        self.n_agent = config.n_agents
        self.device = device
        ## transformer
        self.transformer_model = torch.nn.Transformer(d_model=self.n_embd, nhead=self.n_heads, num_encoder_layers=self.n_encoder, num_decoder_layers=self.n_decoder,batch_first=True).to(self.device)
        #self.transformer_model = torch.nn.Transformer(d_model=self.n_embd, nhead=self.n_heads, num_encoder_layers=1, num_decoder_layers=1).to(self.device)
        #self.encoder = self.transformer_model.encoder()
        #self.decoder = self.transformer_model.decoder()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_obs_shape = (self.batchsize,self.n_agent*self.traj_length,obs_dim)
        self.output_action_shape = (self.batchsize,self.n_agent*self.traj_length,action_dim)
        ## embeding
        self.state_embed = nn.Linear(obs_dim, self.n_embd).to(self.device)
        self.action_embed = nn.Linear(action_dim, self.n_embd).to(self.device)
        self.decoder_state_embed = nn.Linear(self.n_embd,obs_dim).to(self.device)
        self.decoder_action_embed = nn.Linear(self.n_embd,action_dim).to(self.device)
        
        self.positional_type = positional_type
        self.positional_encoding = Positional_Encoding(self.max_len*self.n_agent, self.n_embd, self.device)
        self.add_new_position()
        
        self.input_obs_base = torch.zeros((2000, self.n_agent*self.traj_length,self.obs_dim)).to(self.device).type(torch.float32)
        self.input_action_base = torch.zeros((2000,self.n_agent*self.traj_length,self.action_dim)).to(self.device).type(torch.float32)           
        self.embeded_input_1_base = torch.zeros((2000,self.max_len*self.n_agent,self.n_embd)).to(self.device)            
        
        self.masked_data_base = torch.zeros((2000,2*self.n_agent*self.traj_length,self.n_embd)).to(self.device)
        self.mask_base = torch.zeros((2000,2*self.n_agent*self.traj_length)).to(self.device)
        self.agent_masking_num = torch.arange(0,self.n_agent).repeat(200)
        
        self.execution_masking = torch.zeros((self.n_agent, self.traj_length*self.n_agent*2))
        for i in range(self.n_agent) :
            self.execution_masking[i,(i%self.n_agent)*2::self.n_agent*2] = 100
            self.execution_masking[i,(i%self.n_agent)*2+1::self.n_agent*2] = 100

    def encode(self, src) :
        memory = self.transformer_model.encoder(src)
        return memory
    
    #def random_masking(self,input_data) :
        #dd
        
    def decode(self, tgt, memory) :
        out = self.transformer_model.decoder(tgt,memory)
        batch_size,_,_ = out.shape
        self.output_obs_shape = (batch_size,self.n_agent*self.traj_length,self.obs_dim)
        self.output_action_shape = (batch_size,self.n_agent*self.traj_length,self.action_dim)
        
        output_obs = self.decoder_state_embed(out[:,0::2])
        output_action = self.decoder_action_embed(out[:,1::2])
        
        return output_obs, output_action
    def masking_per_agent(self,train,ori_data,agent_num) :
        return masked_data
    
    def masking(self,masking_ratio,train,ori_data,agent_num,masking_type) :
        
        batch_size,_,_ = ori_data.shape
        #masked_data = torch.zeros(ori_data.shape).to(self.device)
        masked_data = self.masked_data_base[0:ori_data.shape[0],0:ori_data.shape[1],0:ori_data.shape[2]].clone().detach().to(self.device)
        
        if train :
            if masking_ratio == 'random' :
                candi_ratio = [0.15, 0.35, 0.5, 0.75, 0.95]
                ratio = random.choice(candi_ratio)
                mask = torch.randint(100,size=ori_data.shape[0:2]).to(self.device)
            else :
                mask = self.mask_base[0:ori_data.shape[0],0:ori_data.shape[1]].clone().detach().to(self.device)
                if masking_type != 0 :
                    for num in torch.randperm(self.n_agent-0)[:masking_type+1] :
                        mask[:,(num%self.n_agent)*2::self.n_agent*2] = 100
                        mask[:,(num%self.n_agent)*2+1::self.n_agent*2] = 100 
                else :
                    mask[:,(agent_num%self.n_agent)*2::self.n_agent*2] = 100
                    mask[:,(agent_num%self.n_agent)*2+1::self.n_agent*2] = 100 
            #mask = torch.randint(100,size=ori_data.shape[0:2]).to(self.device)
        else :
            mask = self.execution_masking.repeat(int(batch_size/self.n_agent),1)
        
        if masking_ratio == 'random' :
            mask = mask >= ratio * 100 
        else :
            mask = mask >= 100 
        masked_data[mask] = ori_data[mask]
        return masked_data 
    
    def add_new_position(self) :
        new_position = torch.zeros((self.max_len*self.n_agent,1), device = self.device)
        if self.positional_type == 'agent' :
            for i in range(self.n_agent) :
                for j in range(self.traj_length) :
                    new_position[self.n_agent*2*j+i*2,0] = i+1
                    new_position[self.n_agent*2*j+i*2+1,0] = i+1
        elif self.positional_type == 'time' :
            for i in range(self.traj_length) :
                for j in range(self.n_agent*2) :
                    new_position[i*self.n_agent*2+j,0] = i+1
        elif self.positional_type == 'both' or self.positional_type == 'both2' :
            for i in range(self.n_agent) :
                for j in range(self.traj_length) :
                    new_position[self.n_agent*2*j+i*2,0] = i+1
                    new_position[self.n_agent*2*j+i*2+1,0] = i+1
            self.positional_encoding.get_pe(new_position,one_position=False)
            for i in range(self.traj_length) :
                for j in range(self.n_agent*2) :
                    new_position[i*self.n_agent*2+j,0] = i+1
                    
        self.positional_encoding.get_pe(new_position,one_position=True)
        
    def forward(self, ratio,obs, action,train=True,agent_num=1,masking_type=0) :
        
        if train is not True :
            t1 = time.time()

            agent_num = agent_num -1 
            batch_size,_,_ = obs.shape
            
            embeded_input_1 = self.embeded_input_1_base[0:batch_size,:,:].clone().detach().to(self.device)
                        
            embeded_input_1[:,0::2] = self.state_embed(obs)
            embeded_input_1[:,1::2] = self.action_embed(action)
            
            masked_input = self.masking(ratio, train, embeded_input_1,agent_num,masking_type)
            
            positional_encoding_added_masked_input = self.positional_encoding(masked_input,self.positional_type)
            positional_encoding_added_masked_input_2 = positional_encoding_added_masked_input.clone()
            encoder_return = self.encode(positional_encoding_added_masked_input)
            out_obs,out_action = self.decode(positional_encoding_added_masked_input_2,encoder_return)
            
            return out_obs, out_action    
        else :
            batch_size,n_agent,traj_length,obs_size = obs.shape
            _,_,_,action_size = action.shape
                        
            input_obs = self.input_obs_base[0:batch_size,:,:].clone().detach()
            
            input_action = self.input_action_base[0:batch_size,:,:].clone().detach()
            
            t2 = time.time()
            
            for agent in range(n_agent) :
                input_obs[:,agent::n_agent,:] = obs[:,agent,0::1,:]
                input_action[:,agent::n_agent,:] = action[:,agent,0::1,:]
                
            embeded_input_1 = self.embeded_input_1_base[0:batch_size,:,:].clone().detach().to(self.device)
            embeded_input_1[:,0::2] = self.state_embed(input_obs)
            embeded_input_1[:,1::2] = self.action_embed(input_action)
            masked_input = self.masking(ratio,train,embeded_input_1,agent_num,masking_type)
            
            positional_encoding_added_masked_input = self.positional_encoding(masked_input,self.positional_type)
            t7 = time.time()
            
            positional_encoding_added_masked_input_2 = positional_encoding_added_masked_input.clone()
            encoder_return = self.encode(positional_encoding_added_masked_input)
            
            out_obs,out_action = self.decode(positional_encoding_added_masked_input_2,encoder_return)
            
            
            return input_obs, input_action, out_obs, out_action
            

import copy
from components.episode_buffer import EpisodeBatch
from controllers.n_controller import NMAC
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_gae_targets
import torch as th
from torch.optim import Adam
from utils.value_norm import ValueNorm

class PPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        # a trick to reuse mac
        dummy_args = copy.deepcopy(args)
        dummy_args.n_actions = 1
        self.critic = NMAC(scheme, None, dummy_args)
        self.params = list(mac.parameters()) + list(self.critic.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.last_lr = args.lr

        self.use_value_norm = getattr(self.args, "use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=self.args.device)
            
    def MT_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None, logger='',write_log=False,lr = 0.0005,batch_size=32,n_repeat=2):
        
        self.mac.agent.mae.train()
        for name,child in self.mac.agent.named_children() :
            if name == 'mae' :
                for param in child.parameters() :
                    param.requires_grad = True
        loss_fn = th.nn.MSELoss()
        opt = th.optim.SGD(self.mac.agent.mae.parameters(),lr=lr)
        
        action_all_batch = batch["action_all"][:,:-1]
        obs_all_batch = batch["obs_all"][:,:-1]
        
        obs_batch = batch["obs"][:,:-1]
        t_batch = batch["terminated"][:, :-1].float()
        #filled_batch = batch["filled"][:,:-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - t_batch[:, :-1])
                
        for i in range(len(obs_all_batch)) :
            loss_avg = 0                 
            n_batch = 0
            if self.args.masking_type == 'agent' :
                if n_repeat >= 1 : 
                    for agent in range(self.args.n_agents) :
                        except_masking = (t_batch[i] == 0).nonzero(as_tuple=True)[0]
                        input_obs, input_action, out_obs, out_action = self.mac.agent.mae('mix',obs_all_batch[i,except_masking],action_all_batch[i,except_masking]/self.args.n_actions,train=True,agent_num=agent,masking_type=0)

                        loss_1 = loss_fn(input_obs, out_obs)
                        loss_2 = loss_fn(input_action, out_action)
                        loss = loss_1 + loss_2

                        loss_avg = loss_avg + loss

                        loss = loss / len(except_masking)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        n_batch += len(except_masking)
                    
                    loss_avg = loss_avg / n_batch

                    for _ in range(n_repeat-1) :
                        if self.args.n_agents-1 != 1 :
                            seq =  th.randint(1, self.args.n_agents-1, size=(1,))
                            seq = seq.item()
                        else :
                            seq = 0 
                        except_masking = (t_batch[i] == 0).nonzero(as_tuple=True)[0]
                        input_obs, input_action, out_obs, out_action = self.mac.agent.mae('mix',obs_all_batch[i,except_masking],action_all_batch[i,except_masking]/self.args.n_actions,train=True,agent_num=agent,masking_type=seq)

                        loss_1 = loss_fn(input_obs, out_obs)
                        loss_2 = loss_fn(input_action, out_action)
                        loss = loss_1 + loss_2

                        loss_avg = loss_avg + loss

                        loss = loss / len(except_masking)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        n_batch += len(except_masking)
                        
                    if n_repeat != 1 :
                        loss_avg_2 = loss_avg / n_batch
                        if write_log :
                            logger.log_wandb("MT_all",loss_avg_2)
                    
            elif self.args.masking_type == 'random' : 
                for repeat in range(n_repeat) :
                ## batch, num_agents, trajectory, action(or obs)                    
                    except_masking = (t_batch[i] == 0).nonzero(as_tuple=True)[0]

                    input_obs, input_action, out_obs, out_action = self.mac.agent.mae('random',obs_all_batch[i,except_masking],action_all_batch[i,except_masking]/self.args.n_actions,train=True,agent_num=1)

                    loss_1 = loss_fn(input_obs, out_obs)
                    loss_2 = loss_fn(input_action, out_action)
                    loss = loss_1 + loss_2

                    loss_avg = loss_avg + loss

                    loss = loss / len(except_masking)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    n_batch += len(except_masking)

            
                loss_avg = loss_avg / n_batch
                
            if write_log :
                logger.log_wandb("MT",loss_avg)

        for name,child in self.mac.agent.named_children() :
            if name == 'mae' :
                for param in child.parameters() :
                    param.requires_grad = False

        self.mac.agent.mae.eval()
        #self.update_target_mae()

        return loss_avg.item()
        
    def update_target_mae(self) :
        for param, target_param in zip(self.mac.agent.mae.parameters(), self.target_mac.agent.mae.parameters()) :
            target_param.data.copy_(param.data)
        
  
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        
        old_probs = batch["probs"][:, :-1]
        old_probs[avail_actions == 0] = 1e-10
        old_logprob = th.log(th.gather(old_probs, dim=3, index=actions)).detach()
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        
        # targets and advantages
        with th.no_grad():
            old_values = []
            self.critic.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.critic.forward(batch, t=t)
                old_values.append(agent_outs)
            old_values = th.stack(old_values, dim=1) 

            if self.use_value_norm:
                value_shape = old_values.shape
                values = self.value_norm.denormalize(old_values.view(-1)).view(value_shape)

            advantages, targets = build_gae_targets(rewards.unsqueeze(2).repeat(1, 1, self.n_agents, 1), 
                    mask_agent, values, self.args.gamma, self.args.gae_lambda)

            if self.use_value_norm:
                targets_shape = targets.shape
                targets = targets.reshape(-1)
                self.value_norm.update(targets)
                targets = self.value_norm.normalize(targets).view(targets_shape)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        
        # PPO Loss
        for _ in range(self.args.mini_epochs):
            # Critic
            values = []
            self.critic.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.critic.forward(batch, t=t)
                values.append(agent_outs)
            values = th.stack(values, dim=1) 

            # value clip
            values_clipped = old_values[:,:-1] + (values - old_values[:,:-1]).clamp(-self.args.eps_clip,
                                                                                self.args.eps_clip)

            # 0-out the targets that came from padded data
            td_error = th.max((values - targets.detach())** 2, (values_clipped - targets.detach())** 2)
            masked_td_error = td_error * mask_agent
            critic_loss = 0.5 * masked_td_error.sum() / mask_agent.sum()

            # Actor
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.mac.forward(batch, t=t)
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1)  # Concat over time

            pi[avail_actions == 0] = 1e-10
            pi_taken = th.gather(pi, dim=3, index=actions)
            log_pi_taken = th.log(pi_taken)
            
            ratios = th.exp(log_pi_taken - old_logprob)
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            actor_loss = -(th.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()
            
            # entropy
            entropy_loss = categorical_entropy(pi).mean(-1, keepdim=True) # mean over agents
            entropy_loss[mask == 0] = 0 # fill nan
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()
            loss = actor_loss + self.args.critic_coef * critic_loss - self.args.entropy * entropy_loss / entropy_loss.item()

            # Optimise agents
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_agent.sum().item()
            self.logger.log_stat("advantage_mean", (advantages * mask_agent).sum().item() / mask_elems, t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("lr", self.last_lr, t_env)
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("target_mean", (targets * mask_agent).sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env


    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/agent_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))

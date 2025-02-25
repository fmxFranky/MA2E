import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_td_lambda_targets
from torch.distributions import Categorical
import torch as th
from torch.optim import Adam


class COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr)
        
    def save_MT(self,filename='') :
        print(filename)
        th.save(self.mac.agent.mae.state_dict(), filename)
        
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
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)

        actions = actions[:,:-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()
        
        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        advantages = (q_taken - baseline).detach()

        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()
        
        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env
        
        '''
        #dist_entropy = pi
        dist_entropy = Categorical(pi).entropy().view(-1)
        dist_entropy[mask == 0] = 0 # fill nan
        entropy_loss = (dist_entropy * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        loss = coma_loss - self.args.entropy * entropy_loss
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env
        '''

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        target_q_vals = self.target_critic(batch)[:, :]
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t)
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            targets_t = targets[:, t]

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

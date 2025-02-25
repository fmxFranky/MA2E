import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from utils.rl_utils import build_td_lambda_targets
from envs.matrix_game import print_matrix_status
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.train_t = 0
        
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
        self.update_target_mae()

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
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_back = chosen_action_qvals
        
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()
        
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

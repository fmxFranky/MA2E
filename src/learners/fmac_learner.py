import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.fmac_critic import FMACCritic
from modules.critics.lica import LICACritic
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_td_lambda_targets
from components.epsilon_schedules import DecayThenFlatSchedule
from utils.th_utils import get_parameters_num


class FMACLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = FMACCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        
        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        print('Mixer Size: ')
        print(get_parameters_num(self.critic_params))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0

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
        self.update_target_mae()

        return loss_avg.item()
        
    def update_target_mae(self) :
        for param, target_param in zip(self.mac.agent.mae.parameters(), self.target_mac.agent.mae.parameters()) :
            target_param.data.copy_(param.data)
        
  

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, off=False):
        # Get the relevant data
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Retrace Q target
        with th.no_grad():
            q1, _ = self.target_critic(batch,  batch["actions_onehot"].detach())
            target_vals = self.target_mixer(q1, batch["state"])
            
            lambd = 0 if off else self.args.lambd
            target_vals = build_td_lambda_targets(rewards, 
                    terminated, mask, target_vals, self.n_agents, self.args.gamma, lambd)

        # Train the critic
        # Current Q network forward
        q1, _ = self.critic(batch[:, :-1], actions_onehot.detach())
        q_taken = self.mixer(q1, batch["state"][:,:-1])
        critic_loss = 0.5 * ((q_taken - target_vals.detach()) * mask).pow(2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # Train the actor
        if not off:
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.mac.forward(batch, t=t)
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1)  # Concat over time b, t, a, probs

            q1, _ = self.critic(batch[:,:-1], pi)
            q = self.mixer(q1, batch["state"][:, :-1])
            pg_loss = -(q * mask).sum() / mask.sum() 

            entropy_loss = categorical_entropy(pi).mean(-1, keepdim=True) # mean over agents
            entropy_loss[mask == 0] = 0 # fill nan
            entropy_loss = (entropy_loss* mask).sum() / mask.sum()
            loss = pg_loss - self.args.entropy_coef * entropy_loss / entropy_loss.item()

            self.agent_optimiser.zero_grad()
            loss.backward()
            agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        # target_update
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # log
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm.item(), t_env)
            self.logger.log_stat("target_vals", (target_vals * mask).sum().item() / mask.sum().item(), t_env)

            if not off:
                self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
                self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
                self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
                agent_mask = mask.repeat(1, 1, self.n_agents)
                self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * agent_mask).sum().item() / agent_mask.sum().item(), t_env)
                self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        # self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda"):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.to(device=device)
        self.target_critic.to(device=device)
        if self.mixer is not None:
            self.mixer.to(device=device)
            self.target_mixer.to(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
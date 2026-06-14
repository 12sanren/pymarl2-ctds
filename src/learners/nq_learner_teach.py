import copy
import os

import torch as th
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num


class NQLearnerTeach:
    """CTDS learner: finetuned QMIX teacher TD + per-agent Q distillation to student."""

    def __init__(self, mac_t, mac_s, scheme, logger, args):
        self.args = args
        self.mac = mac_t
        self.mac_s = mac_s
        self.logger = logger
        self.teacher_only = getattr(args, "teacher_only", False)

        self.last_target_update_episode = 0
        self.device = th.device("cuda" if args.use_cuda else "cpu")
        self.params = list(mac_t.parameters())
        self.params_kd = list(mac_s.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print("Mixer Size: ")
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == "adam":
            self.optimiser = Adam(
                params=self.params,
                lr=args.lr,
                weight_decay=getattr(args, "weight_decay", 0),
            )
            self.optimiser_kd = Adam(
                params=self.params_kd,
                lr=args.lr,
                weight_decay=getattr(args, "weight_decay", 0),
            )
        else:
            self.optimiser = RMSprop(
                params=self.params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
            self.optimiser_kd = RMSprop(
                params=self.params_kd,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )

        self.target_mac = copy.deepcopy(mac_t)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_out.append(self.mac.forward(batch, t=t))
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_mac_out.append(self.target_mac.forward(batch, t=t))
            target_mac_out = th.stack(target_mac_out, dim=1)

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, "q_lambda", False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])
                targets = build_q_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    qvals,
                    self.args.gamma,
                    self.args.td_lambda,
                )
            else:
                targets = build_td_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    self.args.td_lambda,
                )

        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = chosen_action_qvals - targets.detach()
        td_error2 = 0.5 * td_error.pow(2)
        mask_td = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask_td
        loss_td = masked_td_error.sum() / mask_td.sum()

        self.optimiser.zero_grad()
        loss_td.backward()
        grad_norm_td = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        loss_kd = th.tensor(0.0, device=loss_td.device)
        grad_norm_kd = 0.0
        masked_kd_error = None

        if not self.teacher_only:
            self.mac_s.agent.train()
            mac_s_out = []
            self.mac_s.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                mac_s_out.append(self.mac_s.forward(batch, t=t))
            mac_s_out = th.stack(mac_s_out, dim=1)

            kd_error = (mac_s_out - mac_out.clone().detach())[:, :-1]
            kd_error = (kd_error**2).sum(dim=(2, 3))
            mask_kd = mask.squeeze(-1)
            masked_kd_error = kd_error * mask_kd
            loss_kd = masked_kd_error.sum() / mask_kd.sum()

            self.optimiser_kd.zero_grad()
            loss_kd.backward()
            grad_norm_kd = th.nn.utils.clip_grad_norm_(
                self.params_kd, self.args.grad_norm_clip
            )
            self.optimiser_kd.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("grad_norm_td", grad_norm_td, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs",
                masked_td_error.abs().sum().item() / mask_elems,
                t_env,
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            if not self.teacher_only:
                self.logger.log_stat("loss_kd", loss_kd.item(), t_env)
                self.logger.log_stat("grad_norm_kd", grad_norm_kd, t_env)
                self.logger.log_stat(
                    "kd_error_abs",
                    masked_kd_error.abs().sum().item() / mask_elems,
                    t_env,
                )
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.mac_s.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if not self.teacher_only:
            s_save_path = os.path.join(path, "student")
            os.makedirs(s_save_path, exist_ok=True)
            self.mac_s.save_models(s_save_path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt_td.th".format(path))
        if not self.teacher_only:
            th.save(self.optimiser_kd.state_dict(), "{}/opt_kd.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt_td.th".format(path), map_location=lambda storage, loc: storage)
        )
        if not self.teacher_only:
            s_save_path = os.path.join(path, "student")
            self.mac_s.load_models(s_save_path)
            self.optimiser_kd.load_state_dict(
                th.load(
                    "{}/opt_kd.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )

from functools import partial

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class EpisodeRunnerTeach:
    """CTDS episode runner: teacher collects data; separate teacher/student tests."""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.train_stats = {}
        self.test_returns_teacher = []
        self.test_stats_teacher = {}
        self.test_returns_student = []
        self.test_stats_student = {}
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac_t, mac_s):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac_t = mac_t
        self.mac_s = mac_s

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def _teacher_obs(self):
        if hasattr(self.env, "get_obs_teacher"):
            return self.env.get_obs_teacher()
        if hasattr(self.env, "get_obs_kaitu"):
            return self.env.get_obs_kaitu()
        return self.env.get_obs()

    def _share_obs(self):
        if not hasattr(self.env, "get_share_obs"):
            return self._teacher_obs()
        share = self.env.get_share_obs()
        if getattr(self.env, "teacher_add_local_obs", False):
            local = self.env.get_obs()
            return [
                np.concatenate([share[i], local[i]]).astype(np.float32)
                for i in range(len(share))
            ]
        return share

    def _pre_transition_data(self):
        return {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "obs_teacher": [self._teacher_obs()],
            "share_obs": [self._share_obs()],
        }

    def run(self, test_mode=False):
        assert not test_mode, "Use test() for evaluation in CTDS mode"
        self.reset()

        terminated = False
        episode_return = 0
        self.mac_t.init_hidden(batch_size=self.batch_size)

        while not terminated:
            self.batch.update(self._pre_transition_data(), ts=self.t)
            actions = self.mac_t.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=False
            )
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        self.batch.update(self._pre_transition_data(), ts=self.t)
        actions = self.mac_t.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=False
        )
        self.batch.update({"actions": actions}, ts=self.t)

        self._update_train_stats(episode_return, env_info, prefix="")
        return self.batch

    def test(self):
        self.student_test()
        self.teacher_test()

    def student_test(self):
        self._run_eval(self.mac_s, self.test_returns_student, self.test_stats_student, "test_stu_")

    def teacher_test(self):
        self._run_eval(self.mac_t, self.test_returns_teacher, self.test_stats_teacher, "test_tea_")

    def _run_eval(self, mac, cur_returns, cur_stats, log_prefix):
        self.reset()
        terminated = False
        episode_return = 0
        mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            self.batch.update(self._pre_transition_data(), ts=self.t)
            actions = mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True
            )
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        self.batch.update(self._pre_transition_data(), ts=self.t)
        actions = mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats.update(
            {k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)}
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        cur_returns.append(episode_return)

        target_len = self.args.test_nepisode
        if log_prefix == "test_stu_":
            if len(self.test_returns_student) == target_len:
                self._log(cur_returns, cur_stats, log_prefix)
        elif len(self.test_returns_teacher) == target_len:
            self._log(cur_returns, cur_stats, log_prefix)

    def _update_train_stats(self, episode_return, env_info, prefix):
        cur_stats = self.train_stats
        cur_returns = self.train_returns
        cur_stats.update(
            {k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)}
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        self.t_env += self.t
        cur_returns.append(episode_return)

        if self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, prefix)
            if hasattr(self.mac_t.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac_t.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

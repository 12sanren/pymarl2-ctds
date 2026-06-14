import torch as th


class SmacObsParser:
    """Parse flat SMAC obs/state vectors into structured tokens for CTDS+ teacher."""

    def __init__(self, obs_layout, n_agents, device=None):
        self.n_agents = n_agents
        self.move_feats = obs_layout["move_feats"]
        self.n_enemies = obs_layout["n_enemies"]
        self.n_enemy_feats = obs_layout["n_enemy_feats"]
        self.n_allies = obs_layout["n_allies"]
        self.n_ally_feats = obs_layout["n_ally_feats"]
        self.own_feats = obs_layout["own_feats"]
        self.ally_state_dim = obs_layout["ally_state_dim"]
        self.enemy_state_dim = obs_layout["enemy_state_dim"]
        self.state_last_action = obs_layout.get("state_last_action", False)
        self.state_timestep_number = obs_layout.get("state_timestep_number", False)
        self.n_actions = obs_layout.get("n_actions", 0)
        self.device = device

    def parse_obs(self, obs_teacher):
        bs, na, _ = obs_teacher.shape
        offset = 0

        move = obs_teacher[:, :, offset : offset + self.move_feats]
        offset += self.move_feats

        enemy_flat = obs_teacher[
            :, :, offset : offset + self.n_enemies * self.n_enemy_feats
        ]
        enemy = enemy_flat.view(bs, na, self.n_enemies, self.n_enemy_feats)
        offset += self.n_enemies * self.n_enemy_feats

        ally_flat = obs_teacher[
            :, :, offset : offset + self.n_allies * self.n_ally_feats
        ]
        ally = ally_flat.view(bs, na, self.n_allies, self.n_ally_feats)
        offset += self.n_allies * self.n_ally_feats

        own = obs_teacher[:, :, offset : offset + self.own_feats]
        self_feats = th.cat([move, own], dim=-1)

        enemy_dist = (
            enemy[..., 1]
            if self.n_enemy_feats > 1
            else th.zeros(bs, na, self.n_enemies, device=obs_teacher.device)
        )
        ally_dist = (
            ally[..., 1]
            if self.n_ally_feats > 1
            else th.zeros(bs, na, self.n_allies, device=obs_teacher.device)
        )

        return {
            "self_feats": self_feats,
            "enemy_feats": enemy,
            "ally_feats": ally,
            "enemy_dist": enemy_dist,
            "ally_dist": ally_dist,
        }

    def _ally_state_size(self):
        return self.n_agents * self.ally_state_dim

    def _enemy_state_size(self):
        return self.n_enemies * self.enemy_state_dim

    def parse_state(self, state):
        bs = state.shape[0]
        offset = 0

        ally_flat = state[:, offset : offset + self._ally_state_size()]
        ally_state = ally_flat.view(bs, self.n_agents, self.ally_state_dim)
        offset += self._ally_state_size()

        enemy_flat = state[:, offset : offset + self._enemy_state_size()]
        enemy_state = enemy_flat.view(bs, self.n_enemies, self.enemy_state_dim)

        return ally_state, enemy_state

    def extract_state_unique(self, state):
        ally_state, _ = self.parse_state(state)
        return ally_state[:, :, 1:2]

    @staticmethod
    def layout_from_env_info(env_info):
        layout = dict(env_info["obs_layout"])
        layout["n_actions"] = env_info["n_actions"]
        return layout

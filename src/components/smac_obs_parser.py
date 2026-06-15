import torch as th


class SmacObsParser:
    """Parse SMAC obs / MAPPO share_obs vectors into structured teacher tokens."""

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

        self.share_move_feats = obs_layout["share_move_feats"]
        self.share_n_enemies = obs_layout["share_n_enemies"]
        self.share_n_enemy_feats = obs_layout["share_n_enemy_feats"]
        self.share_n_allies = obs_layout["share_n_allies"]
        self.share_n_ally_feats = obs_layout["share_n_ally_feats"]
        self.share_own_feats = obs_layout["share_own_feats"]
        self.share_agent_id_feats = obs_layout.get("share_agent_id_feats", 0)
        self.share_timestep_feats = obs_layout.get("share_timestep_feats", 0)
        self.teacher_add_local_obs = obs_layout.get("teacher_add_local_obs", False)

        self.device = device

    def parse_obs(self, obs):
        bs, na, _ = obs.shape
        offset = 0

        move = obs[:, :, offset : offset + self.move_feats]
        offset += self.move_feats

        enemy_flat = obs[:, :, offset : offset + self.n_enemies * self.n_enemy_feats]
        enemy = enemy_flat.view(bs, na, self.n_enemies, self.n_enemy_feats)
        offset += self.n_enemies * self.n_enemy_feats

        ally_flat = obs[:, :, offset : offset + self.n_allies * self.n_ally_feats]
        ally = ally_flat.view(bs, na, self.n_allies, self.n_ally_feats)
        offset += self.n_allies * self.n_ally_feats

        own = obs[:, :, offset : offset + self.own_feats]
        self_feats = th.cat([move, own], dim=-1)

        enemy_dist = (
            enemy[..., 1]
            if self.n_enemy_feats > 1
            else th.zeros(bs, na, self.n_enemies, device=obs.device)
        )
        ally_dist = (
            ally[..., 1]
            if self.n_ally_feats > 1
            else th.zeros(bs, na, self.n_allies, device=obs.device)
        )

        return {
            "self_feats": self_feats,
            "enemy_feats": enemy,
            "ally_feats": ally,
            "enemy_dist": enemy_dist,
            "ally_dist": ally_dist,
        }

    def parse_share_obs(self, share_obs):
        """Parse MAPPO get_state_agent() layout: ally|enemy|move|own|agent_id|timestep|local."""
        bs, na, _ = share_obs.shape
        offset = 0

        ally_flat = share_obs[
            :, :, offset : offset + self.share_n_allies * self.share_n_ally_feats
        ]
        ally = ally_flat.view(bs, na, self.share_n_allies, self.share_n_ally_feats)
        offset += self.share_n_allies * self.share_n_ally_feats

        enemy_flat = share_obs[
            :, :, offset : offset + self.share_n_enemies * self.share_n_enemy_feats
        ]
        enemy = enemy_flat.view(bs, na, self.share_n_enemies, self.share_n_enemy_feats)
        offset += self.share_n_enemies * self.share_n_enemy_feats

        move = share_obs[:, :, offset : offset + self.share_move_feats]
        offset += self.share_move_feats

        own = share_obs[:, :, offset : offset + self.share_own_feats]
        offset += self.share_own_feats
        self_feats = th.cat([move, own], dim=-1)

        if self.share_agent_id_feats:
            offset += self.share_agent_id_feats
        if self.share_timestep_feats:
            offset += self.share_timestep_feats

        local = None
        if self.teacher_add_local_obs:
            local_obs = share_obs[:, :, offset:]
            local = self.parse_obs(local_obs)

        enemy_dist = (
            enemy[..., 1]
            if self.share_n_enemy_feats > 1
            else th.zeros(bs, na, self.share_n_enemies, device=share_obs.device)
        )
        ally_dist = (
            ally[..., 1]
            if self.share_n_ally_feats > 1
            else th.zeros(bs, na, self.share_n_allies, device=share_obs.device)
        )

        return {
            "self_feats": self_feats,
            "enemy_feats": enemy,
            "ally_feats": ally,
            "enemy_dist": enemy_dist,
            "ally_dist": ally_dist,
            "local": local,
        }

    @staticmethod
    def layout_from_env_info(env_info):
        layout = dict(env_info["obs_layout"])
        layout["n_actions"] = env_info["n_actions"]
        return layout

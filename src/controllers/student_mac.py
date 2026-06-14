from .n_controller import NMAC
from modules.agents import REGISTRY as agent_REGISTRY


class StudentMAC(NMAC):
    """MAC for decentralised student: uses standard local obs."""

    def _build_agents(self, input_shape):
        agent_key = getattr(self.args, "student_agent", self.args.agent)
        self.agent = agent_REGISTRY[agent_key](input_shape, self.args)

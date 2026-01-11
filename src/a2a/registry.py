import logging
from typing import Dict, Any

class AgentRegistry:
    """
    Registry to manage decentralized ShopAgent instances.
    """
    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self.logger = logging.getLogger("agent_registry")

    def register(self, shop_id: str, agent_instance: Any):
        self._agents[shop_id] = agent_instance
        self.logger.info(f"Registered shop: {shop_id}")

    def get_all_agents(self) -> Dict[str, Any]:
        """Returns the dictionary of agents for broadcasting queries."""
        return self._agents
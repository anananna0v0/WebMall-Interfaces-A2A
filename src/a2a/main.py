import logging
from a2a.config import WEBMALL_SHOPS
from a2a.registry import AgentRegistry
from a2a.agents.buyer_agent import BuyerAgent
from a2a.agents.shop_agent import ShopAgent

# Set up logging for the system initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("a2a_main")

def initialize_system():
    """
    Initializes the decentralized A2A system by creating ShopAgents 
    and registering them with the AgentRegistry.
    """
    registry = AgentRegistry()

    # Iterate through the shops defined in config.py
    for shop_id, shop_info in WEBMALL_SHOPS.items():
        try:
            # Instantiate ShopAgent without the 'config' argument
            # The new ShopAgent handles its own service initialization internally
            agent = ShopAgent(
                shop_id=shop_id,
                index_name=shop_info["index_name"]
            )
            
            # Register the agent in the decentralized registry
            registry.register(shop_id, agent)
            logger.info(f"Initialized and registered: {shop_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize shop {shop_id}: {str(e)}")

    # Initialize the BuyerAgent with the completed registry
    buyer_agent = BuyerAgent(registry)
    
    return buyer_agent

if __name__ == "__main__":
    # Test execution for a single task
    buyer = initialize_system()
    test_instruction = "Find the cheapest AMD Ryzen 9 5900X"
    result = buyer.execute_procurement_task(test_instruction)
    print(result)
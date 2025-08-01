import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, default="./resources/configs/mnist/server_fedasync.yaml")
args = argparser.parse_args()


server_agent_config = OmegaConf.load(args.config)
server_agent = ServerAgent(server_agent_config=server_agent_config)
communicator = GRPCServerCommunicator(
    server_agent,
    logger=server_agent.logger,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)
#print(server_agent_config.server_configs.alpha)
server_agent_config.server_configs.comm_configs.grpc_configs["server_uri"] = "192.168.151.6:50051"

server_agent_config.server_configs.comm_configs.grpc_configs["server_uri"] = "0.0.0.0:50051"
# 
print(server_agent_config.server_configs.comm_configs.grpc_configs["server_uri"])
serve(
    communicator,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)


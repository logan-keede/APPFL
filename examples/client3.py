from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator

client_agent_config = OmegaConf.load('./resources/configs/mnist/client_3.yaml')
client_agent_config.comm_configs.grpc_configs["server_uri"] = "0.0.0.0:50051"
print(client_agent_config.comm_configs.grpc_configs)
client_agent = ClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id = client_agent.get_id(),
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

init_global_model = client_communicator.get_global_model(init_model=True)
client_agent.load_parameters(init_global_model)

# Send the number of local data to the server
sample_size = client_agent.get_sample_size()
client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)
print(client_agent_config.comm_configs.grpc_configs["server_uri"])

while True:
    client_agent.train()
    local_model = client_agent.get_parameters()
    if isinstance(local_model, tuple):
        local_model, metadata = local_model[0], local_model[1]
    else:
        metadata = {}
    new_global_model, metadata = client_communicator.update_global_model(local_model, **metadata)
    if metadata['status'] == 'DONE':
        break
    if 'local_steps' in metadata:
        client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']
    client_agent.load_parameters(new_global_model)
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import FedAsyncAggregator
from typing import Union, Dict, OrderedDict, Any, Optional


class FedAsyncMAggregator(FedAsyncAggregator):
    """
    FedAvgM Aggregator class for Federated Learning.
    For more details, check paper `Measuring the effects of non-identical data distribution for federated visual classification`
    at https://arxiv.org/pdf/1909.06335.pdf

    Required aggregator_configs fields:
        - server_momentum_param_1: `beta` in the paper
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        super().__init__(model, aggregator_configs, logger)
        self.v_vector = {}

    def compute_steps(
        self, client_id: Union[str, int],
        local_model: Union[Dict, OrderedDict],
    ):
        """
        Compute the changes to the global model after the aggregation.
        """
        super().compute_steps(client_id, local_model)
        if len(self.v_vector) == 0:
            for name in self.step:
                self.v_vector[name] = torch.zeros_like(self.step[name])

        for name in self.step:
            self.v_vector[name] = (
                self.aggregator_configs.server_momentum_param_1 * self.v_vector[name]
                + self.step[name]
            )
            self.step[name] = self.v_vector[name]
            # print(self.step[name])
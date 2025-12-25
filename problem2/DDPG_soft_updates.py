# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


import torch.nn as nn

def soft_updates(network: nn.Module,
                 target_network: nn.Module,
                 tau: float) -> nn.Module:
    """ Performs a soft copy of the network's parameters to the target
        network's parameter

        Args:
            network (nn.Module): neural network from which we want to copy the
                parameters
            target_network (nn.Module): network that is being updated
            tau (float): time constant that defines the update speed in (0,1)

        Returns:
            target_network (nn.Module): the target network

    """
    tgt_state = target_network.state_dict()
    for k, v in network.state_dict().items():
        tgt_state[k] = (1 - tau)  * tgt_state[k]  + tau * v
    target_network.load_state_dict(tgt_state)
    return target_network

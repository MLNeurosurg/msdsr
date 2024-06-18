"""Factory function to instantiate objects"""
from typing import Dict, Union


def instantiate_loss(cf: Dict[str, Union[str, Dict]]):
    from msdsr.ldm.modules.losses.contperceptual import LPIPSWithDiscriminator
    from msdsr.ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
    from torch.nn import Identity
    losses = {
        "LPIPSWithDiscriminator": LPIPSWithDiscriminator,
        "VQLPIPSWithDiscriminator": VQLPIPSWithDiscriminator,
        "Identity": Identity
    }
    return losses[cf["which"]](**cf["params"])


def instantiate_lr_scheduler(cf: Dict[str, Union[str, Dict]]):
    from msdsr.ldm.lr_scheduler import (LambdaWarmUpCosineScheduler,
                                        LambdaWarmUpCosineScheduler2,
                                        LambdaLinearScheduler)
    lrs = {
        "LambdaWarmUpCosineScheduler": LambdaWarmUpCosineScheduler,
        "LambdaWarmUpCosineScheduler2": LambdaWarmUpCosineScheduler2,
        "LambdaLinearScheduler": LambdaLinearScheduler
    }
    return lrs[cf["which"]](**cf["params"])


def instantiate_diffusion_model(cf: Dict[str, Union[str, Dict]]):
    from msdsr.ldm.modules.diffusionmodules.openaimodel import UNetModel

    models = {"UNetModel": UNetModel}
    return models[cf["which"]](**cf["params"])


def instantiate_first_stage_models(cf: Dict[str, Union[str, Dict]]):
    from msdsr.ldm.models.autoencoder import VQModelInterface, IdentityFirstStage
    models = {
        "VQModelInterface": VQModelInterface,
        "IdentityFirstStage": IdentityFirstStage
    }
    return models[cf["which"]](**cf["params"])


def instantiate_cond_model(cf: Dict[str, Union[str, Dict]]):
    raise NotImplementedError()

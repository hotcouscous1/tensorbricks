from __init__ import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_divisible(
        value: int or float,
        divisor: int = 8,
        round_bias: float = 0.9):

    """
    __version__ = 1.0
    __date__ = Mar 7, 2022
    """
    round_value = max(divisor, int(value + divisor / 2) // divisor * divisor)

    assert 0 < round_bias < 1
    if round_value < round_bias * value:
        round_value += divisor

    return round_value


def round_width(
        width: int or float,
        divisor: int = 8,
        round_bias: float = 0.9):

    return make_divisible(width, divisor, round_bias)


def round_depth(depth: int):
    return math.ceil(depth)



def stochastic_depth(
        input: Tensor,
        survival_prob: float,
        training: bool):

    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1603.09382

    It is to 'randomly removing a substantial fraction of layers independently for each sample or mini-batch',
    according to the paper.
    """

    if not training:
        raise RuntimeError('only while training, drop connect can be applied')

    batch_size = input.shape[0]
    random_mask = survival_prob + torch.rand([batch_size, 1, 1, 1], device=input.device)
    binary_mask = torch.floor(random_mask)

    output = input / survival_prob * binary_mask
    return output



def get_survival_probs(
        num_block_list:list,
        last_survival_prob: float):

    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1603.09382

    Given the last prob, it returns linearly declining probabilities for whole model.
    """

    num_blocks = sum(num_block_list[1:])

    survival_probs = []
    for num, end in zip(num_block_list[1:], itertools.accumulate(num_block_list[1:])):
        if last_survival_prob:
            survival_probs.append(
                1 - (torch.Tensor(range(end-num, end)) / num_blocks) * (1 - last_survival_prob))
        else:
            survival_probs.append([None for _ in range(end-num, end)])

    return survival_probs



def load_pretrained(
        model: nn.Module,
        ckpt_name: str,
        strict: bool = True,
        batch_eps: float = None,
        batch_momentum: float = None):

    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    'checkpoints' dictionary is in __init__.py
    """

    if ckpt_name not in checkpoints:
        raise ValueError('<Sorry, checkpoints for ' + ckpt_name + ' is not ready>')

    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoints[ckpt_name], map_location=device), strict)
    print('<All keys matched successfully>')

    if batch_eps or batch_momentum:
        batch_params(model, batch_eps, batch_momentum)



def batch_params(
        module: nn.Module,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True):

    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Change parameters of every nn.BatchNorm2d in the module
    """

    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = eps
            m.momentum = momentum

            if not affine:
                m.affine = affine

            if not track_running_stats:
                m.track_running_stats = track_running_stats

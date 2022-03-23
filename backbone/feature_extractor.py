from __init__ import *


class FeatureExtractor(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    Created by hotcouscous1.

    This module is to extract features from specific stages, blocks or layers,
    instead of adding another forward-method to existing backbone module.

    'backbone' receives not a class, but an instance of backbone module, as an argument.
    'stages' contains stage-names, called by backbone.named_modules.

    By register_forward_hook, extraction hooks are called after the forward computation of each module.
    """

    def __init__(self,
                 backbone: nn.Module,
                 stages: List[str],
                 return_last: bool = False):

        super(FeatureExtractor, self).__init__()

        self.model, self.stages, self.last = backbone, stages, return_last
        self.features = []

        if backbone.widths:
            self.widths = backbone.widths

        for name, module in self.model.named_modules():
            if name in self.stages:
                module.register_forward_hook(self.extract())


    def extract(self):
        def _extract(module, f_in, f_out):
            self.features.append(f_out)

        return _extract


    def forward(self, input):
        self.features.clear()

        if not self.last:
            _ = self.model(input)
            return self.features
        else:
            out = self.model(input)
            return self.features, out

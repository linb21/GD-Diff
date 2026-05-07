import torch

from models.PatchTST import PatchTST_architecture
from models.TimeMixer.timemixer import timemixer as TimeMixerExperiment


class patchtst(TimeMixerExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.args.result_txt_name = "result_patchtst.txt"

    def _build_model(self):
        return PatchTST_architecture.Model(self.args).float().to(torch.device(self.args.device))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetLiTSTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 10

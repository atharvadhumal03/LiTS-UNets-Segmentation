from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetSmokeTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 3
        self.save_every = 1

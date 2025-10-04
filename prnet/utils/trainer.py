from tqdm import tqdm

from prnet.utils.trainer_base import TrainerBase
from prnet.dataset.facealign_dataset import FaceAlignDataset
from prnet.utils.plotters import plt_preds

class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.debug_plot_every = 10
        self.face_align = FaceAlignDataset(cfg=config, mode="train", logger=logger)

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            if epoch % self.config["OPTIM"]["save_ckpt_every"] == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (img, labels) in pbar:
            img = img.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(img)
            loss, loss_dict = self.loss_fn(preds, labels)
            self.write_dict_to_tb(loss_dict, self.total_iters, prefix="train")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.total_iters += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        pbar.close()

    def evaluate_model(self):
        self.model.eval()

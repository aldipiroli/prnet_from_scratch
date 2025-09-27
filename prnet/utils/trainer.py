import copy

import torch
from tqdm import tqdm

from moco.utils.augmentations import MoCoAugmentations
from moco.utils.trainer_base import TrainerBase


class MoCoKeyQueue:
    def __init__(self, max_batches, batch_size):
        self.max_batches = max_batches
        self.batch_size = batch_size
        self.q = []

    def insert_batch(self, x):
        assert x.shape[0] == self.batch_size

        self.q.extend(x)
        if self.queue_size > self.max_batches:
            self.remove_last_batch()

    def remove_last_batch(self):
        self.q = self.q[self.batch_size :]

    def get_tensor(self):
        return torch.stack(self.q, 0)

    @property
    def queue_size(self):
        if len(self.q) == 0:
            return 0

        q_tensor = torch.stack(self.q, 0)
        return q_tensor.shape[0] // self.batch_size


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.n_classes = int(self.config["DATA"]["n_classes"])
        self.debug_plot_every = 10
        self.moco_augm = MoCoAugmentations(config)
        self.k_queue = MoCoKeyQueue(
            max_batches=config["MODEL"]["moco_queue"]["max_batches"],
            batch_size=config["DATA"]["batch_size"],
        )
        self.temp = config["MODEL"]["temperature"]
        self.m = config["MODEL"]["momentum"]

    def train(self):
        self.logger.info("Started training..")
        self.initialize_encoders()

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            if epoch % self.config["OPTIM"]["save_ckpt_every"] == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        self.f_q.train()
        self.f_k.eval()

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (x) in pbar:
            x = x.to(self.device)
            # augment
            x_q = self.moco_augm.augment(x)
            x_k = self.moco_augm.augment(x)

            # encode
            q = self.f_q(x_q)  # (B, C)
            k = self.f_k(x_k).detach()  # (B, C)

            if self.k_queue.queue_size == 0:  # handle first batch
                self.k_queue.insert_batch(k)
                continue

            # compute cos similarity
            l_pos = torch.bmm(q.unsqueeze(-1).permute(0, 2, 1), k.unsqueeze(-1)).squeeze(
                -1
            )  # (B, 1, C) x (B, C, 1) -> (B, 1)
            l_neg = torch.mm(q, self.k_queue.get_tensor().permute(1, 0))  # (B, C) x (K, C) -> (B, K)
            logits = torch.cat((l_pos, l_neg), -1) / self.temp  # (B, K+1)

            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
            loss, loss_dict = self.loss_fn(logits, labels)
            self.write_dict_to_tb(loss_dict, self.total_iters, prefix="train")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_k_encoder()
            self.k_queue.insert_batch(k)

            self.total_iters += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        pbar.close()

    def initialize_encoders(self):
        self.f_q = self.model
        self.f_k = copy.deepcopy(self.model)
        self.f_k.eval()

    def update_k_encoder(self):
        with torch.no_grad():
            for k_param, q_param in zip(self.f_k.parameters(), self.f_q.parameters()):
                k_param.data.copy_(self.m * k_param.data + (1 - self.m) * q_param.data)

    def evaluate_model(self):
        self.model.eval()
        for x in self.val_loader:
            x = x.to(self.device)

            # Query / Positive
            a = x[0:1]
            q = self.moco_augm.augment(a)
            kp = self.moco_augm.augment(a)
            embed_q = self.model(q)
            embed_kp = self.model(kp)

            # Negative
            b = x[1:2]
            kn = self.moco_augm.augment(b)
            embed_kn = self.model(kn)

            # Compute Similarities
            sim_q_kp = torch.bmm(embed_q.unsqueeze(-1).permute(0, 2, 1), embed_kp.unsqueeze(-1)).squeeze(-1)
            sim_q_kn = torch.bmm(embed_q.unsqueeze(-1).permute(0, 2, 1), embed_kn.unsqueeze(-1)).squeeze(-1)
            self.logger.info(f"Cos sim: q*kp {sim_q_kp.item()}, Cos sim: q*kn {sim_q_kn.item()}")

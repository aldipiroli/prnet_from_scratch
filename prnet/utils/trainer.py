from tqdm import tqdm

from prnet.utils.trainer_base import TrainerBase
from prnet.dataset.facealign_dataset import FaceAlignDataset
from prnet.utils import plotters
from pathlib import Path
from PIL import Image
import os 
import numpy as np
import shutil

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
        for n_iter, (img, uv_position, uv_displacement, uv_default_position, idx) in pbar:
            img = img.to(self.device)
            uv_displacement = uv_displacement.to(self.device)

            pred_coords = self.model(img)
            loss, loss_dict = self.loss_fn(pred_coords, uv_displacement)
            self.write_dict_to_tb(loss_dict, self.total_iters_train, prefix="train")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.total_iters_train += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        pbar.close()

    def evaluate_model(self, plot_preds=False):
        self.model.eval()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for n_iter, (img, uv_position, uv_displacement, uv_default_position,idx) in pbar:
            img = img.to(self.device)
            uv_displacement = uv_displacement.to(self.device)

            pred_coords = self.model(img)
            loss, loss_dict = self.loss_fn(pred_coords, uv_displacement)
            self.write_dict_to_tb(loss_dict, self.total_iters_val, prefix="val")

            self.total_iters_val += 1
            if plot_preds:
                post_processing(
                    img,
                    pred_coords,
                    uv_default_position,
                    uv_position,
                    idx=idx,
                    save_path=Path(self.config["IMG_OUT_DIR"]) / str(n_iter).zfill(3),
                )
            pbar.set_postfix(
                {
                    "mode": "val",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )

        pbar.close()


def post_processing(img_, preds_, default_coords_, gt_coords_, idx, batch_id=0, save_path="tmp/"):
    os.makedirs(save_path, exist_ok=True)
    img = img_[batch_id].permute(1, 2, 0).detach().cpu().numpy()
    preds = preds_[batch_id].permute(1, 2, 0).detach().cpu().numpy()
    default_coords = default_coords_[batch_id].permute(1, 2, 0).detach().cpu().numpy()
    gt_coords = gt_coords_[batch_id].permute(1, 2, 0).detach().cpu().numpy()

    gt_coords = gt_coords.reshape(-1, 3)
    full_pred = default_coords + preds
    full_pred = full_pred.reshape(-1, 3)

    # save img
    img = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    img.save(save_path / "img.png")

    # render
    initial_cam = [
        (2.615775456519061, 0.3396522774365594, 1.0064318838263264),
        (0.4160606927031121, 0.4946040892903853, 0.6136833386173057),
        (-0.08516130783292337, -0.9927075551045604, 0.08531858933733204),
    ]
    plotters.create_rotation_video(
        plot_func=plotters.plot_multiple_3d,
        output_path=save_path / "pred_gt.gif",
        cloud_list = [full_pred, gt_coords],
        point_size=10,
        start_camera_position=initial_cam,
    )
    plotters.create_rotation_video(
        plot_func=plotters.plot_3d_error_color,
        output_path=save_path / "error_rotation.gif",
        pred=full_pred,
        gt=gt_coords,
        point_size=10,
        cmap_name="coolwarm",
        start_camera_position=initial_cam,
    )

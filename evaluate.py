import argparse

from prnet.dataset.proprocessed_facealign_dataset import PreprocessedFaceAlignDataset
from prnet.model.loss import FaceLoss
from prnet.model.prnet import PRNet
from prnet.utils.misc import get_logger, load_config, make_artifacts_dirs
from prnet.utils.trainer import Trainer


def train(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = PreprocessedFaceAlignDataset(cfg=config, mode="train", logger=logger)
    val_dataset = PreprocessedFaceAlignDataset(cfg=config, mode="val", logger=logger)

    model = PRNet(config)
    trainer.set_model(model)
    trainer.load_checkpoint(args.ckpt)

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(FaceLoss(config, logger))

    trainer.evaluate_model(plot_preds=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/prnet_config.yaml", help="Config path")
    parser.add_argument("ckpt", type=str)
    args = parser.parse_args()
    train(args)

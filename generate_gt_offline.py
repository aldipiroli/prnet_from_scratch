from prnet.dataset.facealign_dataset import FaceAlignDataset
from prnet.utils.misc import load_config, get_logger, make_artifacts_dirs
import argparse
import os
import pickle
from tqdm import tqdm


def check_already_processed(out_dir, original_files):
    files = os.listdir(out_dir)
    files = list(set([os.path.splitext(file)[0] for file in files]))
    remaining_files = list(set(original_files) - set(files))
    return remaining_files


def generate_offline_gt(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    train_dataset = FaceAlignDataset(cfg=config, mode="train", logger=logger)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    remaining_files = check_already_processed(args.out_dir, train_dataset.files)
    logger.info(f"Total files {len(train_dataset.files)}, remaining {len(remaining_files)}")
    train_dataset.files = remaining_files

    total = len(train_dataset)
    for img, gt, filename in tqdm(train_dataset, total=total, desc="Generating GT"):
        data = {"img": img, "gt": gt}
        with open(f"{args.out_dir}/{filename}.pkl", "wb") as file:
            pickle.dump(data, file)
    logger.info("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/prnet_config.yaml", help="Config path")
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()
    generate_offline_gt(args)

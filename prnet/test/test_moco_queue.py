# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from moco.utils.misc import load_config
from moco.utils.trainer import MoCoKeyQueue


def test_k_queue_single_batch():
    config = load_config("moco/config/moco_config.yaml")
    max_batches = config["MODEL"]["moco_queue"]["max_batches"]
    batch_size = config["DATA"]["batch_size"]
    k_queue = MoCoKeyQueue(max_batches=max_batches, batch_size=batch_size)

    B, C, H, W = batch_size, 3, 256, 256
    x = torch.randn(B, C, H, W)
    k_queue.insert_batch(x)
    assert k_queue.queue_size == 1


def test_k_queue_max_batches():
    config = load_config("moco/config/moco_config.yaml")
    max_batches = config["MODEL"]["moco_queue"]["max_batches"]
    batch_size = config["DATA"]["batch_size"]
    k_queue = MoCoKeyQueue(max_batches=max_batches, batch_size=batch_size)

    B, C, H, W = batch_size, 3, 256, 256
    x = torch.randn(B, C, H, W)
    for i in range(max_batches * 2):
        k_queue.insert_batch(x)
    assert k_queue.queue_size == max_batches


def test_k_queue_tensor_shape():
    config = load_config("moco/config/moco_config.yaml")
    max_batches = config["MODEL"]["moco_queue"]["max_batches"]
    batch_size = config["DATA"]["batch_size"]
    k_queue = MoCoKeyQueue(max_batches=max_batches, batch_size=batch_size)

    B, C, H, W = batch_size, 3, 256, 256
    x = torch.randn(B, C, H, W)
    k_queue.insert_batch(x)
    k_queue_tensor = k_queue.get_tensor()
    assert k_queue_tensor.shape == (batch_size, 3, 256, 256)

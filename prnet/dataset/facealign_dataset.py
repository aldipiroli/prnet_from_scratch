import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
from skimage import io
from torch.utils.data import Dataset

from prnet.dataset import utils
from prnet.dataset.external import face3d
from prnet.dataset.external.face3d.morphable_model import MorphabelModel
import torchvision.transforms as transforms

class FaceAlignDataset(Dataset):
    def __init__(self, cfg, mode, logger):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.data_path = self.root_dir / cfg["DATA"]["sub_dataset"]
        self.bfm_path = self.root_dir / cfg["DATA"]["BFM"]
        self.files = os.listdir(self.data_path)
        self.files = list(set([os.path.splitext(file)[0] for file in self.files]))
        self.img_size = cfg["DATA"]["img_size"]
        self.load_uv_coords()
        self.load_morphable_model()

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def load_uv_coords(self):
        # .mat source: https://github.com/yfeng95/face3d/issues/95
        uv_coords_path = self.bfm_path / "BFM_UV.mat"
        self.uv_coords = face3d.morphable_model.load.load_uv_coords(uv_coords_path)
        self.uv_coords = utils.process_uv(self.uv_coords, self.img_size[0], self.img_size[1])
        self.logger.info(f"Loaded uv_coords from {uv_coords_path}")

    def load_morphable_model(self):
        # .mat source: https://github.com/yfeng95/face3d/issues/95
        bmf_path = self.bfm_path / "BFM.mat"
        self.bfm = MorphabelModel(bmf_path)
        self.logger.info(f"Loaded bmf from {bmf_path}")

    def get_uv_map_texture(self, img, img_position, triangles):
        mask_texture = np.zeros_like(img, dtype=np.uint8)
        position_coords = img_position[:, :2].astype(np.int32)
        uv_coords = self.uv_coords[:, :2].astype(np.int32)

        for tri in triangles:
            pos_img = np.clip(position_coords[tri], 0, self.img_size[0] - 1)
            pos_uv = uv_coords[tri]
            colors = np.mean(img[pos_img[:, 1], pos_img[:, 0]], 0) * 255
            cv2.fillConvexPoly(mask_texture, pos_uv, (colors))
        return mask_texture

    def get_uv_coord_3d_map(self, img, img_position, triangles):
        h, w = img.shape[:2]
        uv_coord_3d_map = np.zeros((h, w, 3), dtype=np.float32)
        uv_coords = self.uv_coords[:, :2].astype(np.int32)

        for tri in triangles:
            pos_uv = uv_coords[tri]
            coord_3d = img_position[tri]
            mean_3d = np.mean(coord_3d, 0)  # TODO: barycentric weighting
            color = tuple(map(float, mean_3d))
            cv2.fillConvexPoly(uv_coord_3d_map, pos_uv, color)
        return uv_coord_3d_map

    def overlay_image_to_face_mask(self, img, img_position, triangles):
        mask = np.zeros_like(img, dtype=np.uint8)
        position_coords = img_position[:, :2].astype(np.int32)

        for tri in triangles:
            pos_img = np.clip(position_coords[tri], [0, 0], [self.img_size[0] - 1, self.img_size[1] - 1])
            depth = np.mean(img_position[tri][:, -1], 0)
            cv2.fillConvexPoly(mask, pos_img, (1, 0, 0))
        img_overlay = img + 0.2 * mask
        return img_overlay

    def get_face_depth_mask(self, img, img_position, triangles):
        face_depth_mask = np.zeros_like(img, dtype=np.float32)
        position_coords = img_position[:, :2].astype(np.int32)

        for tri in triangles:
            pos_img = np.clip(position_coords[tri], [0, 0], [self.img_size[0] - 1, self.img_size[1] - 1])
            depth = np.mean(img_position[tri][:, -1], 0)
            curr_depth_mask = np.zeros_like(img, dtype=np.uint8)
            cv2.fillConvexPoly(curr_depth_mask, pos_img, (depth, 0, 0))
            face_depth_mask = np.maximum(face_depth_mask, curr_depth_mask)
        return face_depth_mask

    def get_colored_pcl(self, img, img_position):
        H, W = img.shape[0], img.shape[1]
        img_pos_coords = img_position[:, :2].astype(np.int32)
        pcl_colors = []
        for coords in img_pos_coords:
            valid = utils.check_if_inside_img(coords, self.img_size)
            if valid:
                coords = np.clip(coords, [0, 0], [H - 1, W - 1])
                colors = img[coords[1], coords[0]]
            else:
                colors = [0, 0, 0]
            pcl_colors.append(colors)
        pcl_colors = np.stack(pcl_colors, 0)
        return pcl_colors

    def overlay_3d_map_on_image_as_points(self, img, uv_coord_3d_map):
        H, W = img.shape[0], img.shape[1]
        overlay = img.copy()
        for i in range(0, self.img_size[0]):
            for j in range(0, self.img_size[1]):
                coords_2d = uv_coord_3d_map[i, j][:2]
                coords_2d = np.clip(coords_2d, [0, 0], [H - 1, W - 1]).astype(np.int32)
                overlay[coords_2d[1], coords_2d[0]] = (1, 0, 0)
        return overlay

    def overlay_3d_map_on_image_as_triangles(self, img, uv_coord_3d_map):
        uv_coords = self.uv_coords[..., :2]
        vertex_pos = []
        for uv in uv_coords:
            uv = np.clip(uv, [0, 0], [self.img_size[0] - 1, self.img_size[1] - 1]).astype(np.int32)
            u, v = uv
            vertex_pos.append(uv_coord_3d_map[v, u])
        vertex_pos = np.array(vertex_pos)

        # normalize depth
        depth = vertex_pos[..., -1]
        depth -= depth.min()
        depth /= depth.max()
        vertex_pos[..., -1] = depth

        face_depth_mask = np.zeros_like(img)
        for tri in self.bfm.full_triangles:
            vertex = vertex_pos[tri]
            pos = vertex[..., :2].astype(np.int32)
            depth = np.mean(vertex[..., -1])
            if depth != depth:
                depth = 0
            curr_depth_mask = np.zeros_like(img)
            cv2.fillConvexPoly(curr_depth_mask, pos, color=(depth * 255, 0, 0))
            face_depth_mask = np.maximum(face_depth_mask, curr_depth_mask)

        alpha = 0.7
        overlay = img + alpha * face_depth_mask
        return overlay

    def get_ground_truth(self, full_img, info):
        img, img_position = utils.get_point_aligned_with_image(full_img, info, self.bfm, self.img_size)
        uv_coord_3d_map = self.get_uv_coord_3d_map(img, img_position, self.bfm.full_triangles)
        return img, uv_coord_3d_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.data_path / f"{self.files[idx]}.jpg"
        img_input = io.imread(img_path) / 255.0

        info_path = self.data_path / f"{self.files[idx]}.mat"
        info = sio.loadmat(info_path)

        img, uv_coord_3d_map = self.get_ground_truth(img_input, info)

        img = self.transforms(img).float()
        uv_coord_3d_map = self.transforms(uv_coord_3d_map).float()
        return img, uv_coord_3d_map, self.files[idx]

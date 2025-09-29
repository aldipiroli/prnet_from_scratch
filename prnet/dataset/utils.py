# Most of the code is borrowed from: https://github.com/yfeng95/face3d
import numpy as np
from skimage import transform


def process_uv(uv_coords, uv_h=256, uv_w=256):
    """
    Convert UV coordinates from [0,1] normalized to pixel coordinates.
    Flips vertical axis to match image convention.
    """
    uv_coords[:, 0] *= uv_w - 1
    uv_coords[:, 1] *= uv_h - 1
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1  # flip vertically
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))
    return uv_coords


def get_point_aligned_with_image(image, info, bfm, img_size):
    h, w, _ = image.shape
    pose = info["Pose_Para"].T.astype(np.float32)
    shape_para = info["Shape_Para"].astype(np.float32)
    exp_para = info["Exp_Para"].astype(np.float32)

    # 2. Generate mesh vertices from shape and expression params
    vertices = bfm.generate_vertices(shape_para, exp_para)

    # Apply pose transform (3DDFA style)
    s = pose[-1, 0]
    angles = pose[:3, 0]
    t = pose[3:6, 0]
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)

    # Convert to image coordinates
    image_vertices = transformed_vertices.copy()
    image_vertices[:, 1] = h - image_vertices[:, 1] - 1  # flip y

    # 3. Compute crop transform
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left, right = np.min(kpt[:, 0]), np.max(kpt[:, 0])
    top, bottom = np.min(kpt[:, 1]), np.max(kpt[:, 1])
    center = np.array([0.5 * (left + right), 0.5 * (top + bottom)])
    old_size = 0.5 * ((right - left) + (bottom - top))
    size = int(old_size * 1.5)

    # Random perturbation
    marg = old_size * 0.1
    center[0] += np.random.uniform(-marg, marg)
    center[1] += np.random.uniform(-marg, marg)
    size *= np.random.rand() * 0.2 + 0.9

    # Similarity transform to target crop size
    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    dst_pts = np.array([[0, 0], [0, img_size[0] - 1], [img_size[1] - 1, 0]])
    tform = transform.estimate_transform("similarity", src_pts, dst_pts)

    cropped_image = transform.warp(image, tform.inverse, output_shape=(img_size[0], img_size[1]))

    # 4. Transform vertices consistently with image crop
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)

    # Scale z properly with similarity scale (not just params[0,0])
    a, b = tform.params[0, 0], tform.params[0, 1]
    scale = np.sqrt(a * a + b * b)
    position[:, 2] = image_vertices[:, 2] * scale
    position[:, 2] -= np.min(position[:, 2])
    return cropped_image, position


def get_point_aligned_with_full_image(image, info, bfm):
    h, w, _ = image.shape

    # Parameters from .mat
    pose = info["Pose_Para"].T.astype(np.float32)
    shape_para = info["Shape_Para"].astype(np.float32)
    exp_para = info["Exp_Para"].astype(np.float32)

    # 1. Generate mesh vertices from shape and expression params
    vertices = bfm.generate_vertices(shape_para, exp_para)

    # 2. Apply pose transform (3DDFA style)
    s = pose[-1, 0]
    angles = pose[:3, 0]
    t = pose[3:6, 0]
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)

    # 3. Convert to image coordinates (flip y)
    image_vertices = transformed_vertices.copy()
    image_vertices[:, 1] = h - image_vertices[:, 1] - 1

    # Normalize depth
    position = image_vertices.copy()
    position[:, 2] -= np.min(position[:, 2])

    return image, position


def check_if_inside_img(coord, img_shape):
    return coord[0] > 0 and coord[0] < img_shape[0] and coord[1] > 0 and coord[1] < img_shape[1]


def overlay_mask(img, face_depth_mask):
    img_float = img.astype(np.float32)
    face_depth_mask_float = face_depth_mask.astype(np.float32)

    non_zero_mask = face_depth_mask > 0

    overlay = img_float.copy()
    overlay[non_zero_mask] = face_depth_mask_float[non_zero_mask]
    return overlay

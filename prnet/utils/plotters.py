import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import open3d as o3d

def plot_3d(points, colors=None, point_size=5):
    cloud = pv.PolyData(points)
    if colors is None:
        colors = np.array([[255, 0, 0]] * len(points), dtype=np.uint8)
    else:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    cloud["colors"] = colors
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=point_size, scalars="colors", rgb=True)
    plotter.set_background("black")
    plotter.show(title="3D Point Cloud")


def plot_multiple_3d(cloud_list, colours=[[255, 0, 0], [0, 255, 0]], point_size=5,start_camera_position=None):
    cmap = plt.get_cmap("tab10")
    n = len(cloud_list)

    plotter = pv.Plotter()
    for pts, col in zip(cloud_list, colours):
        pts = np.asarray(pts, dtype=float)
        cloud = pv.PolyData(pts)
        rgb_array = np.tile(col, (pts.shape[0], 1))
        cloud["rgb"] = rgb_array
        plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=point_size,
            scalars="rgb",
            rgb=True,
        )
    if start_camera_position is not None:
        plotter.camera_position = start_camera_position
    plotter.set_background("black")
    # plotter.show()
    return plotter


def plot_3d_error_color(pred, gt, point_size=5, cmap_name="coolwarm", start_camera_position=None):
    errors = np.linalg.norm(pred - gt, axis=1)
    cloud = pv.PolyData(pred)
    cloud["error"] = errors
    plotter = pv.Plotter()
    plotter.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=point_size,
        cmap=cmap_name,
        scalar_bar_args={"color": "k","n_labels": 0,"width":0.3,"title_font_size":1},
    )
    if start_camera_position is not None:
        plotter.camera_position = start_camera_position

    plotter.set_background("black")
    # plotter.show()
    # print("camera_position", plotter.camera_position)
    return plotter


def create_rotation_video(plot_func, output_path="rotation.mp4", n_frames=180, rotation_axis='z', *args, **kwargs):
    plotter = plot_func(*args, **kwargs)
    plotter.off_screen = True
    plotter.open_movie(output_path, framerate=30)

    angle_per_frame = 180 / n_frames
    for i in range(n_frames):
        if rotation_axis.lower() == 'z':
            plotter.camera.Azimuth(angle_per_frame)
        elif rotation_axis.lower() == 'y':
            plotter.camera.Elevation(angle_per_frame)
        elif rotation_axis.lower() == 'x':
            plotter.camera.Roll(angle_per_frame)
        else:
            raise ValueError("rotation_axis must be 'x', 'y', or 'z'")
        plotter.render()
        plotter.write_frame()
    plotter.close()


def plt_preds(img, preds, face_align, batch_id=0, save_path="tmp.png"):
    img = img[batch_id].permute(1, 2, 0).cpu().numpy()
    preds = preds[batch_id].permute(1, 2, 0).detach().cpu().numpy()
    overlay = face_align.overlay_3d_map_on_image_as_points(img, preds)
    plt.imshow(overlay)
    plt.savefig(save_path)
    plt.close("all")

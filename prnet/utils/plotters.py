import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


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

def plt_preds(self, img, preds, batch_id=0, save_path="tmp.png"):
    img = img[batch_id].permute(1, 2, 0).cpu().numpy()
    preds = preds[batch_id].permute(1, 2, 0).detach().cpu().numpy()
    overlay = self.face_align.overlay_3d_map_on_image_as_points(img, preds)
    plt.imshow(overlay)
    plt.savefig(save_path)
    plt.close("all")
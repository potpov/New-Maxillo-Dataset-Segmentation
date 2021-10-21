import json
from hull.voxelize.voxelize import voxelize
from scipy.spatial import Delaunay
import numpy as np
from collections import defaultdict
from scipy.ndimage import binary_fill_holes
import os
import pathlib
from glob import glob


def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos, tetra.vertices, axis=0)
    normsq = np.sum(tetrapos ** 2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
    r = np.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c) / (2 * np.abs(a))
    # Find tetrahedrals
    tetras = tetra.vertices[r < alpha, :]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:, TriComb].reshape(-1, 3)
    Triangles = np.sort(Triangles, axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles: TrianglesDict[tuple(tri)] += 1
    Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] == 1])
    # edges
    EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])
    Edges = Triangles[:, EdgeComb].reshape(-1, 2)
    Edges = np.sort(Edges, axis=1)
    Edges = np.unique(Edges, axis=0)

    Vertices = np.unique(Edges)
    return Vertices, Edges, Triangles


def bilinear_interpolation(plane, x_func, y_func):
    """
    bilinear interpolation between four pixels of the image given a float set of coords
    Args:
        x_func (float): x coordinate
        y_func (float): y coordinate

    Returns:
        (float) interpolated value according to https://en.wikipedia.org/wiki/Bilinear_interpolation
    """

    x1, x2 = int(np.floor(x_func)), int(np.floor(x_func) + 1)
    y1, y2 = int(np.floor(y_func)), int(np.floor(y_func) + 1)
    dx, dy = x_func - x1, y_func - y1
    P1 = plane[:, y1, x1] * (1 - dx) * (1 - dy)
    P2 = plane[:, y2, x1] * (1 - dx) * dy
    P3 = plane[:, y1, x2] * dx * (1 - dy)
    P4 = plane[:, y2, x2] * dx * dy
    return P1 + P2 + P3 + P4


def concave_hull(coords, shape, alpha=5):
    verts, faces, triangles = alpha_shape_3D(coords, alpha=alpha)

    f = []
    for t in triangles:
        f.append(np.stack((coords[t[0]], coords[t[1]], coords[t[2]])))
    f = np.stack(f)

    alpha_vol = np.zeros(shape)
    for z, y, x in voxelize(np.stack(f)):
        alpha_vol[z, y, x] = 1
    return alpha_vol.astype(int), binary_fill_holes(alpha_vol).astype(int)


def convex_hull(gt):
    from hull.smoother import delaunay as mydelaunay
    from scipy.ndimage import binary_erosion
    convex_hull = mydelaunay(gt)
    convex_hull = binary_fill_holes(convex_hull).astype(int)
    reduced = binary_erosion(convex_hull, iterations=2)
    return reduced


def read_from_file(patient):
    try:
        with open(
                os.path.join(patient, 'masks.json')
        ) as f:
            mask_config = json.load(f)

        planes = np.load(os.path.join(patient, 'planes.npy'), allow_pickle=True)
    except Exception as e:
        print(f"WARNING: patient {patient} \nmiss folders. {e}")
        return

    # gt = convert_to_two_labels(gt)
    planes = planes[:, ::-1, ...]  # X Y Z to Z Y X

    ########
    # MOVING THE CONTROL POINTS TO THE VOLUME
    ########
    assert len(mask_config['masks']) == planes.shape[0], f'different number of masks and planes -> unable to compute alpha shape, use exported volume for {patient}'
    voxel_cp = []
    for i, cps in enumerate(mask_config['masks']):
        if cps is None:
            continue
        plane = planes[i]
        for cp in cps['cp']:
            x = cp['x'] / 4
            y = cp['y'] / 4
            xyz = bilinear_interpolation(plane, x, y)
            voxel_cp.append(xyz)

    coords = np.stack(voxel_cp)
    return coords


if __name__ == '__main__':

    TMP_DIR = r'\dense_export_dir'  # where we save results before moving them to our dataset

    patients = ['list_of_patients_dirs_here!']
    for numpatient, patient in enumerate(patients):
        print(f"making patient: {patient}")
        try:
            coords = read_from_file(patient)
            gt = np.load(os.path.join(patient, 'data.npy'))
            _, result_filled = concave_hull(coords, gt.shape)
        except:
            print("for this patient planes and coords dont match. using the tool exported volume.")
            result_filled = gt

        # save result
        log_dir = pathlib.Path(os.path.join(TMP_DIR, patient))
        log_dir.mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(TMP_DIR, patient, 'gt_alpha.npy'), result_filled)

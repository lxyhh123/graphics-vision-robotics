import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def points_array(points, minimum=3):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < minimum or (not np.isfinite(p).all()):
        raise ValueError('需要足够的有限N×3点坐标')
    return p

def height(x, y):
    return 0.22 * np.sin(2.5 * x) * np.cos(2 * y) + 0.09 * x + 0.04 * y * y

def truth_normals(points):
    p = np.asarray(points)
    x, y = (p[:, 0], p[:, 1])
    zx = 0.55 * np.cos(2.5 * x) * np.cos(2 * y) + 0.09
    zy = -0.44 * np.sin(2.5 * x) * np.sin(2 * y) + 0.08 * y
    n = np.c_[-zx, -zy, np.ones(len(p))]
    return n / np.linalg.norm(n, axis=1)[:, None]

def make_data(seed=2026, n=45, noise=0.008, hole_radius=0.23):
    if n < 5 or noise < 0 or (not 0 <= hole_radius < 1):
        raise ValueError('生成参数无效')
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    clean = np.c_[x.ravel(), y.ravel(), height(x, y).ravel()]
    clean = clean[np.linalg.norm(clean[:, :2], axis=1) >= hole_radius]
    noisy = clean.copy()
    noisy[:, 2] += rng.normal(0, noise, len(clean))
    out = np.c_[rng.uniform(-1, 1, (70, 2)), rng.uniform(0.55, 0.9, 70)]
    return (clean, np.vstack([noisy, out]))

def voxel_sample(points, size):
    p = points_array(points, 1)
    if not np.isfinite(size) or size <= 0:
        raise ValueError('体素边长须为正')
    cells = np.floor(p / size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 3))
    np.add.at(sums, inverse, p)
    return sums / np.bincount(inverse)[:, None]

def filter_outliers(points, k=12, std_ratio=1.2):
    p = points_array(points)
    if not 1 <= k < len(p) or std_ratio <= 0:
        raise ValueError('邻域或阈值无效')
    distances, _ = cKDTree(p).query(p, k=k + 1)
    means = distances[:, 1:].mean(axis=1)
    mask = means <= means.mean() + std_ratio * means.std()
    if mask.sum() < 3:
        raise ValueError('过滤后点过少')
    return (p[mask], mask)

def estimate_normals(points, k=16):
    p = points_array(points)
    if not 3 <= k <= len(p):
        raise ValueError('法线邻域至少3点且不超过总点数')
    _, indices = cKDTree(p).query(p, k=k)
    result = []
    for index in indices:
        q = p[index]
        q = q - q.mean(axis=0)
        values, vectors = np.linalg.eigh(q.T @ q / len(q))
        if values[1] <= 1e-12:
            raise ValueError('共线或重合邻域不能确定曲面法线')
        normal = vectors[:, 0]
        if normal[2] < 0:
            normal = -normal
        result.append(normal)
    return np.asarray(result)

def prepare_heightfield(points):
    p = points_array(points)
    xy, inverse = np.unique(np.round(p[:, :2], 12), axis=0, return_inverse=True)
    z = np.array([np.mean(p[inverse == i, 2]) for i in range(len(xy))])
    return np.c_[xy, z]

def triangulate_heightfield(points, max_edge=0.17, hole_radius=0):
    p = points_array(points)
    if max_edge <= 0 or hole_radius < 0:
        raise ValueError('网格约束参数无效')
    if len(np.unique(p[:, :2], axis=0)) != len(p):
        raise ValueError('重复参数坐标破坏单值高度场假设')
    if np.linalg.matrix_rank(p[:, :2] - p[:, :2].mean(0)) < 2:
        raise ValueError('参数域退化')
    faces = []
    for face in Delaunay(p[:, :2]).simplices:
        tri = p[face, :2]
        valid = True
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            d = tri[b] - tri[a]
            if np.linalg.norm(d) > max_edge:
                valid = False
            t = np.clip(-tri[a] @ d / (d @ d), 0, 1)
            if np.linalg.norm(tri[a] + t * d) < hole_radius - 1e-12:
                valid = False
        signs = tri[:, 0] * np.roll(tri[:, 1], -1) - tri[:, 1] * np.roll(tri[:, 0], -1)
        if hole_radius > 0 and (np.all(signs >= 0) or np.all(signs <= 0)):
            valid = False
        if valid:
            faces.append(face)
    return np.asarray(faces, dtype=int).reshape(-1, 3)

def save_ply(path, points, faces=None, normals=None):
    p = np.asarray(points)
    faces = np.empty((0, 3), int) if faces is None else faces
    with open(path, 'w', encoding='ascii') as f:
        header = ('ply\nformat ascii 1.0\nelement vertex %d\n'
                  'property float x\nproperty float y\nproperty float z\n')
        f.write(header % len(p))
        if normals is not None:
            f.write('property float nx\nproperty float ny\nproperty float nz\n')
        face_header = ('element face %d\n'
                       'property list uchar int vertex_indices\nend_header\n')
        f.write(face_header % len(faces))
        values = p if normals is None else np.c_[p, normals]
        for row in values:
            f.write(' '.join(('%.9g' % v for v in row)) + '\n')
        for row in faces:
            f.write('3 ' + ' '.join(map(str, row)) + '\n')

def main():
    ap = argparse.ArgumentParser(description='E3高度场处理；all输出完整对照，pipeline输出指定参数')
    ap.add_argument('--mode', choices=['all', 'pipeline'], default='all')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--voxel', type=float, default=0.065)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--std-ratio', type=float, default=1.2)
    ap.add_argument('--max-edge', type=float, default=0.17)
    ap.add_argument('--noise', type=float, default=0.008)
    ap.add_argument('--output', type=Path, default=Path('e3_output'))
    a = ap.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    clean, raw = make_data(a.seed, noise=a.noise)
    try:
        filtered, mask = filter_outliers(raw, 12, a.std_ratio)
        sample = prepare_heightfield(voxel_sample(filtered, a.voxel))
        normals = estimate_normals(sample, a.k)
        faces = triangulate_heightfield(sample, a.max_edge, 0.23)
    except (NotImplementedError, ValueError) as e:
        ap.exit(2, str(e) + '\n')
    for name, p in [('clean', clean), ('raw', raw), ('filtered', filtered), ('sample', sample)]:
        np.savetxt(out / (name + '.csv'), p, delimiter=',', header='x,y,z', comments='')
        save_ply(out / (name + '.ply'), p)
    save_ply(out / 'mesh.ply', sample, faces, normals)
    np.savetxt(out / 'normal_truth.csv', truth_normals(sample), delimiter=',',
        header='nx,ny,nz', comments='')
    np.savez(out / 'input_truth.npz', clean=clean, raw=raw, true_normals=truth_normals(clean),
        seed=a.seed)
    truth = truth_normals(sample)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(normals * truth, axis=1), -1, 1)))
    errors = sample[:, 2] - height(sample[:, 0], sample[:, 1])
    rows = []
    sizes = [0.04, 0.065, 0.1] if a.mode == 'all' else [a.voxel]
    ks = [8, 16, 32] if a.mode == 'all' else [a.k]
    for size in sizes:
        q = prepare_heightfield(voxel_sample(filtered, size))
        for k in ks:
            n = estimate_normals(q, k)
            an = np.rad2deg(np.arccos(np.clip(np.sum(n * truth_normals(q), axis=1), -1, 1)))
            rows.append(dict(voxel=size, k=k, points=len(q),
                height_rmse=float(np.sqrt(np.mean((q[:, 2] - height(q[:, 0], q[:,
                1])) ** 2))), mean_normal_deg=float(an.mean()),
                p95_normal_deg=float(np.percentile(an, 95))))
    with open(out / 'parameter_scan.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig = plt.figure(figsize=(12, 4))
    for i, (title, p) in enumerate([('Raw: noise and 70 outliers', raw), ('Filtered',
        filtered), ('Voxel centroids', sample)]):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.scatter(*p.T, s=2, c=p[:, 2], cmap='viridis')
        ax.set_title(title)
        ax.set_zlim(-0.4, 1)
    fig.tight_layout()
    fig.savefig(out / 'sampling.png', dpi=160)
    plt.close(fig)
    fig = plt.figure(figsize=(11, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*sample.T, s=2)
    step = max(1, len(sample) // 100)
    ax.quiver(*sample[::step].T, *normals[::step].T, length=0.1, color='red')
    ax.set_title('Oriented PCA normals')
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_trisurf(sample[:, 0], sample[:, 1], sample[:, 2], triangles=faces, cmap='viridis',
        linewidth=0.1)
    ax.set_title('Heightfield mesh: guarded hole')
    fig.tight_layout()
    fig.savefig(out / 'normals_mesh.png', dpi=160)
    plt.close(fig)
    unsafe = triangulate_heightfield(sample, 10, 0)

    # 仅使用最大边长约束，不进行圆孔保护
    edge_only = triangulate_heightfield(
        sample, a.max_edge, 0
    )

    # 比较三种三角网格重建方式
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    results = [
        (unsafe, "No constraints"),
        (edge_only, "Edge-length only"),
        (faces, "Edge + hole constraints"),
    ]

    for ax, (mesh_faces, title) in zip(axs, results):
        ax.triplot(
            sample[:, 0],
            sample[:, 1],
            mesh_faces,
            lw=0.3
        )
        ax.set_title(f"{title}\nFaces: {len(mesh_faces)}")
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out / "hole_compare.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].tripcolor(sample[:, 0], sample[:, 1], faces, errors, cmap='coolwarm')
    axs[0].set_title('Vertex height error')
    axs[1].triplot(sample[:, 0], sample[:, 1], unsafe, lw=0.3)
    axs[1].set_title('No hole guards: false fill')
    axs[2].triplot(sample[:, 0], sample[:, 1], faces, lw=0.3)
    axs[2].set_title('Known-hole guards')
    for ax in axs:
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out / 'errors_hole.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for size in sizes:
        z = [r for r in rows if r['voxel'] == size]
        ax.plot([r['k'] for r in z], [r['mean_normal_deg'] for r in z], '-o',
            label=f'voxel={size}')
    ax.set(xlabel='Neighborhood k', ylabel='Mean normal error (degree)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'parameter_scan.png', dpi=160)
    plt.close(fig)
    metrics = dict(seed=a.seed, noise=a.noise, voxel=a.voxel, k=a.k, std_ratio=a.std_ratio,
        max_edge=a.max_edge, raw_points=len(raw), clean_points=len(clean),
        filtered_points=len(filtered), sample_points=len(sample), faces=len(faces),
        unsafe_faces=len(unsafe), height_rmse=float(np.sqrt(np.mean(errors ** 2))),
        mean_normal_deg=float(angles.mean()), p95_normal_deg=float(np.percentile(angles, 95)),
        parameter_scan=rows)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'parameter_scan'}, indent=2))
if __name__ == '__main__':
    main()

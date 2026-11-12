import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def validate(points):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or (not np.isfinite(p).all()):
        raise ValueError('需要至少三个有限三维点')
    return p

def transform(points, R, t):
    return np.asarray(points) @ np.asarray(R).T + np.asarray(t)

def rigid_svd(source, target):
    p, q = (validate(source), validate(target))
    if p.shape != q.shape:
        raise ValueError('对应点数不同')
    pc, qc = (p.mean(0), q.mean(0))
    a, b = (p - pc, q - qc)
    if np.linalg.matrix_rank(a) < 2 or np.linalg.matrix_rank(b) < 2:
        raise ValueError('共线或重合对应不能唯一约束三维旋转')
    U, _, Vt = np.linalg.svd(a.T @ b)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ D @ U.T
    return (R, qc - R @ pc)

def icp(source, target, max_iter=60, trim=1.0, gate=np.inf, initial=None, tolerance=1e-09):
    """学生任务：根据实验四正文要求完成本函数。"""
    raise NotImplementedError("请完成 icp 函数")

def axis_rotation(angle):
    a = np.array([0.3, 0.5, 1.0])
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def make_case(seed=11, overlap=1.0, outlier_fraction=0.12, noise=0.003):
    if not 0 < overlap <= 1 or not 0 <= outlier_fraction < 1 or noise < 0:
        raise ValueError('数据参数无效')
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, (650, 2))
    x, y = xy.T
    base = np.c_[x, y, 0.24 * np.sin(3 * x) * np.cos(2 * y) + 0.13 * x + 0.08 * y * y]
    truth_R = axis_rotation(np.deg2rad(12))
    truth_t = np.array([0.12, -0.08, 0.055])
    target = base + rng.normal(0, noise, base.shape)
    evaluation = (base - truth_t) @ truth_R
    cut = np.quantile(base[:, 0], overlap)
    subset = evaluation[base[:, 0] <= cut]
    source = subset + rng.normal(0, noise, subset.shape)
    count = int(len(source) * outlier_fraction / (1 - outlier_fraction))
    out = rng.uniform([-1.6, -1.5, 0.6], [1.6, 1.5, 1.3], (count, 3))
    return (np.vstack([source, out]), target, evaluation, base, truth_R, truth_t)

def evaluate(R, t, truth_R, truth_t, evaluation, truth_target, target):
    cosine = np.clip((np.trace(R @ truth_R.T) - 1) / 2, -1, 1)
    moved = transform(evaluation, R, t)
    distances, _ = cKDTree(target).query(moved)
    return dict(rotation_error_deg=float(np.rad2deg(np.arccos(cosine))),
        translation_error=float(np.linalg.norm(t - truth_t)),
        common_rmse=float(np.sqrt(np.mean(np.sum((moved - truth_target) ** 2, axis=1)))),
        common_nn_rmse=float(np.sqrt(np.mean(distances ** 2))),
        common_inlier_fraction=float(np.mean(distances < 0.03)))

def voxel_fuse(points, size=0.035):
    """学生任务：根据实验四正文要求完成本函数。"""
    raise NotImplementedError("请完成 voxel_fuse 函数")

def save_ply(path, points):
    with open(path, 'w', encoding='ascii') as f:
        header = ('ply\nformat ascii 1.0\nelement vertex %d\n'
                  'property float x\nproperty float y\nproperty float z\nend_header\n')
        f.write(header % len(points))
        for row in points:
            f.write(' '.join(('%.9g' % v for v in row)) + '\n')

def main():
    ap = argparse.ArgumentParser(description='E4配准研究；all运行三个种子及失败情形，single运行单个配置')
    ap.add_argument('--mode', choices=['all', 'single'], default='all')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--overlap', type=float, default=1.0)
    ap.add_argument('--outliers', type=float, default=0.12)
    ap.add_argument('--noise', type=float, default=0.003)
    ap.add_argument('--trim', type=float, default=0.8)
    ap.add_argument('--gate', type=float, default=0.25)
    ap.add_argument('--initial-angle', type=float, default=0)
    ap.add_argument('--max-iter', type=int, default=60)
    ap.add_argument('--output', type=Path, default=Path('e4_output'))
    a = ap.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [('normal', a.overlap, a.initial_angle)]
    if a.mode == 'all':
        scenarios += [('low_overlap_bad_init', 0.25, 100)]
    seeds = [11, 23, 37] if a.mode == 'all' else [a.seed]
    rows = []
    histories = {}
    first = None
    for scenario, overlap, initial_angle in scenarios:
        for seed in seeds:
            source, target, ev, truth_target, truth_R, truth_t = make_case(seed, overlap,
                a.outliers, a.noise)
            initial = (axis_rotation(np.deg2rad(initial_angle)), np.zeros(3))
            methods = [('baseline', 1.0, np.inf), ('trimmed_gated', a.trim, a.gate)]
            if a.mode == 'all' and scenario == 'normal':
                methods += [('trim_0.6', 0.6, a.gate), ('gate_0.1', a.trim, 0.1)]
            for method, trim, gate in methods:
                try:
                    R, t, h = icp(source, target, a.max_iter, trim, gate, initial)
                except NotImplementedError as e:
                    ap.exit(2, str(e) + '\n')
                except ValueError as e:
                    rows.append(dict(scenario=scenario, seed=seed, method=method,
                        overlap=overlap, initial_angle=initial_angle, trim=trim,
                        gate=None if np.isinf(gate) else gate, status=str(e), iterations=0,
                        fit_rmse=None, retained=None, rotation_error_deg=None,
                        translation_error=None, common_rmse=None, common_nn_rmse=None,
                        common_inlier_fraction=None))
                    continue
                m = evaluate(R, t, truth_R, truth_t, ev, truth_target, target)
                row = dict(scenario=scenario, seed=seed, method=method, overlap=overlap,
                    initial_angle=initial_angle, trim=trim,
                    gate=None if np.isinf(gate) else gate, status='ok', iterations=len(h),
                    fit_rmse=h[-1]['fit_rmse'], retained=h[-1]['retained'], **m)
                rows.append(row)
                histories[f'{scenario}_{seed}_{method}'] = h
                if scenario == 'normal' and seed == seeds[0] and (method in ['baseline',
                    'trimmed_gated']):
                    save_ply(out / (method + '_aligned.ply'), transform(source, R, t))
                    if method == 'trimmed_gated':
                        from scipy.spatial import cKDTree
                        moved = transform(source, R, t)
                        d, _ = cKDTree(target).query(moved)
                        fused = voxel_fuse(np.vstack([moved[d < 0.03], target]))
                        save_ply(out / 'fusion.ply', fused)
                        first = (source, target, ev, truth_target, truth_R, truth_t, R, t,
                            fused)
            np.savez(out / f'input_{scenario}_{seed}.npz', source=source, target=target,
                evaluation_source=ev, evaluation_target=truth_target, truth_R=truth_R,
                truth_t=truth_t, seed=seed, overlap=overlap)
    with open(out / 'comparison.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / 'convergence.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'iteration', 'fit_rmse', 'retained'])
        w.writeheader()
        for key, h in histories.items():
            for v in h:
                w.writerow(dict(case=key, **v))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for method in ['baseline', 'trimmed_gated']:
        key = f'normal_{seeds[0]}_{method}'
        if key in histories:
            h = histories[key]
            axs[0].plot([v['iteration'] for v in h], [v['fit_rmse'] for v in h], label=method)
    axs[0].set(xlabel='Iteration', ylabel='Selected correspondence fit RMSE',
        title='Training residual: different subsets')
    axs[0].legend()
    for j, method in enumerate(['baseline', 'trimmed_gated', 'trim_0.6', 'gate_0.1']):
        z = [r for r in rows if r['scenario'] == 'normal'
             and r['method'] == method and r['status'] == 'ok']
        if z:
            axs[1].scatter([j] * len(z), [r['common_rmse'] for r in z], label=method)
    axs[1].set(xticks=[], ylabel='Common truth evaluation RMSE',
        title='Identical fixed evaluation for each seed')
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'convergence_comparison.png', dpi=160)
    plt.close(fig)
    if first:
        source, target, ev, truth_target, truth_R, truth_t, R, t, fused = first
        for name, p in [('source', source), ('target', target)]:
            save_ply(out / (name + '.ply'), p)
            np.savetxt(out / (name + '.csv'), p, delimiter=',', header='x,y,z', comments='')
        transform_info = dict(direction='source_to_target', R=truth_R.tolist(),
                              t=truth_t.tolist(), estimated_R=R.tolist(),
                              estimated_t=t.tolist())
        (out / 'truth_transform.json').write_text(
            json.dumps(transform_info, indent=2), encoding='utf-8')
        fig = plt.figure(figsize=(12, 4))
        for i, (title, p) in enumerate([('Before', source), ('Trimmed + gated',
            transform(source, R, t)), ('Fused: inlier gate + voxel', fused)]):
            ax = fig.add_subplot(1, 3, i + 1, projection='3d')
            ax.scatter(*target.T, s=2, color='blue', alpha=0.4)
            ax.scatter(*p.T, s=2, color='orange', alpha=0.5)
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out / 'registration_fusion.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for method in ['baseline', 'trimmed_gated']:
        z = [r for r in rows if r['method'] == method and r['status'] == 'ok']
        ax.scatter([r['rotation_error_deg'] for r in z], [r['common_rmse'] for r in z],
            label=method)
    ax.set(xlabel='Rotation error (degree)', ylabel='Common truth RMSE',
        title='Normal and low-overlap/bad-initialization cases')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'failure_cases.png', dpi=160)
    plt.close(fig)
    metrics = dict(configuration=dict(seed=a.seed, seeds=seeds, outlier_fraction=a.outliers,
        noise=a.noise, trim=a.trim, gate=a.gate, max_iter=a.max_iter), rows=rows)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
if __name__ == '__main__':
    main()

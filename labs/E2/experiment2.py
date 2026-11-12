"""实验二完整程序。运行方式见同目录的《实验二指导书》。"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

# 材质实验只需修改这里，或使用 --materials 指定自己的 JSON 参数文件。
MATERIALS = {
    "matte": {"base": [0.27, 0.55, 0.78], "ka": 0.14,
              "kd": 0.65, "ks": 0.05, "shininess": 8},
    "glossy": {"base": [0.27, 0.55, 0.78], "ka": 0.14,
               "kd": 0.65, "ks": 0.60, "shininess": 64},
}
BASE_MATERIAL = {"base": [0.27, 0.55, 0.78], "ka": 0.14,
                 "kd": 0.65, "ks": 0.38, "shininess": 32}
BACKGROUND = (232, 236, 242)

def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def as_uint8(image):
    if image.dtype == np.uint8:
        return image
    return np.uint8(np.rint(np.clip(image, 0, 1) * 255))

def save_panel(path, items, columns=2):
    """将输出图按顺序排列；标题使用文件名，便于对照。"""
    size = items[0][1].shape[0]
    rows = (len(items) + columns - 1) // columns
    panel = Image.new("RGB", (columns * size, rows * (size + 28)), "white")
    draw = ImageDraw.Draw(panel)
    for index, (name, array) in enumerate(items):
        x = (index % columns) * size
        y = (index // columns) * (size + 28)
        draw.text((x + 8, y + 7), name, fill="black")
        panel.paste(Image.fromarray(as_uint8(array)), (x, y + 28))
    panel.save(path)

# 1. 重心坐标和三角形光栅化
def barycentric(triangle, points):
    a, b, c = np.asarray(triangle, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    denominator = ((b[1] - c[1]) * (a[0] - c[0])
                   + (c[0] - b[0]) * (a[1] - c[1]))
    if abs(denominator) < 1e-12:
        raise ValueError("三角形投影面积为零")
    x, y = points[:, 0], points[:, 1]
    w0 = ((b[1] - c[1]) * (x - c[0])
          + (c[0] - b[0]) * (y - c[1])) / denominator
    w1 = ((c[1] - a[1]) * (x - c[0])
          + (a[0] - c[0]) * (y - c[1])) / denominator
    return np.column_stack((w0, w1, 1 - w0 - w1))

def rasterize_triangle(vertices, size):
    """输入三个 (x,y,depth) 顶点；返回像素列、行、权重和深度。"""
    vertices = np.asarray(vertices, dtype=float)
    screen = np.column_stack(((vertices[:, 0] + 1) * size / 2,
                              (1 - vertices[:, 1]) * size / 2))
    left = max(0, int(np.floor(screen[:, 0].min())))
    right = min(size - 1, int(np.ceil(screen[:, 0].max())))
    top = max(0, int(np.floor(screen[:, 1].min())))
    bottom = min(size - 1, int(np.ceil(screen[:, 1].max())))
    columns, rows = np.meshgrid(np.arange(left, right + 1),
                               np.arange(top, bottom + 1))
    samples = np.column_stack((columns.ravel() + 0.5, rows.ravel() + 0.5))
    weights = barycentric(screen, samples)
    inside = np.all(weights >= -1e-10, axis=1)
    weights = weights[inside]
    depth = weights @ vertices[:, 2]
    return columns.ravel()[inside], rows.ravel()[inside], weights, depth

def run_coverage(output, size):
    vertices = [[-1., -1., 0.], [0.5, -1., 0.], [-1., 0.5, 0.]]
    image = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    x, y, _, _ = rasterize_triangle(vertices, size)
    image[y, x] = (66, 145, 210)
    Image.fromarray(image).save(output / "coverage.png")
    weights = barycentric([[0, 0], [4, 0], [0, 4]], [[1, 1], [4, 4]])
    save_json(output / "metrics.json", {
        "inside_weights": weights[0].tolist(),
        "outside_weights": weights[1].tolist(),
        "covered_pixels": len(x), "size": size,
    })

# 2. 深度缓冲：深度越小，越靠近观察者
def render_depth_scene(size, use_depth, draw_order):
    triangles = {
        "far": ([[-0.75, -0.65, 0.6], [0.55, -0.65, 0.6],
                 [-0.10, 0.65, 0.6]], (220, 90, 80)),
        "near": ([[-0.50, -0.40, -0.4], [0.80, -0.40, -0.4],
                  [0.15, 0.85, -0.4]], (66, 145, 210)),
    }
    image = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    depth_buffer = np.full((size, size), np.inf)
    for name in draw_order:
        vertices, color = triangles[name]
        x, y, _, depth = rasterize_triangle(vertices, size)
        keep = depth < depth_buffer[y, x] if use_depth else np.ones(len(x), dtype=bool)
        x, y, depth = x[keep], y[keep], depth[keep]
        image[y, x] = color
        depth_buffer[y, x] = depth
    return image

def run_depth(output, size):
    results = {}
    for use_depth, prefix in ((True, "depth"), (False, "no_depth")):
        for order, names in (("far_first", ["far", "near"]),
                             ("near_first", ["near", "far"])):
            filename = f"{prefix}_{order}.png"
            results[filename] = render_depth_scene(size, use_depth, names)
            Image.fromarray(results[filename]).save(output / filename)
    metrics = {"size": size}
    for prefix in ("depth", "no_depth"):
        difference = results[f"{prefix}_far_first.png"] != results[f"{prefix}_near_first.png"]
        metrics[f"{prefix}_order_difference_pixels"] = int(np.count_nonzero(np.any(difference, axis=2)))
    save_json(output / "depth_metrics.json", metrics)
    save_panel(output / "depth_comparison.png", list(results.items()))

# 3. 球体网格：两极各保留一个顶点，避免生成重复顶点和退化三角形
def sphere_mesh(rings=16, sectors=32):
    vertices = [[0., 1., 0.]]
    for i in range(1, rings):
        theta = np.pi * i / rings
        for j in range(sectors):
            phi = 2 * np.pi * j / sectors
            vertices.append([np.sin(theta) * np.cos(phi), np.cos(theta),
                             np.sin(theta) * np.sin(phi)])
    south = len(vertices)
    vertices.append([0., -1., 0.])
    faces = []
    for j in range(sectors):
        next_j = (j + 1) % sectors
        faces.append([0, 1 + next_j, 1 + j])
        for i in range(rings - 2):
            a, b = 1 + i * sectors + j, 1 + i * sectors + next_j
            c, d = a + sectors, b + sectors
            faces.extend([[a, b, c], [b, d, c]])
        last = 1 + (rings - 2) * sectors
        faces.append([last + j, last + next_j, south])
    return np.asarray(vertices), np.asarray(faces, dtype=int)

def normalize(vectors):
    vectors = np.asarray(vectors, dtype=float)
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(length, 1e-12)

def transform_normals(model_matrix, normals):
    normal_matrix = np.linalg.inv(model_matrix).T
    # normals 按行存储，因此右乘 normal_matrix.T。
    return normalize(normals @ normal_matrix.T)

def shade(points, normals, light, eye, material):
    normals = normalize(normals)
    light_dir = normalize(light - points)
    view_dir = normalize(eye - points)
    half_dir = normalize(light_dir + view_dir)
    ndotl = np.maximum(np.sum(normals * light_dir, axis=1), 0)
    ndoth = np.maximum(np.sum(normals * half_dir, axis=1), 0)
    base = np.asarray(material["base"])
    ambient = np.broadcast_to(material["ka"] * base, points.shape).copy()
    diffuse = material["kd"] * ndotl[:, None] * base
    strength = material["ks"] * ndoth ** material["shininess"]
    strength[ndotl <= 0] = 0  # 背光表面不产生该光源的镜面高光。
    specular = np.repeat(strength[:, None], 3, axis=1)
    return {"ambient": ambient, "diffuse": diffuse, "specular": specular,
            "combined": ambient + diffuse + specular}

def render_sphere(size, material, rings=16, sectors=32):
    started = perf_counter()
    vertices, faces = sphere_mesh(rings, sectors)
    model_matrix = np.eye(3) * 0.78
    world_vertices = vertices @ model_matrix.T
    world_normals = transform_normals(model_matrix, vertices)
    projected = world_vertices.copy()
    projected[:, 2] *= -1  # 从 +z 观察时，世界 z 越大，深度越小。
    images = {name: np.zeros((size, size, 3), dtype=float)
              for name in ("ambient", "diffuse", "specular", "combined")}
    depth_buffer = np.full((size, size), np.inf)
    light, eye = np.array([2., 2., 3.]), np.array([0., 0., 3.])
    for face in faces:
        try:
            x, y, weights, depth = rasterize_triangle(projected[face], size)
        except ValueError:
            continue  # 有些非退化空间三角形从当前视角看投影面积为零。
        keep = depth < depth_buffer[y, x]
        if not np.any(keep):
            continue
        x, y, depth, weights = x[keep], y[keep], depth[keep], weights[keep]
        points = weights @ world_vertices[face]
        normals = weights @ world_normals[face]
        colors = shade(points, normals, light, eye, material)
        for name in images:
            images[name][y, x] = colors[name]
        depth_buffer[y, x] = depth
    visible = np.isfinite(depth_buffer)
    specular = images["specular"][visible, 0]
    metrics = {
        "size": size, "rings": rings, "sectors": sectors,
        "vertices": len(vertices), "triangles": len(faces),
        "visible_pixels": int(visible.sum()),
        "specular_peak": float(specular.max()),
        "specular_pixels_above_0_1": int(np.count_nonzero(specular > 0.1)),
        "render_seconds": round(perf_counter() - started, 4),
        "material": material,
    }
    return images, metrics

def run_lighting(output, size, rings, sectors):
    images, metrics = render_sphere(size, BASE_MATERIAL, rings, sectors)
    for name, array in images.items():
        Image.fromarray(as_uint8(array)).save(output / f"{name}.png")
    save_panel(output / "lighting_comparison.png", list(images.items()))
    save_json(output / "metrics.json", metrics)

def validate_materials(materials):
    for name in ("matte", "glossy"):
        material = materials[name]
        base = np.asarray(material["base"], dtype=float)
        coefficients = np.asarray([material[key] for key in ("ka", "kd", "ks")], dtype=float)
        shininess = float(material["shininess"])
        if base.shape != (3,) or not np.all(np.isfinite(base)) or np.any((base < 0) | (base > 1)):
            raise ValueError(f"{name}: base 必须是三个 0 到 1 的有限数值")
        if not np.all(np.isfinite(coefficients)) or np.any((coefficients < 0) | (coefficients > 1)):
            raise ValueError(f"{name}: ka、kd、ks 必须在 0 到 1 之间")
        if not np.isfinite(shininess) or shininess < 1:
            raise ValueError(f"{name}: shininess 必须是大于或等于 1 的有限数值")

def run_materials(output, size, rings, sectors, materials):
    items, metrics = [], {}
    for name in ("matte", "glossy"):
        images, metrics[name] = render_sphere(size, materials[name], rings, sectors)
        array = as_uint8(images["combined"])
        Image.fromarray(array).save(output / f"{name}.png")
        items.append((name, array))
    save_panel(output / "material_comparison.png", items)
    save_json(output / "materials_used.json", materials)
    save_json(output / "metrics.json", metrics)

# 4. 统一入口：每一步只改变运行参数，不需要拼接程序
def main():
    parser = argparse.ArgumentParser(description="实验二：软件光栅化与光照材质设计")
    parser.add_argument("--stage", choices=["coverage", "depth", "lighting", "materials", "all"], default="all")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--rings", type=int, default=16)
    parser.add_argument("--sectors", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--materials", type=Path, help="包含 matte 和 glossy 的 JSON 参数文件")
    args = parser.parse_args()
    if not 64 <= args.size <= 1024:
        parser.error("size 应在 64 到 1024 之间")
    if not 3 <= args.rings <= 128 or not 3 <= args.sectors <= 256:
        parser.error("rings 应在 3 到 128 之间，sectors 应在 3 到 256 之间")
    materials = MATERIALS
    if args.stage in ("materials", "all"):
        try:
            if args.materials:
                materials = json.loads(args.materials.read_text(encoding="utf-8-sig"))
            validate_materials(materials)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"材质参数错误：{error}")
    stages = ("coverage", "depth", "lighting", "materials") if args.stage == "all" else (args.stage,)
    for stage in stages:
        output = args.output / stage
        output.mkdir(parents=True, exist_ok=True)
        if stage == "coverage":
            run_coverage(output, args.size)
        elif stage == "depth":
            run_depth(output, args.size)
        elif stage == "lighting":
            run_lighting(output, args.size, args.rings, args.sectors)
        else:
            run_materials(output, args.size, args.rings, args.sectors, materials)
        print(f"{stage}: {output.resolve()}")

if __name__ == "__main__":
    main()

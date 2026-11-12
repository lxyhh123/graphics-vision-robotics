"""实验一：三维变换、层次场景与投影。将本文件直接运行即可生成图与数值。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 数学约定：右手坐标系；y 向上；点用列向量；角度输入为度。
def translation(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix

def rotation_y(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0],
                     [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)

def scaling(x, y, z):
    return np.diag([x, y, z, 1.0])

def make_scene(angle=35.0, offset=1.45, wrong=False):
    """展台是显示几何叶节点，展品继承旋转框架而不继承展台缩放。"""
    move_a = translation(offset, 0.5, 0)
    turn_a = rotation_y(25)
    local_a = turn_a @ move_a if wrong else move_a @ turn_a
    return [
        dict(name='exhibition', parent=None, matrix=np.eye(4)),
        dict(name='turntable', parent='exhibition', matrix=rotation_y(angle)),
        dict(name='base', parent='turntable',
             matrix=translation(0, -0.12, 0) @ scaling(5.6, 0.24, 3.6)),
        dict(name='display_A', parent='turntable',
             matrix=local_a @ scaling(0.75, 1.0, 0.75)),
        dict(name='display_B', parent='turntable',
             matrix=translation(-offset, 0.4, 0) @ rotation_y(-20)
                    @ scaling(0.8, 0.8, 0.8)),
        dict(name='small_piece', parent='display_A',
             matrix=translation(0, 0.625, 0) @ scaling(0.45, 0.25, 0.45)),
    ]

def world_matrices(nodes):
    table = {node['name']: node for node in nodes}
    if len(table) != len(nodes):
        raise ValueError('节点名称重复')
    result, active = {}, set()

    def visit(name):
        if name in result:
            return result[name]
        if name in active:
            raise ValueError('场景层次形成循环')
        if name not in table:
            raise ValueError('找不到父节点：' + name)
        active.add(name)
        node = table[name]
        parent = node['parent']
        parent_world = np.eye(4) if parent is None else visit(parent)
        result[name] = parent_world @ node['matrix']
        active.remove(name)
        return result[name]

    for name in table:
        visit(name)
    return result

def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError('相机方向或上方向不能构成坐标系')
    return vector / length

def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, dtype=float)
    forward = normalize(np.asarray(target, dtype=float) - eye)
    right = normalize(np.cross(forward, up))
    vertical = np.cross(right, forward)
    view = np.eye(4)
    view[:3, :3] = [right, vertical, -forward]
    view[:3, 3] = -view[:3, :3] @ eye
    return view

def orthographic(left, right, bottom, top, near, far):
    if not (left < right and bottom < top and 0 < near < far):
        raise ValueError('正交投影范围非法')
    return np.array([
        [2 / (right-left), 0, 0, -(right+left) / (right-left)],
        [0, 2 / (top-bottom), 0, -(top+bottom) / (top-bottom)],
        [0, 0, -2 / (far-near), -(far+near) / (far-near)],
        [0, 0, 0, 1],
    ])

def perspective(fov, aspect, near, far):
    if not (0 < fov < 180 and aspect > 0 and 0 < near < far):
        raise ValueError('透视投影参数非法')
    f = 1 / np.tan(np.deg2rad(fov) / 2)
    return np.array([
        [f/aspect, 0, 0, 0], [0, f, 0, 0],
        [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
        [0, 0, -1, 0],
    ])

def point(matrix, xyz=(0, 0, 0)):
    answer = matrix @ np.array([*xyz, 1.0])
    return answer[:3] / answer[3]

# 每个面都引用同一组顶点。所有边端点因此共用完全相同的投影坐标。
CUBE = np.array([
    [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
    [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
])
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3),
         (1, 5, 6, 2), (3, 2, 6, 7), (0, 1, 5, 4)]
COLORS = {
    'base': (130, 148, 165), 'display_A': (229, 104, 77),
    'display_B': (45, 161, 179), 'small_piece': (155, 102, 205),
}

def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()

def draw_scene(nodes, mode, output, angle, offset):
    width, height = 1100, 720
    left, top, plot_width, plot_height = 45, 58, 765, 600
    view = look_at((5.5, 4.2, 6), (0, 0.2, 0))
    projection = (orthographic(-3.8, 3.8, -2.8, 2.8, .5, 30)
                  if mode == 'orthographic'
                  else perspective(40, plot_width / plot_height, .5, 30))
    image = Image.new('RGB', (width, height), (249, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((25, 30, 825, 683), radius=14,
                           fill=(239, 244, 250), outline=(200, 211, 224), width=2)
    draw.text((47, 31), mode.upper().replace('_', ' '),
              fill=(29, 48, 72), font=font(22))

    world = world_matrices(nodes)
    for name, color in COLORS.items():
        faces = []
        local = np.column_stack((CUBE, np.ones(8)))
        camera = (view @ world[name] @ local.T).T
        clip = (projection @ camera.T).T
        if np.any(clip[:, 3] <= 1e-10):
            raise ValueError('顶点位于相机后方或透视除法无效')
        ndc = clip[:, :3] / clip[:, 3, None]
        if np.any(np.abs(ndc) > 1.0 + 1e-9):
            raise ValueError('场景超出视域；请减小偏移')
        screen = np.column_stack((left + (ndc[:, 0]+1)*plot_width/2,
                                  top + (1-ndc[:, 1])*plot_height/2))
        for face in FACES:
            distance = -float(np.mean(camera[list(face), 2]))
            vertices = [tuple(screen[index]) for index in face]
            faces.append((distance, vertices, color, face))
        # 各物体内部由远到近画面。展台先画，彼此不穿插的展品后画。
        for _, vertices, face_color, face in sorted(
                faces, key=lambda item: -item[0]):
            brightness = 0.88 if face in (FACES[0], FACES[2], FACES[5]) else 1.0
            fill = tuple(round(channel * brightness) for channel in face_color)
            draw.polygon(vertices, fill=fill)
            draw.line(vertices + vertices[:1], fill=(48, 64, 82), width=2,
                      joint='curve')

    draw.text((855, 62), 'SCENE OBJECTS', fill=(29, 48, 72), font=font(22))
    for row, (name, color) in enumerate(COLORS.items()):
        y = 118 + row * 80
        draw.rounded_rectangle((855, y, 886, y+31), radius=5, fill=color)
        draw.text((902, y+3), name, fill=(32, 49, 70), font=font(18))
    draw.text((855, 490), f'angle: {angle:g} deg',
              fill=(65, 79, 98), font=font(16))
    draw.text((855, 518), f'offset: {offset:g}',
              fill=(65, 79, 98), font=font(16))
    draw.text((855, 558), 'values saved in metrics.json',
              fill=(65, 79, 98), font=font(12))
    draw.text((48, 685), 'Filled faces hide rear edges; projection uses shared vertices.',
              fill=(65, 79, 98), font=font(14))
    image.save(output)

def draw_hierarchy(nodes, output):
    positions = {
        'exhibition': (550, 72), 'turntable': (550, 170),
        'base': (205, 330), 'display_A': (550, 330),
        'display_B': (895, 330), 'small_piece': (550, 485),
    }
    image = Image.new('RGB', (1100, 570), (249, 250, 252))
    draw = ImageDraw.Draw(image)
    for node in nodes:
        parent = node['parent']
        if parent is not None:
            x1, y1 = positions[parent]
            x2, y2 = positions[node['name']]
            draw.line((x1, y1+32, x2, y2-32), fill=(59, 84, 112), width=3)
    for name, (x, y) in positions.items():
        fill = COLORS.get(name, (215, 228, 243))
        draw.rounded_rectangle((x-110, y-32, x+110, y+32),
                               radius=12, fill=fill, outline=(41, 63, 88), width=2)
        draw.text((x, y), name, anchor='mm', fill=(23, 36, 52), font=font(18))
    image.save(output)
def edge_lengths(matrix):
    """计算单位立方体沿自身 x、y、z 轴变换后的边长。"""
    return np.linalg.norm(matrix[:3, :3], axis=0)

def run_experiment(angle=35.0, offset=1.45, output=Path('e1_output')):
    if not 0.5 <= offset <= 1.65:
        raise ValueError('本章绘图案例要求偏移量在 0.5 到 1.65 之间')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    nodes = make_scene(angle, offset)
    wrong = make_scene(angle, offset, wrong=True)
    world = world_matrices(nodes)
    bad_world = world_matrices(wrong)
    a = point(world['display_A'])
    bad = point(bad_world['display_A'])
    piece = point(world['small_piece'])
    values = {
        'angle_deg': angle, 'offset': offset, 'node_count': len(nodes),
        'display_A_origin': a.tolist(),
        'display_B_origin': point(world['display_B']).tolist(),
        'display_A_edge_lengths': edge_lengths(world['display_A']).tolist(),
        'small_piece_edge_lengths': edge_lengths(world['small_piece']).tolist(),
        'wrong_order_origin': bad.tolist(),
        'small_piece_origin': piece.tolist(),
        'wrong_order_distance': float(np.linalg.norm(a-bad)),
        'hand_TR': point(translation(2, 0, 0) @ rotation_y(90), (1, 0, 0)).tolist(),
        'hand_RT': point(rotation_y(90) @ translation(2, 0, 0), (1, 0, 0)).tolist(),
    }
    draw_hierarchy(nodes, output / 'hierarchy.png')
    draw_scene(nodes, 'orthographic', output / 'orthographic.png', angle, offset)
    draw_scene(nodes, 'perspective', output / 'perspective.png', angle, offset)
    draw_scene(wrong, 'wrong_order', output / 'wrong_order.png', angle, offset)
    (output / 'metrics.json').write_text(
        json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(values, ensure_ascii=False, indent=2))
    return values

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--angle', type=float, default=35.0)
    parser.add_argument('--offset', type=float, default=1.45)
    parser.add_argument('--output', type=Path, default=Path('e1_output'))
    options = parser.parse_args()
    run_experiment(options.angle, options.offset, options.output)

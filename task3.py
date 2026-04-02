"""
Taichi 刚体碰撞 Demo - Lab 1
已实现2个box的碰撞检测 collision_manifold，3D 立方体可视化
请手动实现刚体运动、冲量碰撞响应，地面处理
"""

import taichi as ti
import numpy as np

# GGUI 需要 GPU，若无 GPU 可改为 ti.cpu
ti.init(arch=ti.gpu, default_fp=ti.f32)

# 常量
# 这里做四个方块的物理测试
N_BODIES = 4
# 加上重力
GRAVITY = ti.Vector([0.0, -9.8, 0.0])
DT = 1.0 / 60.0
RESTITUTION = 0.6
EPSILON = 1e-6
GROUND_RESTITUTION = 0.25#专门的地面恢复系数

# 立方体单位顶点 ([-1,1]^3)，8个顶点 - 用于 Taichi kernel 和 Python
CUBE_LOCAL_VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
], dtype=np.float32)

# 12 个三角形索引
CUBE_INDICES = np.array([
    0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
    0, 3, 7, 0, 7, 4, 1, 2, 6, 1, 6, 5,
    0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7,
], dtype=np.int32)

# Taichi 可用的几何数据
cube_local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
cube_indices_ti = ti.field(dtype=ti.i32, shape=36)
cube_local_verts.from_numpy(CUBE_LOCAL_VERTICES)
cube_indices_ti.from_numpy(CUBE_INDICES)

# 刚体状态字段
position = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_BODIES)
angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
half_extent = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
mass = ti.field(dtype=ti.f32, shape=N_BODIES)

# 可视化：每个立方体 8 顶点，总共 N_BODIES * 8 个顶点
mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES * 8)
mesh_indices = ti.field(dtype=ti.i32, shape=N_BODIES * 36)



@ti.kernel
def init_rigid_bodies():
    """初始化 2 个立方体做对撞，无重力"""
    # 立方体 0：左边，向右运动
    position[0] = ti.Vector([-1.0, 0.75, 0.2])
    velocity[0] = ti.Vector([1.0, 0.0, 0.0])
    rotation[0] = ti.Matrix.identity(ti.f32, 3)
    angular_velocity[0] = ti.Vector([0.0, 0.0, 0.0])
    half_extent[0] = ti.Vector([0.3, 0.3, 0.3])
    mass[0] = 1.0

    # 立方体 1：右边，向左运动
    position[1] = ti.Vector([1.0, 0.5, 0.0])
    velocity[1] = ti.Vector([-1.0, 0.0, 0.0])
    rotation[1] = ti.Matrix.identity(ti.f32, 3)
    angular_velocity[1] = ti.Vector([0.0, 0.0, 0.0])
    half_extent[1] = ti.Vector([0.3, 0.3, 0.3])
    mass[1] = 1.0

    # 立方体 2：下落并带旋转
    position[2] = ti.Vector([0.0, 3.0, 0.0])
    velocity[2] = ti.Vector([0.0, 0.0, 0.0])
    rotation[2] = ti.Matrix.identity(ti.f32, 3)
    angular_velocity[2] = ti.Vector([5.0, 0.0, 0.0])
    half_extent[2] = ti.Vector([0.3, 0.3, 0.3])
    mass[2] = 1.0

    # 立方体 3：下落并带旋转
    position[3] = ti.Vector([0.2, 5.0, 0.1])
    velocity[3] = ti.Vector([0.0, 0.0, 0.0])
    rotation[3] = ti.Matrix.identity(ti.f32, 3)
    angular_velocity[3] = ti.Vector([0.0, 5.0, 0.0])
    half_extent[3] = ti.Vector([0.4, 0.4, 0.4])
    mass[3] = 1.0

    #########
    # This is an example of two rigid boxes
    # modify the code to support more complex demos
    #########

@ti.func
def skew(w):
    return ti.Matrix([
        [0.0,   -w[2],  w[1]],
        [w[2],   0.0,  -w[0]],
        [-w[1],  w[0],  0.0]
    ])

@ti.func
def safe_normalize(v):
    n = ti.sqrt(v.dot(v) + 1e-12)
    return v / n

@ti.func
#施密特正交化
def orthonormalize(R):
    x = ti.Vector([R[0, 0], R[1, 0], R[2, 0]])
    y = ti.Vector([R[0, 1], R[1, 1], R[2, 1]])
    x = safe_normalize(x)
    y = y - x * x.dot(y)
    y = safe_normalize(y)
    z = x.cross(y)

    return ti.Matrix([
        [x[0], y[0], z[0]],
        [x[1], y[1], z[1]],
        [x[2], y[2], z[2]],
    ])


@ti.kernel
def integrate():
    """刚体运动积分"""
    #########
    # add your code here  
    # you may need some tool functions, e.g., skew(...)
    for i in range(N_BODIES):
        velocity[i] += GRAVITY * DT
        position[i] += velocity[i] * DT
        R=rotation[i]
        w=angular_velocity[i]
        R += skew(w) @ R * DT
        rotation[i] = orthonormalize(R)
    return


def get_box_vertices_correct(i: int) -> np.ndarray:
    """立方体顶点：局部 (±dx, ±dy, ±dz)"""
    pos = position[i].to_numpy()
    rot = rotation[i].to_numpy()
    ext = half_extent[i].to_numpy()
    verts = np.zeros((8, 3), dtype=np.float32)
    for k in range(8):
        local = CUBE_LOCAL_VERTICES[k] * ext
        verts[k] = rot @ local + pos
    return verts


def collision_manifold(i: int, j: int):
    """
    返回:
        collided: bool
        normal: (3,)
        penetration: float
        contact_point: (3,)
    """
    verts_a = get_box_vertices_correct(i)
    verts_b = get_box_vertices_correct(j)
    rot_a = rotation[i].to_numpy()
    rot_b = rotation[j].to_numpy()

    axes_a = [rot_a[:, k] for k in range(3)]
    axes_b = [rot_b[:, k] for k in range(3)]

    axes = axes_a + axes_b

    # 9 cross axes
    for ia in range(3):
        for ib in range(3):
            cross_axis = np.cross(axes_a[ia], axes_b[ib])
            n2 = np.dot(cross_axis, cross_axis)
            if n2 > EPSILON:
                axes.append(cross_axis / np.sqrt(n2))

    min_overlap = float('inf')
    best_axis = None

    center_a = position[i].to_numpy()
    center_b = position[j].to_numpy()

    for axis in axes:
        proj_a = verts_a @ axis
        proj_b = verts_b @ axis

        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()

        # 分离轴
        if max_a < min_b - EPSILON or max_b < min_a - EPSILON:
            return False, None, 0.0, None

        overlap = min(max_a - min_b, max_b - min_a)

        if overlap < min_overlap:
            min_overlap = overlap

            # 方向从 j 指向 i
            d = center_a - center_b
            if np.dot(axis, d) < 0:
                axis = -axis

            best_axis = axis

    if best_axis is None:
        return False, None, 0.0, None

    normal = best_axis / np.linalg.norm(best_axis)
    penetration = min_overlap

    # --------- 计算 contact point ---------
    # A 上沿 -normal 的 support 点
    idx_a = np.argmin(verts_a @ normal)
    pa = verts_a[idx_a]

    # B 上沿 +normal 的 support 点
    idx_b = np.argmax(verts_b @ normal)
    pb = verts_b[idx_b]

    contact_point = 0.5 * (pa + pb)

    return True, normal, penetration, contact_point

#逆惯性张量求解
def get_inv_inertia_world(i):
    m = float(mass[i])
    e = half_extent[i].to_numpy().astype(np.float32)
    I_body = np.array([
        [(1.0 / 3.0) * m * (e[1] * e[1] + e[2] * e[2]), 0.0, 0.0],
        [0.0, (1.0 / 3.0) * m * (e[0] * e[0] + e[2] * e[2]), 0.0],
        [0.0, 0.0, (1.0 / 3.0) * m * (e[0] * e[0] + e[1] * e[1])],
    ], dtype=np.float32)
    R = rotation[i].to_numpy().astype(np.float32)
    I_body_inv = np.linalg.inv(I_body)
    return R @ I_body_inv @ R.T

def resolve_collision_fixed(i: int, j: int, normal: np.ndarray, penetration: float, contact: np.ndarray):
    """冲量法碰撞响应 + 位置修正"""
    #########
    # add your code here to update position, velocity, and so on.
    pi=position[i].to_numpy()
    pj=position[j].to_numpy()
    vi=velocity[i].to_numpy()
    vj=velocity[j].to_numpy()
    mi=float(mass[i])
    mj=float(mass[j])
    wi=angular_velocity[i].to_numpy()
    wj=angular_velocity[j].to_numpy()
    inv_m_i = 1.0 / mi
    inv_m_j = 1.0 / mj
    inv_I_i = get_inv_inertia_world(i) #求解逆惯性张量
    inv_I_j = get_inv_inertia_world(j)
    ri=contact - pi
    rj=contact - pj
    #在这里询问ai后,采用percent和slop的方式进行位置修正,这样可以避免小抖动
    percent = 0.8
    slop = 0.001
    correction_mag = max(penetration - slop, 0.0) / (inv_m_i + inv_m_j) * percent
    correction = correction_mag * normal
    pi += inv_m_i * correction
    pj -= inv_m_j * correction
    v_contact_i = vi + np.cross(wi, ri)
    v_contact_j = vj + np.cross(wj, rj)
    rel_v=v_contact_i - v_contact_j
    vel_along_normal=np.dot(rel_v,normal)
    if vel_along_normal > 0:#如果相对速度沿法线方向是分离的,则不处理
        position[i] = ti.Vector(pi)
        position[j] = ti.Vector(pj)
        return
    r_i_cross_n = np.cross(ri, normal)
    r_j_cross_n = np.cross(rj, normal)
    denom = inv_m_i + inv_m_j#等效质量倒数
    denom += np.dot(normal,np.cross(inv_I_i @ r_i_cross_n, ri) +np.cross(inv_I_j @ r_j_cross_n, rj))
    if denom<1e-6:
        return
    jn = -(1.0 + RESTITUTION) * vel_along_normal / denom
    impulse = jn * normal
    vi += impulse * inv_m_i
    vj -= impulse * inv_m_j
    wi += inv_I_i @ np.cross(ri, impulse)
    wj -= inv_I_j @ np.cross(rj, impulse)
    position[i] = ti.Vector(pi)
    position[j] = ti.Vector(pj)
    angular_velocity[i] = ti.Vector(wi)
    angular_velocity[j] = ti.Vector(wj)
    velocity[i] = ti.Vector(vi)
    velocity[j] = ti.Vector(vj)

@ti.kernel
def update_mesh_vertices():
    """根据刚体状态更新可视化顶点"""
    for i in range(N_BODIES):
        pos = position[i]
        rot = rotation[i]
        ext = half_extent[i]
        for k in range(8):
            lv = cube_local_verts[k]
            local = ti.Vector([lv[0] * ext[0], lv[1] * ext[1], lv[2] * ext[2]])
            world = rot @ local + pos
            mesh_vertices[i * 8 + k] = world
        for t in range(12):
            for v in range(3):
                mesh_indices[i * 36 + t * 3 + v] = i * 8 + cube_indices_ti[t * 3 + v]

@ti.kernel
def apply_force(body_id: ti.i32, fx: ti.f32, fy: ti.f32, fz: ti.f32):
    velocity[body_id] += ti.Vector([fx, fy, fz]) * (DT / mass[body_id])

def main():
    init_rigid_bodies()
    update_mesh_vertices()

    window = ti.ui.Window("Rigid Body Collision - Lab 1", res=(1024, 768))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(3, 2, 3)
    camera.lookat(0, 0.5, 0)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.ambient_light((0.6, 0.6, 0.6))
    scene.point_light((5, 5, 5), (1.2, 1.2, 1.2))

    colors = [(0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.3, 0.8, 0.3), (0.8, 0.8, 0.2)]

    # --------------------------
    # 创建地板
    floor_size = 5.0
    floor_color = (0.7, 0.7, 0.7)

    floor_vertices = np.array([
        [-floor_size, 0.0, -floor_size],
        [ floor_size, 0.0, -floor_size],
        [ floor_size, 0.0,  floor_size],
        [-floor_size, 0.0,  floor_size],
    ], dtype=np.float32)
    floor_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

    floor_vertices_ti = ti.Vector.field(3, dtype=ti.f32, shape=4)
    floor_indices_ti = ti.field(dtype=ti.i32, shape=6)
    floor_vertices_ti.from_numpy(floor_vertices)
    floor_indices_ti.from_numpy(floor_indices)
    # --------------------------

    drag_last_pos = None
    drag_target = -1

    while window.running:
        if window.is_pressed(ti.ui.LMB):
            mx, my = window.get_cursor_pos()
            if drag_last_pos is not None:
                dx = mx - drag_last_pos[0]
                dy = my - drag_last_pos[1]
                fx = dx * 500.0
                fz = -dy * 500.0
                if drag_target == 0:
                    apply_force(0, fx, 0.0, fz)
                elif drag_target == 1:
                    apply_force(1, fx, 0.0, fz)
            else:
                drag_target = 0 if mx < 0.5 else 1
            drag_last_pos = (mx, my)
        else:
            drag_last_pos = None
            drag_target = -1

        for _ in range(2):  # 子步提高稳定性
            integrate()
            ti.sync()
            for i in range(N_BODIES):
                for j in range(i + 1, N_BODIES):
                    collided, normal, penetration, contact = collision_manifold(i, j)

                    if collided:
                        resolve_collision_fixed(i, j, normal, penetration, contact)

        # 地面碰撞（取 OBB 顶点最小 y）
        for i in range(N_BODIES):
            verts = get_box_vertices_correct(i)
            min_y = float(verts[:, 1].min())
            if min_y < 0:
                print(i,verts)
                pi = position[i].to_numpy()
                vi = velocity[i].to_numpy()
                wi = angular_velocity[i].to_numpy()
                inv_m = 1.0 / float(mass[i])
                inv_I = get_inv_inertia_world(i)
                pi[1] -= min_y
                ids = np.where(np.abs(verts[:, 1] - min_y) <  1e-4)[0]
                contact = verts[ids].mean(axis=0)
                contact[1] = 0.0
                normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                r = contact - pi
                v_contact = vi + np.cross(wi, r)
                vel_along_normal = np.dot(v_contact, normal)
                if vel_along_normal < 0.0:
                    r_cross_n = np.cross(r, normal)
                    denom = inv_m + np.dot(normal, np.cross(inv_I @ r_cross_n, r))
                    if denom > 1e-6:
                        jn = -(1.0 + GROUND_RESTITUTION) * vel_along_normal / denom
                        impulse = jn * normal
                        vi += impulse * inv_m
                        wi += inv_I @ np.cross(r, impulse)
                if abs(vi[1]) < 1e-3:
                    vi[1] = 0.0
                position[i] = ti.Vector(pi)
                velocity[i] = ti.Vector(vi)
                angular_velocity[i] = ti.Vector(wi)
                #########


        update_mesh_vertices()

        scene.set_camera(camera)
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)

        # 设置背景色（例如浅灰）
        canvas = window.get_canvas()
        canvas.set_background_color((0.8, 0.8, 0.85))

        for body_id in range(N_BODIES):
            # 先画实心面片
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=colors[body_id],
                index_offset=body_id * 36,
                index_count=36,
            )
            # 再叠加一层线框轮廓
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=(0.0, 0.0, 0.0),
                index_offset=body_id * 36,
                index_count=36,
                show_wireframe=True,
            )

        # 画地板
        scene.mesh(floor_vertices_ti, floor_indices_ti, color=floor_color)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()

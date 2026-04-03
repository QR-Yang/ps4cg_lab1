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
# 模拟稳定堆叠 3 个方块
N_BODIES = 3
# 加上重力
GRAVITY = ti.Vector([0.0, -9.8, 0.0])
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 2
SOLVER_ITERS = 6
DT = FRAME_DT / SUBSTEPS
LINEAR_DAMPING = 0.999
ANGULAR_DAMPING = 0.999
RESTITUTION = 0.05
EPSILON = 1e-6
GROUND_RESTITUTION = 0.05
GROUND_PENETRATION_SLOP = 1e-4
GROUND_BAUMGARTE = 0.35
GROUND_SLEEP_LINEAR = 0.05
GROUND_SLEEP_ANGULAR = 0.05
GROUND_SLEEP_MIN_Y = 0.01
POSITION_SLOP = 1e-3
POSITION_PERCENT = 0.40
DRAG_FORCE_SCALE = 5000.0

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
    """初始化 3 个立方体稳定堆叠"""
    # 让上面 3 个箱子竖直叠放，带些微小错位可检验稳定性
    for i in range(3):
        position[i] = ti.Vector([0.0, 0.35 + i * 0.70, 0.0])
        velocity[i] = ti.Vector([0.0, 0.0, 0.0])
        rotation[i] = ti.Matrix.identity(ti.f32, 3)
        angular_velocity[i] = ti.Vector([0.0, 0.0, 0.0])
        half_extent[i] = ti.Vector([0.35, 0.35, 0.35])
        mass[i] = 1.0

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
        velocity[i] *= LINEAR_DAMPING
        angular_velocity[i] *= ANGULAR_DAMPING
        
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
    percent = POSITION_PERCENT
    slop = POSITION_SLOP
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
    v_contact_i_after = vi + np.cross(wi, ri)
    v_contact_j_after = vj + np.cross(wj, rj)
    rel_v_after = v_contact_i_after - v_contact_j_after
    
    tangent = rel_v_after - np.dot(rel_v_after, normal) * normal
    tan_len = np.linalg.norm(tangent)
    if tan_len > 1e-6:
        tangent = tangent / tan_len
        r_i_cross_t = np.cross(ri, tangent)
        r_j_cross_t = np.cross(rj, tangent)
        denom_t = inv_m_i + inv_m_j + np.dot(tangent, np.cross(inv_I_i @ r_i_cross_t, ri) + np.cross(inv_I_j @ r_j_cross_t, rj))    
        if denom_t > 1e-6:
            jt = -np.dot(rel_v_after, tangent) / denom_t
            friction_coeff = 0.5
            if abs(jt) > jn * friction_coeff:
                jt = np.sign(jt) * jn * friction_coeff
                
            friction_impulse = jt * tangent
            vi += friction_impulse * inv_m_i
            vj -= friction_impulse * inv_m_j
            wi += inv_I_i @ np.cross(ri, friction_impulse)
            wj -= inv_I_j @ np.cross(rj, friction_impulse)

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
    velocity[body_id] += ti.Vector([fx, fy, fz]) * (FRAME_DT / mass[body_id])
def resolve_ground(i: int):
    verts = get_box_vertices_correct(i)
    ys = verts[:, 1]
    min_y = float(np.min(ys))
    if min_y >= 0.0:
        return
    p = position[i].to_numpy().astype(np.float32)
    v = velocity[i].to_numpy().astype(np.float32)
    w = angular_velocity[i].to_numpy().astype(np.float32)
    inv_m = 1.0 / float(mass[i])
    inv_I = get_inv_inertia_world(i)
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    ids = np.where(np.abs(ys - min_y) < 1e-4)[0]
    if len(ids) == 0:
        contact = verts[np.argmin(ys)].copy()
    else:
        contact = verts[ids].mean(axis=0).astype(np.float32)
    contact[1] = 0.0
    penetration = -min_y
    correction = max(penetration - GROUND_PENETRATION_SLOP, 0.0) * GROUND_BAUMGARTE
    p[1] += correction
    r = contact - p
    v_contact = v + np.cross(w, r)
    vel_along_normal = np.dot(v_contact, normal)
    if vel_along_normal < 0.0:
        r_cross_n = np.cross(r, normal)
        denom = inv_m + np.dot(normal, np.cross(inv_I @ r_cross_n, r))

        if denom > 1e-8 and np.isfinite(denom):
            jn = -(1.0 + GROUND_RESTITUTION) * vel_along_normal / denom
            impulse = jn * normal
            v += impulse * inv_m
            w += inv_I @ np.cross(r, impulse)
    v[0] *= 0.999
    v[2] *= 0.999
    w *= 0.9998
    if (
        np.linalg.norm(v) < GROUND_SLEEP_LINEAR and
        np.linalg.norm(w) < GROUND_SLEEP_ANGULAR and
        abs(min_y) < GROUND_SLEEP_MIN_Y
    ):
        v[:] = 0.0
        w[:] = 0.0
    position[i] = ti.Vector(p)
    velocity[i] = ti.Vector(v)
    angular_velocity[i] = ti.Vector(w)

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

    colors = [
        (0.85, 0.25, 0.25),
        (0.20, 0.60, 0.85),
        (0.25, 0.80, 0.35),
        (0.85, 0.70, 0.25),
    ]

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

    active_target = 0

    while window.running:
        while window.get_event(ti.ui.PRESS):
            if window.event.key == 'q':
                active_target = 0
            elif window.event.key == 'w':
                active_target = 1
            elif window.event.key == 'e':
                active_target = 2
            elif window.event.key == 'r':
                active_target = 3

        if active_target != -1:
            fx = 0.0
            fz = 0.0
            move_force = 50.0
            
            if window.is_pressed('i'):
                fz -= move_force  # 前
            if window.is_pressed('k'):
                fz += move_force  # 后
            if window.is_pressed('j'):
                fx -= move_force  # 左
            if window.is_pressed('l'):
                fx += move_force  # 右
                
            if fx != 0.0 or fz != 0.0:
                apply_force(active_target, fx, 0.0, fz)

        for _ in range(SUBSTEPS):
            integrate()
            ti.sync()

            for _ in range(SOLVER_ITERS):
                # 刚体-刚体
                for i in range(N_BODIES):
                    for j in range(i + 1, N_BODIES):
                        collided, normal, penetration, contact = collision_manifold(i, j)
                        if collided:
                            resolve_collision_fixed(i, j, normal, penetration, contact)

                # 刚体-地面
                for i in range(N_BODIES):
                    resolve_ground(i)

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

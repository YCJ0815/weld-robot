# T型断点检测逻辑详解

## 核心算法流程

```
对每条过焊孔边 (hole_edge):
  ├─ 1. 找到过焊孔边的两个端点
  │   ├─ hole_start_pt = 过焊孔边起点
  │   └─ hole_end_pt = 过焊孔边终点
  │
  ├─ 2. 找到连接到这两个端点的节点
  │   ├─ nodes_at_hole = [(nid_start, "start"), (nid_end, "end")]
  │   └─ 如果找不到2个节点 → 跳过
  │
  ├─ 3. 获取这两个节点的关联焊缝
  │   ├─ welds_at_start = 连接到hole_start的焊缝列表
  │   ├─ welds_at_end = 连接到hole_end的焊缝列表
  │   └─ 如果任一为空 → 跳过
  │
  ├─ 4. 尝试所有焊缝对 (weld_a, weld_b)
  │   │   其中 weld_a ∈ welds_at_start, weld_b ∈ welds_at_end
  │   │
  │   ├─ 4.1 确定焊缝在过焊孔边处的端点
  │   │   ├─ weld_a_at_hole = weld_a 靠近 hole_start 的端点
  │   │   ├─ weld_a_away = weld_a 远离 hole_start 的端点
  │   │   ├─ weld_b_at_hole = weld_b 靠近 hole_end 的端点
  │   │   └─ weld_b_away = weld_b 远离 hole_end 的端点
  │   │
  │   ├─ 4.2 检查共面性
  │   │   ├─ 计算四面体体积: V = |det([hole_dir, weld_a_away-hole_start, weld_b_away-hole_start])|
  │   │   ├─ 归一化: norm_vol = V / (|hole_dir| * |weld_a_away-hole_start| * |weld_b_away-hole_start|)
  │   │   ├─ 如果 norm_vol > t_type_coplanar_tol (默认2.0) → 不共面，跳过
  │   │   └─ ✓ 共面检查通过
  │   │
  │   ├─ 4.3 计算焊缝方向
  │   │   ├─ weld_a_dir = normalize(weld_a_away - weld_a_at_hole)
  │   │   └─ weld_b_dir = normalize(weld_b_away - weld_b_at_hole)
  │   │
  │   ├─ 4.4 检查焊缝夹角
  │   │   ├─ angle = arccos(dot(weld_a_dir, weld_b_dir))
  │   │   ├─ angle_deg = degrees(angle)
  │   │   ├─ 如果 angle_deg < 15° 或 angle_deg > 165° → 跳过
  │   │   └─ ✓ 角度检查通过
  │   │
  │   ├─ 4.5 计算焊缝延长线的交点
  │   │   ├─ 直线A: p = weld_a_at_hole + t * weld_a_dir
  │   │   ├─ 直线B: p = weld_b_at_hole + s * weld_b_dir
  │   │   ├─ 求解: weld_a_at_hole + t*weld_a_dir = weld_b_at_hole + s*weld_b_dir
  │   │   ├─ 得到交点: intersection_pt
  │   │   ├─ 检查距离: dist_a = |intersection_pt - weld_a_at_hole|
  │   │   │           dist_b = |intersection_pt - weld_b_at_hole|
  │   │   ├─ extension_len = 2.0 * |hole_end - hole_start|
  │   │   ├─ 如果 dist_a > extension_len 或 dist_b > extension_len → 跳过
  │   │   └─ ✓ 交点在有效范围内
  │   │
  │   └─ 4.6 查找第三条焊缝C
  │       ├─ 对每条焊缝 weld_c (不是 weld_a 或 weld_b):
  │       │   ├─ 计算 intersection_pt 到 weld_c 的最近距离
  │       │   ├─ 方法: 投影 intersection_pt 到 weld_c 线段
  │       │   │   seg_dir = normalize(weld_c_end - weld_c_start)
  │       │   │   t = dot(intersection_pt - weld_c_start, seg_dir)
  │       │   │   t = clamp(t, 0, 1)  # 限制在线段范围内
  │       │   │   closest_pt = weld_c_start + t * seg_dir
  │       │   │   dist = |intersection_pt - closest_pt|
  │       │   └─ 记录最小距离的焊缝
  │       │
  │       ├─ 如果找到焊缝C且距离 < 10.0 mm (默认)
  │       └─ ✓ 找到T型断点！
  │           └─ 记录: {hole_edge_id, weld_a_id, weld_b_id, weld_c_id, intersection_point}
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `t_type_coplanar_tol` | 2.0 | 共面性容差（四面体体积/边长乘积） |
| `t_type_angle_min_deg` | 15.0 | 焊缝A、B最小夹角（度） |
| `t_type_angle_max_deg` | 165.0 | 焊缝A、B最大夹角（度） |
| `t_type_extension_ratio` | 2.0 | 焊缝延长比例（相对于过焊孔边长度） |
| `t_type_max_distance_to_weld` | 10.0 | 交点到焊缝C的最大距离（mm） |

## 可能的问题点

### 1. **共面性检查失败**
- 症状：`not_coplanar` 计数很高
- 原因：焊缝A、B与过焊孔边不在同一平面
- 解决：增加 `t_type_coplanar_tol` 参数

### 2. **角度检查失败**
- 症状：`angle_out_of_range` 计数很高
- 原因：焊缝夹角不在 15°~165° 范围内
- 解决：调整 `t_type_angle_min_deg` 和 `t_type_angle_max_deg`

### 3. **交点查找失败**
- 症状：`no_intersection` 计数很高
- 原因：
  - 焊缝延长线不相交（平行或异面）
  - 交点超出延长范围
- 解决：增加 `t_type_extension_ratio`

### 4. **找不到焊缝C**
- 症状：找到交点但最后仍然失败
- 原因：没有焊缝在交点附近
- 解决：增加 `t_type_max_distance_to_weld`

## 调试方法

运行带 `--debug` 标志的脚本：

```bash
python debug_t_type.py
```

输出示例：
```
[T-type-debug] hole=G1 weld_a=W2 weld_b=W3 coplanar_check: norm_vol=0.123456 tol=2.0 result=True
[T-type-debug] hole=G1 weld_a=W2 weld_b=W3 angle_check: angle_deg=90.5° min=15.0° max=165.0°
[T-type-debug] hole=G1 weld_a=W2 weld_b=W3 intersection_check: found=True
  -> intersection found at dist_a=5.234 dist_b=6.123 extension_len=20.000
  -> ACCEPTED: found weld_c=W5 at distance 3.456
```

## 数学细节

### 共面性检查
四个点 P1, P2, P3, P4 共面当且仅当：
```
|(P2-P1) · ((P3-P1) × (P4-P1))| / (|P2-P1| * |P3-P1| * |P4-P1|) < tolerance
```

### 直线交点计算
求解线性系统：
```
p1 + t*d1 = p2 + s*d2

矩阵形式:
[d1 | -d2] [t]   [p2 - p1]
           [s] = 

使用克拉默法则求解 t 和 s
```

### 点到线段距离
```
seg_dir = normalize(seg_end - seg_start)
t = dot(point - seg_start, seg_dir)
t = clamp(t, 0, 1)  # 限制在线段范围
closest_point = seg_start + t * seg_dir
distance = |point - closest_point|
```

## 常见配置

### 严格模式（高精度）
```python
t_type_coplanar_tol = 0.5
t_type_angle_min_deg = 30.0
t_type_angle_max_deg = 150.0
t_type_extension_ratio = 1.5
t_type_max_distance_to_weld = 5.0
```

### 宽松模式（高召回率）
```python
t_type_coplanar_tol = 5.0
t_type_angle_min_deg = 10.0
t_type_angle_max_deg = 170.0
t_type_extension_ratio = 3.0
t_type_max_distance_to_weld = 15.0
```

### 平衡模式（默认）
```python
t_type_coplanar_tol = 2.0
t_type_angle_min_deg = 15.0
t_type_angle_max_deg = 165.0
t_type_extension_ratio = 2.0
t_type_max_distance_to_weld = 10.0
```


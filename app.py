import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import symbols, diff, lambdify, parse_expr

# --- 页面设置 ---
st.set_page_config(page_title="偏导数可视化工具", layout="wide")

st.title("Calculus MAT201: Partial Derivatives Visualizer")
st.markdown("""
这个应用旨在帮助理解 **偏导数 (Partial Derivatives)** 的几何意义：
1. **切线斜率 (Rate of Change)**: 偏导数代表曲面在某一点沿 x 或 y 方向的切线斜率。
2. **切片法 (Slicing)**: 求对 x 的偏导时，相当于固定 y (用平面切开曲面)。
""")

# --- 侧边栏：输入与控制 ---
st.sidebar.header("1. 输入函数与参数")
func_input = st.sidebar.text_input("输入二元函数 f(x, y):", value="x**2 + y**2")
st.sidebar.caption("提示: 使用 Python 语法, 例如 x**2 代表 x的平方, sin(x) 等。")

# 定义范围
x_range = st.sidebar.slider("X 轴范围", -5.0, 5.0, (-2.0, 2.0))
y_range = st.sidebar.slider("Y 轴范围", -5.0, 5.0, (-2.0, 2.0))

st.sidebar.header("2. 选择求导点与方向")
x0 = st.sidebar.slider("x0 (固定点 x)", x_range[0], x_range[1], 1.0)
y0 = st.sidebar.slider("y0 (固定点 y)", y_range[0], y_range[1], 1.0)
derivative_type = st.sidebar.radio("选择求偏导方向:", ["对 x 求偏导 (∂f/∂x)", "对 y 求偏导 (∂f/∂y)"])

# --- 数学计算逻辑 ---
try:
    x, y = symbols('x y')
    expr = parse_expr(func_input)
    f = lambdify((x, y), expr, 'numpy')

    # 计算偏导数
    if derivative_type == "对 x 求偏导 (∂f/∂x)":
        deriv_expr = diff(expr, x)
        fixed_var_val = y0
        varying_var_str = 'x'
        fixed_var_str = 'y'
        title_text = f"固定 y = {y0}, 观察 x 方向的变化"
    else:
        deriv_expr = diff(expr, y)
        fixed_var_val = x0
        varying_var_str = 'y'
        fixed_var_str = 'x'
        title_text = f"固定 x = {x0}, 观察 y 方向的变化"
    
    deriv_f = lambdify((x, y), deriv_expr, 'numpy')
    z0 = f(x0, y0)
    slope = deriv_f(x0, y0)

    # --- 3D 可视化 ---
    # 生成网格数据
    x_vals = np.linspace(x_range[0], x_range[1], 50)
    y_vals = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    fig = go.Figure()

    # 1. 绘制主曲面
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='f(x,y)'))

    # 2. 绘制切片平面 (Slicing Plane) & 3. 绘制切线 (Tangent Line)
    if derivative_type == "对 x 求偏导 (∂f/∂x)":
        # 切平面: y = y0
        # 更好的切平面画法：画一个透明平面
        fig.add_trace(go.Scatter3d(x=x_vals, y=np.full_like(x_vals, y0), z=f(x_vals, y0),
                                   mode='lines', line=dict(color='red', width=5), name=f'截线 (y={y0})'))
        
        # 切线
        tangent_x = np.linspace(x0 - 1, x0 + 1, 10)
        tangent_z = z0 + slope * (tangent_x - x0)
        fig.add_trace(go.Scatter3d(x=tangent_x, y=np.full_like(tangent_x, y0), z=tangent_z,
                                   mode='lines', line=dict(color='yellow', width=8), name='切线'))
    else:
        # 对 y 求偏导
        fig.add_trace(go.Scatter3d(x=np.full_like(y_vals, x0), y=y_vals, z=f(x0, y_vals),
                                   mode='lines', line=dict(color='red', width=5), name=f'截线 (x={x0})'))
        
        # 切线
        tangent_y = np.linspace(y0 - 1, y0 + 1, 10)
        tangent_z = z0 + slope * (tangent_y - y0)
        fig.add_trace(go.Scatter3d(x=np.full_like(tangent_y, x0), y=tangent_y, z=tangent_z,
                                   mode='lines', line=dict(color='yellow', width=8), name='切线'))

    # 标记点 P(x0, y0, z0)
    fig.add_trace(go.Scatter3d(x=[x0], y=[y0], z=[z0], mode='markers', marker=dict(size=8, color='red'), name='点 P'))

    fig.update_layout(title=title_text, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), width=800, height=600)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("计算结果")
        st.latex(f"f({x}, {y}) = " + str(expr))
        st.write(f"在点 $({x0}, {y0})$ 处:")
        st.info(f"偏导数表达式: {deriv_expr}")
        st.success(f"计算出的斜率 (Slope): {slope:.4f}")
        st.write("---")
        st.markdown(f"**解释**: 当 ${fixed_var_str}$ 固定在 {fixed_var_val} 时，${varying_var_str}$ 每变化 1 个单位，函数值 $z$ 大约变化 {slope:.4f} 个单位。")

except Exception as e:
    st.error(f"输入函数有误，请检查语法。错误信息: {e}")

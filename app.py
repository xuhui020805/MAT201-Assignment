import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import symbols, diff, lambdify, parse_expr

# --- Page Config ---
st.set_page_config(page_title="Partial Derivatives Visualizer", layout="wide")

st.title("Calculus MAT201: Partial Derivatives Visualizer")
st.markdown("""
This application visualizes the geometric meaning of **Partial Derivatives**:
1. **Rate of Change**: The partial derivative represents the slope of the tangent line along the x or y direction.
2. **Slicing Method**: Finding the partial derivative with respect to x implies holding y constant (slicing the surface with a plane).
""")

# --- Sidebar: Inputs ---
st.sidebar.header("1. Function & Parameters")
func_input = st.sidebar.text_input("Enter function f(x, y):", value="x**2 + y**2")
st.sidebar.caption("Hint: Use Python syntax, e.g., x**2 for x^2, sin(x), etc.")

# Define Ranges
x_range = st.sidebar.slider("X Axis Range", -5.0, 5.0, (-2.0, 2.0))
y_range = st.sidebar.slider("Y Axis Range", -5.0, 5.0, (-2.0, 2.0))

st.sidebar.header("2. Derivative Settings")
x0 = st.sidebar.slider("Point x0", x_range[0], x_range[1], 1.0)
y0 = st.sidebar.slider("Point y0", y_range[0], y_range[1], 1.0)
derivative_type = st.sidebar.radio("Choose Derivative Direction:", 
                                   ["Partial w.r.t x (∂f/∂x)", "Partial w.r.t y (∂f/∂y)"])

# --- Calculation Logic ---
try:
    x, y = symbols('x y')
    expr = parse_expr(func_input)
    f = lambdify((x, y), expr, 'numpy')

    # Calculate Derivatives
    if derivative_type == "Partial w.r.t x (∂f/∂x)":
        deriv_expr = diff(expr, x)
        fixed_var_val = y0
        varying_var_str = 'x'
        fixed_var_str = 'y'
        title_text = f"Visual: Holding y = {y0}, observing change along x"
        plane_color = 'red'
    else:
        deriv_expr = diff(expr, y)
        fixed_var_val = x0
        varying_var_str = 'y'
        fixed_var_str = 'x'
        title_text = f"Visual: Holding x = {x0}, observing change along y"
        plane_color = 'blue'
    
    deriv_f = lambdify((x, y), deriv_expr, 'numpy')
    z0 = f(x0, y0)
    slope = deriv_f(x0, y0)

    # --- 3D Visualization ---
    x_vals = np.linspace(x_range[0], x_range[1], 50)
    y_vals = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    fig = go.Figure()

    # 1. Plot Surface
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='Surface f(x,y)'))

    # 2. Plot Slicing Curve & Tangent Line
    if derivative_type == "Partial w.r.t x (∂f/∂x)":
        # Trace Curve (y held constant)
        fig.add_trace(go.Scatter3d(x=x_vals, y=np.full_like(x_vals, y0), z=f(x_vals, y0),
                                   mode='lines', line=dict(color='red', width=5), name=f'Trace (y={y0})'))
        
        # Tangent Line
        tangent_x = np.linspace(x0 - 1, x0 + 1, 10)
        tangent_z = z0 + slope * (tangent_x - x0)
        fig.add_trace(go.Scatter3d(x=tangent_x, y=np.full_like(tangent_x, y0), z=tangent_z,
                                   mode='lines', line=dict(color='yellow', width=8), name='Tangent Line'))
    else:
        # Trace Curve (x held constant)
        fig.add_trace(go.Scatter3d(x=np.full_like(y_vals, x0), y=y_vals, z=f(x0, y_vals),
                                   mode='lines', line=dict(color='red', width=5), name=f'Trace (x={x0})'))
        
        # Tangent Line
        tangent_y = np.linspace(y0 - 1, y0 + 1, 10)
        tangent_z = z0 + slope * (tangent_y - y0)
        fig.add_trace(go.Scatter3d(x=np.full_like(tangent_y, x0), y=tangent_y, z=tangent_z,
                                   mode='lines', line=dict(color='yellow', width=8), name='Tangent Line'))

    # Mark Point P
    fig.add_trace(go.Scatter3d(x=[x0], y=[y0], z=[z0], mode='markers', marker=dict(size=8, color='red'), name='Point P'))

    fig.update_layout(title=title_text, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), width=800, height=600)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Calculation Results")
        st.latex(f"f({x}, {y}) = " + str(expr))
        st.write(f"At point $({x0}, {y0})$:")
        st.info(f"Symbolic Derivative: {deriv_expr}")
        st.success(f"Calculated Slope: {slope:.4f}")
        st.write("---")
        st.markdown(f"**Interpretation**: When ${fixed_var_str}$ is fixed at {fixed_var_val}, for every 1 unit change in ${varying_var_str}$, the function value $z$ changes by approximately {slope:.4f} units.")

except Exception as e:
    st.error(f"Input Error: {e}")

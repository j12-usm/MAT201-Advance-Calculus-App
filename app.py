import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Multivariable Calculus App", layout="wide")
st.title("📐 Multivariable Calculus Learning App")

x, y = sp.symbols("x y", real=True)

# -----------------------------
# Safe function parser
# -----------------------------
def parse_function(expr_input):
    try:
        expr_input = expr_input.replace("^", "**")
        f = sp.sympify(expr_input, locals={
            "x": x, "y": y,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "exp": sp.exp, "sqrt": sp.sqrt, "ln": sp.log, "log": sp.log, "e": sp.E,
        })
        return f, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Domain analyzer
# -----------------------------
def analyze_domain(expr):
    conditions = []
    for power in expr.atoms(sp.Pow):
        if power.exp == sp.Rational(1, 2):
            conditions.append(sp.latex(power.base) + r" \ge 0")
    for arg in expr.atoms(sp.log):
        conditions.append(sp.latex(arg.args[0]) + r" > 0")
    denom = sp.denom(expr)
    if denom != 1:
        conditions.append(sp.latex(denom) + r" \ne 0")
    for arg in expr.atoms(sp.asin, sp.acos):
        u = arg.args[0]
        conditions.append(r"-1 \le " + sp.latex(u) + r" \le 1")
    if conditions:
        return r"\{(x,y)\in\mathbb{R}^2 \mid " + ",\ ".join(dict.fromkeys(conditions)) + r"\}"
    else:
        return r"\mathbb{R}^2"

# -----------------------------
# Sidebar topic
# -----------------------------
topic = st.sidebar.selectbox("Select Topic", [
    "Function of Two Variables",
    "Partial Derivatives",
    "Differentials"
])

# -----------------------------
# Axis limits
# -----------------------------
X_LIMITS = (-20.0, 20.0)
Y_LIMITS = (-20.0, 20.0)

if "x_min" not in st.session_state:
    st.session_state.x_min, st.session_state.x_max = -5.0, 5.0
    st.session_state.y_min, st.session_state.y_max = -5.0, 5.0

# =================================================
# 1. Function of Two Variables
# =================================================
if topic == "Function of Two Variables":
    st.header("Function of Two Variables")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()
    st.latex(f"f(x,y) = {sp.latex(f)}")

    # Evaluation point
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0, min_value=X_LIMITS[0], max_value=X_LIMITS[1])
    with col2:
        y0 = st.number_input("y₀", value=1.0, min_value=Y_LIMITS[0], max_value=Y_LIMITS[1])

    # Axis ranges
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.x_min = st.number_input("x min", value=st.session_state.x_min, min_value=X_LIMITS[0], max_value=X_LIMITS[1])
        st.session_state.x_max = st.number_input("x max", value=st.session_state.x_max, min_value=X_LIMITS[0], max_value=X_LIMITS[1])
        if st.session_state.x_min >= st.session_state.x_max:
            st.session_state.x_min = st.session_state.x_max - 0.1
        st.session_state.x_min, st.session_state.x_max = st.slider(
            "Adjust x-range",
            min_value=st.session_state.x_min,
            max_value=st.session_state.x_max,
            value=(st.session_state.x_min, st.session_state.x_max)
        )
    with col4:
        st.session_state.y_min = st.number_input("y min", value=st.session_state.y_min, min_value=Y_LIMITS[0], max_value=Y_LIMITS[1])
        st.session_state.y_max = st.number_input("y max", value=st.session_state.y_max, min_value=Y_LIMITS[0], max_value=Y_LIMITS[1])
        if st.session_state.y_min >= st.session_state.y_max:
            st.session_state.y_min = st.session_state.y_max - 0.1
        st.session_state.y_min, st.session_state.y_max = st.slider(
            "Adjust y-range",
            min_value=st.session_state.y_min,
            max_value=st.session_state.y_max,
            value=(st.session_state.y_min, st.session_state.y_max)
        )

    x_min, x_max = st.session_state.x_min, st.session_state.x_max
    y_min, y_max = st.session_state.y_min, st.session_state.y_max

    st.subheader("Domain")
    st.latex(analyze_domain(f))

    # Plot
    f_np = sp.lambdify((x, y), f, "numpy")
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
    Z = f_np(X, Y)
    Z = np.where(np.isfinite(Z), Z, np.nan)

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(projection="3d")
    z0 = f_np(x0, y0)
    ax.plot_surface(X, Y, Z, alpha=0.8, cmap="viridis")
    ax.scatter(x0, y0, z0, color="red", s=50)

    # Label
    label_offset = (np.nanmax(Z)-np.nanmin(Z))*0.05
    x_label = np.clip(x0 + label_offset, x_min, x_max)
    y_label = np.clip(y0 + label_offset, y_min, y_max)
    z_label = np.clip(z0 + label_offset, np.nanmin(Z), np.nanmax(Z))
    ax.text(x_label, y_label, z_label, f"({x0:.2f}, {y0:.2f}, {z0:.2f})", fontsize=10, ha='left', va='bottom')

    ax.view_init(elev=30, azim=45)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    st.pyplot(fig)
    st.success(f"f({x0}, {y0}) = {z0:.3f}")

# =================================================
# 2. Partial Derivatives
# =================================================
elif topic == "Partial Derivatives":
    st.header("Partial Derivatives as Rate of Change")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + x*y")
    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()

    # Symbolic partial derivatives
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx))
    st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy))

    # Evaluation point
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0)
    with col2:
        y0 = st.number_input("y₀", value=1.0)

    st.success(
        f"At ({x0}, {y0}): "
        f"∂f/∂x = {float(fx.subs({x:x0, y:y0})):.3f}, "
        f"∂f/∂y = {float(fy.subs({x:x0, y:y0})):.3f}"
    )

    # Numeric lambdas for partial derivatives
t = np.linspace(-3, 3, 100)

# Ensure array output
fx_np = sp.lambdify(x, fx.subs(y, y0), "numpy")
fy_np = sp.lambdify(y, fy.subs(x, x0), "numpy")

# -------------------------------------------------
# Graph 1: Plane y = y0 → ∂f/∂x vs x
# -------------------------------------------------
st.markdown(
    rf"**Graph 1:** Rate of change of $f$ with respect to $x$ "
    rf"when the plane $y = {y0}$ is fixed."
)

fig_x, ax_x = plt.subplots()
ax_x.plot(t, fx_np(t))
ax_x.set_title(r"Rate of Change $\partial f / \partial x$ (y fixed)")
ax_x.set_xlabel("x")
ax_x.set_ylabel(r"$\partial f / \partial x$")
st.pyplot(fig_x)

# -------------------------------------------------
# Graph 2: Plane x = x0 → ∂f/∂y vs y
# -------------------------------------------------
st.markdown(
    rf"**Graph 2:** Rate of change of $f$ with respect to $y$ "
    rf"when the plane $x = {x0}$ is fixed."
)

fig_y, ax_y = plt.subplots()
ax_y.plot(t, fy_np(t))
ax_y.set_title(r"Rate of Change $\partial f / \partial y$ (x fixed)")
ax_y.set_xlabel("y")
ax_y.set_ylabel(r"$\partial f / \partial y$")
st.pyplot(fig_y)

# =================================================
# 3. Differentials
# =================================================
elif topic == "Differentials":
    st.header("Differentials and Linear Approximation")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.latex(r"f_x = " + sp.latex(fx))
    st.latex(r"f_y = " + sp.latex(fy))

    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0)
        y0 = st.number_input("y₀", value=1.0)
    with col2:
        dx = st.number_input("dx", value=0.1)
        dy = st.number_input("dy", value=0.1)

    dx_sym, dy_sym = sp.symbols("dx dy")
    df_symbolic = fx*dx_sym + fy*dy_sym

    st.latex(r"df = f_x dx + f_y dy")
    st.latex(r"df = " + sp.latex(df_symbolic))

    df_subs = df_symbolic.subs({dx_sym:dx, dy_sym:dy})
    df_numeric = df_subs.subs({x:x0, y:y0})
    f_np = sp.lambdify((x,y), f,"numpy")
    actual_change = f_np(x0+dx, y0+dy) - f_np(x0, y0)

    st.success(f"df ≈ {float(df_numeric):.5f}")
    st.info(f"Actual Δf = {actual_change:.5f}")
    st.warning(f"Approximation error = {abs(actual_change-float(df_numeric)):.5e}")


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

    expr_input = st.text_input("Enter f(x, y):", "x**2 + x*y")
    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.subheader("Symbolic Partial Derivatives")
    st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx))
    st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy))

    col1, col2 = st.columns(2)
    with col1: x0 = st.number_input("x₀", value=1.0)
    with col2: y0 = st.number_input("y₀", value=1.0)

    fx_val = float(fx.subs({x:x0, y:y0}))
    fy_val = float(fy.subs({x:x0, y:y0}))
    st.success(f"At ({x0}, {y0}): ∂f/∂x = {fx_val:.3f}, ∂f/∂y = {fy_val:.3f}")

    f_np = sp.lambdify((x, y), f, "numpy")
    fx_np = sp.lambdify((x, y), fx, "numpy")
    fy_np = sp.lambdify((x, y), fy, "numpy")

    t = np.linspace(-5, 5, 200)

    # w.r.t x
    x_vals = t
    y_fixed = np.full_like(x_vals, y0)
    fx_vals = fx_np(x_vals, y_fixed)
    f_vals_x = f_np(x_vals, y_fixed)

    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(x_vals, f_vals_x, label=f"f(x, {y0})", color="blue")
    ax1.plot(x_vals, fx_vals, label=f"∂f/∂x at y={y0}", color="red", linestyle="--")
    ax1.scatter(x0, f_np(x0, y0), color="green", s=50)
    ax1.text(x0, f_np(x0, y0), f"({x0:.2f},{f_np(x0,y0):.2f})", fontsize=9,
             ha="left", va="bottom", color="green")
    ax1.set_xlabel("x"); ax1.set_ylabel("f(x, y₀) and ∂f/∂x")
    ax1.set_title("Rate of Change w.r.t x")
    ax1.legend(); ax1.grid(True)
    st.pyplot(fig1)

    # w.r.t y
    y_vals = t
    x_fixed = np.full_like(y_vals, x0)
    fy_vals = fy_np(x_fixed, y_vals)
    f_vals_y = f_np(x_fixed, y_vals)

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(y_vals, f_vals_y, label=f"f({x0}, y)", color="blue")
    ax2.plot(y_vals, fy_vals, label=f"∂f/∂y at x={x0}", color="red", linestyle="--")
    ax2.scatter(y0, f_np(x0, y0), color="green", s=50)
    ax2.text(y0, f_np(x0, y0), f"({y0:.2f},{f_np(x0,y0):.2f})", fontsize=9,
             ha="left", va="bottom", color="green")
    ax2.set_xlabel("y"); ax2.set_ylabel("f(x₀, y) and ∂f/∂y")
    ax2.set_title("Rate of Change w.r.t y")
    ax2.legend(); ax2.grid(True)
    st.pyplot(fig2)

# =================================================
# 3. Differentials
# =================================================
elif topic == "Differentials":
    st.header("Differentials and Linear Approximation")

    # -----------------------------
    # Input function
    # -----------------------------
    expr_input = st.text_input("Enter f(x, y):", "x**2 + y**2")
    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()

    # -----------------------------
    # Partial derivatives
    # -----------------------------
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.subheader("Symbolic Partial Derivatives")
    st.latex(r"f_x = " + sp.latex(fx))
    st.latex(r"f_y = " + sp.latex(fy))

    # -----------------------------
    # Input point and increments
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0)
        y0 = st.number_input("y₀", value=1.0)
    with col2:
        dx = st.number_input("dx", value=0.1)
        dy = st.number_input("dy", value=0.1)

    # -----------------------------
    # Evaluate fx and fy at (x0, y0)
    # -----------------------------
    fx_val = float(fx.subs({x: x0, y: y0}))
    fy_val = float(fy.subs({x: x0, y: y0}))

    # -----------------------------
    # Differential df = fx*dx + fy*dy
    # -----------------------------
    st.subheader("Differential df")
    st.latex(r"df = f_x dx + f_y dy")

    # Show substitution explicitly
    df_expression = fx_val * dx + fy_val * dy
    st.markdown(
        f"Substitute f_x({x0},{y0}) = {fx_val}, f_y({x0},{y0}) = {fy_val}, "
        f"dx = {dx}, dy = {dy}:"
    )
    st.latex(r"df \approx ({:.3f})*({:.3f}) + ({:.3f})*({:.3f}) = {:.5f}".format(
        fx_val, dx, fy_val, dy, df_expression
    ))
    st.success(f"Numeric value: df ≈ {df_expression:.5f}")

    # -----------------------------
    # Actual change Δf
    # -----------------------------
    f_np = sp.lambdify((x, y), f, "numpy")
    actual_change = f_np(x0 + dx, y0 + dy) - f_np(x0, y0)
    st.info(
        f"Actual change Δf = f(x₀+dx, y₀+dy) - f(x₀, y₀) = "
        f"f({x0+dx},{y0+dy}) - f({x0},{y0}) = {actual_change:.5f}"
    )
    st.warning(f"Error of differential approximation = |Δf - df| = {abs(actual_change - df_expression):.5e}")

    # -----------------------------
    # Linear approximation (tangent plane)
    # -----------------------------
    L = f.subs({x: x0, y: y0}) + fx_val*(x - x0) + fy_val*(y - y0)

    st.subheader("Linear Approximation (Tangent Plane)")
    st.latex(r"L(x,y) = f(x_0, y_0) + f_x(x_0, y_0) (x-x_0) + f_y(x_0, y_0) (y-y_0)")

    # Show substitution step
    f_at_point = float(f.subs({x: x0, y: y0}))
    st.latex(
        r"L(x,y) = {:.5f} + ({:.3f})*(x-{:.3f}) + ({:.3f})*(y-{:.3f})".format(
            f_at_point, fx_val, x0, fy_val, y0
        )
    )

    # Evaluate linear approximation at (x0+dx, y0+dy)
    L_increment = (fx_val * dx) + (fy_val * dy)
    L_approx = f_at_point + L_increment
    true_value = f_np(x0 + dx, y0 + dy)
    linear_error = abs(true_value - L_approx)

    st.markdown(
        f"L(x₀, y₀) = f({x0},{y0}) = {f_at_point:.5f}\n"
        f"Increment = f_x*dx + f_y*dy = ({fx_val})*({dx}) + ({fy_val})*({dy}) = {L_increment:.5f}"
    )
    st.success(f"L(x₀ + dx, y₀ + dy) ≈ {L_approx:.5f}")
    st.info(f"True f(x₀ + dx, y₀ + dy) = {true_value:.5f}")
    st.warning(f"Linear approximation error = {linear_error:.5e}")

    st.info(
        "Summary:\n"
        "- Differential df gives the linear change: df = f_x dx + f_y dy\n"
        "- Linear approximation L(x₀+dx, y₀+dy) uses the tangent plane at (x₀, y₀)\n"
        "- Smaller dx, dy → better approximation"
    )




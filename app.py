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
        f = sp.sympify(
            expr_input,
            locals={
                "x": x,
                "y": y,
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "asin": sp.asin,
                "acos": sp.acos,
                "atan": sp.atan,
                "exp": sp.exp,
                "sqrt": sp.sqrt,
                "ln": sp.log,
                "log": sp.log,
                "e": sp.E,
            },
        )
        return f, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Domain analyzer
# -----------------------------
def analyze_domain(expr):
    conditions = []

    # Square roots
    for power in expr.atoms(sp.Pow):
        if power.exp == sp.Rational(1, 2):
            conditions.append(sp.latex(power.base) + r" \ge 0")
    # Logarithms
    for arg in expr.atoms(sp.log):
        conditions.append(sp.latex(arg.args[0]) + r" > 0")
    # Denominators
    denom = sp.denom(expr)
    if denom != 1:
        conditions.append(sp.latex(denom) + r" \ne 0")
    # Inverse trig
    for arg in expr.atoms(sp.asin, sp.acos):
        u = arg.args[0]
        conditions.append(r"-1 \le " + sp.latex(u) + r" \le 1")
    if conditions:
        return (
            r"\{(x,y)\in\mathbb{R}^2 \mid "
            + ",\ ".join(dict.fromkeys(conditions))
            + r"\}"
        )
    else:
        return r"\mathbb{R}^2"

# -----------------------------
# Sidebar topic selection
# -----------------------------
topic = st.sidebar.selectbox(
    "Select Topic",
    [
        "Function of Two Variables",
        "Partial Derivatives",
        "Differentials",
    ],
)

# =================================================
# 1. Function of Two Variables
# =================================================
if topic == "Function of Two Variables":
    st.header("Meaning of a Function of Two Variables")

    expr_input = st.text_input(
        "Enter f(x, y):",
        "x^2 + y^2",
        help=(
            "Use standard mathematical syntax.\n"
            "Examples:\n"
            "• sin(x*y)\n"
            "• sqrt(x^2 + y^2)\n"
            "• exp(x+y)\n"
            "Use asin(x) for sin⁻¹(x), and cos(x)^2 for cos²(x)."
        ),
    )

    st.caption(
        "⚠ Examples: `sin(x*y)`, `cos(x)^2`, `sqrt(3*x^4)`  \n"
        "⚠ Use `asin(x)` for sin⁻¹(x)"
    )

    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()

    st.latex(f"f(x,y) = {sp.latex(f)}")

    # -----------------------------
    # Evaluation point
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0)
    with col2:
        y0 = st.number_input("y₀", value=1.0)

    # -----------------------------
    # Axis range controls
    # -----------------------------
    st.subheader("Axis Range Settings")

    # Initialize session state
    if "x_min" not in st.session_state:
        st.session_state.x_min = -5.0
        st.session_state.x_max = 5.0
        st.session_state.y_min = -5.0
        st.session_state.y_max = 5.0

    col3, col4 = st.columns(2)

    # X-axis
    with col3:
        st.markdown("**x-axis range**")
        st.session_state.x_min = st.number_input(
            "x minimum", value=st.session_state.x_min, key="x_min_input"
        )
        st.session_state.x_max = st.number_input(
            "x maximum", value=st.session_state.x_max, key="x_max_input"
        )
        st.session_state.x_min, st.session_state.x_max = st.slider(
            "Adjust x-range",
            min_value=-20.0,
            max_value=20.0,
            value=(st.session_state.x_min, st.session_state.x_max),
            key="x_slider",
        )

    # Y-axis
    with col4:
        st.markdown("**y-axis range**")
        st.session_state.y_min = st.number_input(
            "y minimum", value=st.session_state.y_min, key="y_min_input"
        )
        st.session_state.y_max = st.number_input(
            "y maximum", value=st.session_state.y_max, key="y_max_input"
        )
        st.session_state.y_min, st.session_state.y_max = st.slider(
            "Adjust y-range",
            min_value=-20.0,
            max_value=20.0,
            value=(st.session_state.y_min, st.session_state.y_max),
            key="y_slider",
        )

    # Safety check
    if (
        st.session_state.x_min >= st.session_state.x_max
        or st.session_state.y_min >= st.session_state.y_max
    ):
        st.error("Minimum value must be less than maximum value.")
        st.stop()

    x_min = st.session_state.x_min
    x_max = st.session_state.x_max
    y_min = st.session_state.y_min
    y_max = st.session_state.y_max

    # -----------------------------
    # Domain
    # -----------------------------
    st.subheader("Domain")
    st.latex(analyze_domain(f))

    # -----------------------------
    # Plot
    # -----------------------------
    f_np = sp.lambdify((x, y), f, "numpy")
    x_vals = np.linspace(x_min, x_max, 120)
    y_vals = np.linspace(y_min, y_max, 120)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_np(X, Y)
    Z = np.where(np.isfinite(Z), Z, np.nan)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")

    z0 = f_np(x0, y0)

    ax.plot_surface(X, Y, Z, alpha=0.8, cmap="viridis")
    ax.scatter(x0, y0, z0, color="red", s=50)

    label_offset = (np.nanmax(Z) - np.nanmin(Z)) * 0.05
    ax.text(
        x0,
        y0,
        z0 + label_offset,
        f"({x0:.2f}, {y0:.2f}, {z0:.2f})",
        color="black",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    # Move Z-axis to the left
    ax.view_init(elev=30, azim=240)
    ax.zaxis.set_label_position('left')
    ax.zaxis.set_tick_params(labelleft=True, labelright=False)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    st.pyplot(fig)
    st.success(f"f({x0}, {y0}) = {z0:.3f}")

# =================================================
# 2. Partial Derivatives
# =================================================
elif topic == "Partial Derivatives":
    st.header("Partial Derivatives as Rates of Change")

    expr_input = st.text_input(
        "Enter f(x, y):",
        "x^2 + x*y",
        help="Examples: x^2 + x*y, sin(x*y), exp(x-y)"
    )
    st.caption("Valid examples: `x^2 + x*y`, `sin(x*y)`, `sqrt(x^2+y^2)`")

    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx))
    st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy))

    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0)
    with col2:
        y0 = st.number_input("y₀", value=1.0)

    st.success(
        f"At ({x0}, {y0}): "
        f"∂f/∂x = {float(fx.subs({x:x0,y:y0})):.3f}, "
        f"∂f/∂y = {float(fy.subs({x:x0,y:y0})):.3f}"
    )

    t = np.linspace(-3, 3, 100)
    f_np = sp.lambdify((x, y), f, "numpy")

    fig_x, ax_x = plt.subplots()
    ax_x.plot(t, f_np(t, y0))
    ax_x.axvline(x0, linestyle="--")
    ax_x.set_title("Rate of Change w.r.t x (y fixed)")
    ax_x.set_xlabel("x")
    ax_x.set_ylabel("f(x, y₀)")
    st.pyplot(fig_x)

    fig_y, ax_y = plt.subplots()
    ax_y.plot(t, f_np(x0, t))
    ax_y.axvline(y0, linestyle="--")
    ax_y.set_title("Rate of Change w.r.t y (x fixed)")
    ax_y.set_xlabel("y")
    ax_y.set_ylabel("f(x₀, y)")
    st.pyplot(fig_y)

# =================================================
# 3. Differentials
# =================================================
elif topic == "Differentials":
    st.header("Differentials and Linear Approximation")

    expr_input = st.text_input(
        "Enter f(x, y):",
        "x^2 + y^2",
        help="Examples: x^2 + y^2, sqrt(x^2+y^2), exp(x+y)"
    )
    st.caption("Valid examples: `x^2 + y^2`, `sqrt(x^2 + y^2)`, `exp(x+y)`")

    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.subheader("Partial Derivatives")
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
    df_symbolic = fx * dx_sym + fy * dy_sym

    st.subheader("Differential Formula")
    st.latex(r"df = f_x\,dx + f_y\,dy")
    st.latex(r"df = " + sp.latex(df_symbolic))

    df_substituted = df_symbolic.subs({dx_sym: dx, dy_sym: dy})
    st.subheader("Substitute dx and dy")
    st.latex(r"df = " + sp.latex(df_substituted))

    df_numeric = df_substituted.subs({x: x0, y: y0})
    f_np = sp.lambdify((x, y), f, "numpy")
    actual_change = f_np(x0 + dx, y0 + dy) - f_np(x0, y0)

    st.success(f"df ≈ {float(df_numeric):.5f}")
    st.info(f"Actual change Δf = {actual_change:.5f}")
    st.warning(f"Approximation error = {abs(actual_change - float(df_numeric)):.5e}")

    st.info(
        "The differential provides a linear approximation to the actual change in the function. "
        "The approximation improves as dx and dy become smaller."
    )

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

# -----------------------------
# Axis min/max limits
# -----------------------------
X_LIMITS = (-20.0, 20.0)
Y_LIMITS = (-20.0, 20.0)

# -----------------------------
# Initialize axis ranges in session state
# -----------------------------
if "x_min" not in st.session_state:
    st.session_state.x_min = -5.0
    st.session_state.x_max = 5.0
    st.session_state.y_min = -5.0
    st.session_state.y_max = 5.0

# =================================================
# 1. Function of Two Variables
# =================================================
if topic == "Function of Two Variables":
    st.header("Function of Two Variables")

    expr_input = st.text_input(
        "Enter f(x, y):",
        "x^2 + y^2",
        help=(
            "Examples:\n"
            "• sin(x*y)\n"
            "• sqrt(x^2 + y^2)\n"
            "• exp(x+y)"
        ),
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
        x0 = st.number_input("x₀", value=1.0, min_value=X_LIMITS[0], max_value=X_LIMITS[1])
    with col2:
        y0 = st.number_input("y₀", value=1.0, min_value=Y_LIMITS[0], max_value=Y_LIMITS[1])

    # -----------------------------
    # Axis range settings
    # -----------------------------
    st.subheader("Axis Range Settings")
    col3, col4 = st.columns(2)

    # X-axis
    with col3:
        st.markdown("**x-axis range**")
        st.session_state.x_min = st.number_input(
            "x minimum", value=st.session_state.x_min, min_value=X_LIMITS[0], max_value=X_LIMITS[1]
        )
        st.session_state.x_max = st.number_input(
            "x maximum", value=st.session_state.x_max, min_value=X_LIMITS[0], max_value=X_LIMITS[1]
        )
        # Slider restricted to [x_min, x_max]
        st.session_state.x_min, st.session_state.x_max = st.slider(
            "Adjust x-range",
            min_value=st.session_state.x_min,
            max_value=st.session_state.x_max,
            value=(st.session_state.x_min, st.session_state.x_max),
        )

    # Y-axis
    with col4:
        st.markdown("**y-axis range**")
        st.session_state.y_min = st.number_input(
            "y minimum", value=st.session_state.y_min, min_value=Y_LIMITS[0], max_value=Y_LIMITS[1]
        )
        st.session_state.y_max = st.number_input(
            "y maximum", value=st.session_state.y_max, min_value=Y_LIMITS[0], max_value=Y_LIMITS[1]
        )
        st.session_state.y_min, st.session_state.y_max = st.slider(
            "Adjust y-range",
            min_value=st.session_state.y_min,
            max_value=st.session_state.y_max,
            value=(st.session_state.y_min, st.session_state.y_max),
        )

    if st.session_state.x_min >= st.session_state.x_max or st.session_state.y_min >= st.session_state.y_max:
        st.error("Minimum must be less than maximum.")
        st.stop()

    x_min, x_max = st.session_state.x_min, st.session_state.x_max
    y_min, y_max = st.session_state.y_min, st.session_state.y_max

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

    # Ensure label stays visible
    label_offset = (np.nanmax(Z) - np.nanmin(Z)) * 0.05
    x_label = np.clip(x0 + label_offset, x_min, x_max)
    y_label = np.clip(y0 + label_offset, y_min, y_max)
    z_label = np.clip(z0 + label_offset, np.nanmin(Z), np.nanmax(Z))

    ax.text(
        x_label,
        y_label,
        z_label,
        f"({x0:.2f}, {y0:.2f}, {z0:.2f})",
        color="black",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    ax.view_init(elev=30, azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    st.pyplot(fig)
    st.success(f"f({x0}, {y0}) = {z0:.3f}")

        "The differential provides a linear approximation to the actual change in the function. "
        "The approximation improves as dx and dy become smaller."
    )

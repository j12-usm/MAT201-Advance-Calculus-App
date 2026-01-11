import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Multivariable Calculus App", layout="wide")
st.title("📐 Multivariable Calculus Learning App")

x, y = sp.symbols("x y", real=True)

# -----------------------------
# Safe function parser
# -----------------------------

def log10(x):
    return sp.log(x, 10)

def parse_function(expr_input):
    try:
        expr_input = expr_input.replace("^", "**")
        f = sp.sympify(expr_input, locals={
            "x": x, "y": y,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "exp": sp.exp, "sqrt": sp.sqrt,
            "ln": sp.log,          # natural log
            "log": log10,          # base-10 log
            "e": sp.E,
        })
        return f, None
    except Exception as e:
        return None, str(e)

def latex_with_mixed_ln_log(expr, original_input):

    def _log_to_latex(e):
        if isinstance(e, sp.log):
            arg = e.args[0]

            # log(x, 10) → base-10
            if len(e.args) == 2 and e.args[1] == 10:
                return r"\log_{10}\!\left(" + sp.latex(arg) + r"\right)"

            # log(x) → natural log
            return r"\ln\!\left(" + sp.latex(arg) + r"\right)"

        return sp.latex(e)

    return sp.latex(expr, fold_short_frac=True, symbol_names={}, 
                    mul_symbol="dot", 
                    printer=_log_to_latex)

# -----------------------------
# LaTeX display helpers
# -----------------------------
from sympy.printing.latex import LatexPrinter

def latex_with_mixed_ln_log(expr, original_input):
    expr = sp.simplify(expr)

    user_used_ln = "ln(" in original_input
    user_used_log = "log(" in original_input

    class CustomLatexPrinter(LatexPrinter):
        def _print_log(self, expr):
            arg = expr.args[0]

            # Explicit base-10 log
            if len(expr.args) == 2 and expr.args[1] == 10:
                return r"\log_{10}\!\left(%s\right)" % self._print(arg)

            # User typed ln(x)
            if user_used_ln:
                return r"\ln\!\left(%s\right)" % self._print(arg)

            # User typed log(x)
            if user_used_log:
                return r"\log\!\left(%s\right)" % self._print(arg)

            # fallback
            return r"\ln\!\left(%s\right)" % self._print(arg)

    return CustomLatexPrinter().doprint(expr)

# -----------------------------
# Custom display (show ln instead of log)
# -----------------------------
def latex_with_ln(expr):
    latex_str = sp.latex(expr)
    latex_str = latex_str.replace(r"\log", r"\ln")
    return latex_str

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
    st.header("📊 Function of Two Variables")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    st.caption("Use standard mathematical syntax.")
    st.caption("**Examples:**")
    col1, col2, col3 = st.columns([1.2, 1.8, 1.2])
    with col1:
        st.markdown("<span style='font-size:15px'>• <code>sin(x*y)</code></span>", unsafe_allow_html=True)
    with col2:
        st.markdown("<span style='font-size:15px'>• <code>sqrt(x^2 + y^2)</code></span>", unsafe_allow_html=True)
    with col3:
        st.markdown("<span style='font-size:15px'>• <code>exp(x + y)</code></span>", unsafe_allow_html=True)
    st.caption("Use asin(x) for sin⁻¹(x), and cos(x)^2 for cos²(x).")


    f, error = parse_function(expr_input)

    if error:
        st.error("❌ Invalid function syntax.")
        st.stop()

    st.latex(rf"f(x,y) = {latex_with_mixed_ln_log(f, expr_input)}")

    # Evaluation point
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0, min_value=X_LIMITS[0], max_value=X_LIMITS[1])
    with col2:
        y0 = st.number_input("y₀", value=1.0, min_value=Y_LIMITS[0], max_value=Y_LIMITS[1])

    # Axis ranges
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.x_min = st.number_input(
            "x min", value=st.session_state.x_min,
            min_value=X_LIMITS[0], max_value=X_LIMITS[1]
        )
        st.session_state.x_max = st.number_input(
            "x max", value=st.session_state.x_max,
            min_value=X_LIMITS[0], max_value=X_LIMITS[1]
        )
        if st.session_state.x_min >= st.session_state.x_max:
            st.session_state.x_min = st.session_state.x_max - 0.1

        st.session_state.x_min, st.session_state.x_max = st.slider(
            "Adjust x-range",
            min_value=st.session_state.x_min,
            max_value=st.session_state.x_max,
            value=(st.session_state.x_min, st.session_state.x_max)
        )

    with col4:
        st.session_state.y_min = st.number_input(
            "y min", value=st.session_state.y_min,
            min_value=Y_LIMITS[0], max_value=Y_LIMITS[1]
        )
        st.session_state.y_max = st.number_input(
            "y max", value=st.session_state.y_max,
            min_value=Y_LIMITS[0], max_value=Y_LIMITS[1]
        )
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

    # Domain
    st.subheader("📐 Domain")
    st.latex(analyze_domain(f))

    # -----------------------------
    # Interactive 3D Plot (Plotly)
    # -----------------------------
    f_np = sp.lambdify((x, y), f, "numpy")
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120)
    )
    Z = f_np(X, Y)
    Z = np.where(np.isfinite(Z), Z, np.nan)

    z0 = f_np(x0, y0)

    fig = go.Figure()

    # Surface
    fig.add_trace(go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale="Viridis",
    opacity=0.85,
    colorbar=dict(
        title="f(x,y)",
        x=1.15,        # move colorbar right
        len=0.75,      # slightly shorter
        thickness=18
    )
))


    # High-contrast point
    fig.add_trace(go.Scatter3d(
        x=[x0],
        y=[y0],
        z=[z0],
        mode="markers+text",
        marker=dict(
            size=10,
            color="black",
            line=dict(color="white", width=3)
        ),
        text=[f"<b>({x0:.2f}, {y0:.2f}, {z0:.2f})</b>"],
        textposition="top center",
        textfont=dict(size=14, color="black"),
        name="Point"
    ))

    # Vertical guide line
    fig.add_trace(go.Scatter3d(
        x=[x0, x0],
        y=[y0, y0],
        z=[np.nanmin(Z), z0],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        showlegend=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x,y)",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
        ),
        height=650,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"f({x0}, {y0}) = {z0:.4f}")

# =================================================
# 2. Partial Derivatives
# =================================================
elif topic == "Partial Derivatives":
    st.header("Partial Derivatives as Rate of Change")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    st.caption("Use standard mathematical syntax.")
    st.caption("**Examples:**")
    col1, col2, col3 = st.columns([1.2, 1.8, 1.2])
    with col1:
        st.markdown("<span style='font-size:15px'>• <code>sin(x*y)</code></span>", unsafe_allow_html=True)
    with col2:
        st.markdown("<span style='font-size:15px'>• <code>sqrt(x^2 + y^2)</code></span>", unsafe_allow_html=True)
    with col3:
        st.markdown("<span style='font-size:15px'>• <code>exp(x + y)</code></span>", unsafe_allow_html=True)
    st.caption("Use asin(x) for sin⁻¹(x), and cos(x)^2 for cos²(x).")

    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()

    vars_used = f.free_symbols
    uses_x = x in vars_used
    uses_y = y in vars_used

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.subheader("Symbolic Partial Derivatives")
    st.latex(r"\frac{\partial f}{\partial x} = " + latex_with_mixed_ln_log(fx, expr_input))
    st.latex(r"\frac{\partial f}{\partial y} = " + latex_with_mixed_ln_log(fy, expr_input))


    col1, col2 = st.columns(2)
    with col1: x0 = st.number_input("x₀", value=1.0)
    with col2: y0 = st.number_input("y₀", value=1.0)

    fx_val = float(fx.subs({x:x0, y:y0}))
    fy_val = float(fy.subs({x:x0, y:y0}))
    st.success(f"At ({x0}, {y0}): ∂f/∂x = {fx_val:.3f}, ∂f/∂y = {fy_val:.3f}")

    if uses_x and uses_y:
        f_np = sp.lambdify((x, y), f, "numpy")
    elif uses_x:
        f_np = sp.lambdify(x, f, "numpy")
    elif uses_y:
        f_np = sp.lambdify(y, f, "numpy")

    if uses_x:
        fx_np = sp.lambdify((x, y), fx, "numpy") if uses_y else sp.lambdify(x, fx, "numpy")
    if uses_y:
        fy_np = sp.lambdify((x, y), fy, "numpy") if uses_x else sp.lambdify(y, fy, "numpy")

    t = np.linspace(-5, 5, 200)

# w.r.t x
    if uses_x:
        x_vals = t
        y_fixed = np.full_like(x_vals, y0) if uses_y else None
        f_vals_x = f_np(x_vals, y_fixed) if uses_y else f_np(x_vals)
        fx_vals = fx_np(x_vals, y_fixed) if uses_y else fx_np(x_vals)

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
    else
        st.info("ℹ️ No x-direction graph (function does not depend on x).")
    
    # w.r.t y
    if uses_y: 
        y_vals = t
        x_fixed = np.full_like(y_vals, x0) if uses_x else None 
        fy_vals = fy_np(x_fixed, y_vals) if uses_x else f_np(y_vals)
        f_vals_y = f_np(x_fixed, y_vals) if uses_x else fy_np(y_vals)

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
    else
        st.info("ℹ️ No y-direction graph (function does not depend on y).")


# =================================================
# 3. Differentials
# =================================================
elif topic == "Differentials":
    st.header("Differentials and Linear Approximation")

    # -----------------------------
    # Input function
    # -----------------------------
    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    st.caption("Use standard mathematical syntax.")
    st.caption("**Examples:**")
    col1, col2, col3 = st.columns([1.2, 1.8, 1.2])
    with col1:
        st.markdown("<span style='font-size:15px'>• <code>sin(x*y)</code></span>", unsafe_allow_html=True)
    with col2:
        st.markdown("<span style='font-size:15px'>• <code>sqrt(x^2 + y^2)</code></span>", unsafe_allow_html=True)
    with col3:
        st.markdown("<span style='font-size:15px'>• <code>exp(x + y)</code></span>", unsafe_allow_html=True)
    st.caption("Use asin(x) for sin⁻¹(x), and cos(x)^2 for cos²(x).")

    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()
        
    vars_used = f.free_symbols
    uses_x = x in vars_used
    uses_y = y in vars_used

    # -----------------------------
    # Partial derivatives
    # -----------------------------
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.subheader("Symbolic Partial Derivatives")
    st.latex(r"fx = " + latex_with_mixed_ln_log(fx, expr_input))
    st.latex(r"fy = " + latex_with_mixed_ln_log(fy, expr_input))

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
    # Evaluate fx and fy at point
    # -----------------------------
    fx_sub = fx.subs({x: x0, y: y0})
    fy_sub = fy.subs({x: x0, y: y0})
    fx_val = float(fx_sub)
    fy_val = float(fy_sub)

    st.subheader("Differential df")
    st.latex(r"df = fx*dx + fy*dy")

    # Step-by-step substitution for fx*dx
    st.markdown(f"Step 1: Evaluate fx and fy at (x₀,y₀)")
    st.markdown(f"fx = {sp.latex(fx)} → fx({x0},{y0}) = {sp.latex(fx_sub)} → numeric: {fx_val}")
    st.markdown(f"fy = {sp.latex(fy)} → fy({x0},{y0}) = {sp.latex(fy_sub)} → numeric: {fy_val}")

    # Step 2: Multiply by dx and dy with substitution
    df_x = fx_val * dx
    df_y = fy_val * dy
    df_numeric = df_x + df_y

    st.markdown(
        f"Step 2: Multiply by increments dx, dy:\n"
        f"df = fx*dx + fy*dy = ({fx_val})*({dx}) + ({fy_val})*({dy}) = {df_numeric:.5f}"
    )
    st.success(f"Numeric value: df ≈ {df_numeric:.5f}")

    # -----------------------------
    # Actual change Δf
    # -----------------------------
    f_np = sp.lambdify((x, y), f, "numpy")
    actual_change = f_np(x0 + dx, y0 + dy) - f_np(x0, y0)
    st.info(
        f"Actual change Δf = f(x₀+dx, y₀+dy) - f(x₀, y₀) = "
        f"f({x0+dx},{y0+dy}) - f({x0},{y0}) = {actual_change:.5f}"
    )

    # Format error in scientific notation as *10^a
    error_value = abs(actual_change - df_numeric)
    if error_value != 0:
        error_sci = f"{error_value/10**int(np.floor(np.log10(error_value))):.3f}*10^{int(np.floor(np.log10(error_value)))}"
    else:
        error_sci = "0"

    st.warning(f"Error of differential approximation = |Δf - df| ≈ {error_sci}")

    # -----------------------------
    # Linear approximation (tangent plane)
    # -----------------------------
    L = f.subs({x: x0, y: y0}) + fx_val*(x - x0) + fy_val*(y - y0)

    st.subheader("Linear Approximation (Tangent Plane)")
    st.latex(r"L(x,y) = f(x₀, y₀) + fx(x₀,y₀) (x-x₀) + fy(x₀,y₀) (y-y₀)")

    # Show substitution in L
    f_at_point = float(f.subs({x: x0, y: y0}))
    L_increment = df_numeric
    L_approx = f_at_point + L_increment
    true_value = f_np(x0 + dx, y0 + dy)
    linear_error = abs(true_value - L_approx)

    # Linear error in *10^a format
    if linear_error != 0:
        linear_error_sci = f"{linear_error/10**int(np.floor(np.log10(linear_error))):.3f}*10^{int(np.floor(np.log10(linear_error)))}"
    else:
        linear_error_sci = "0"

    st.markdown(f"f({x0},{y0}) = {f_at_point:.5f}")
    st.markdown(
        f"Increment = fx*dx + fy*dy = ({fx_val})*({dx}) + ({fy_val})*({dy}) = {L_increment:.5f}"
    )
    st.success(f"L(x₀ + dx, y₀ + dy) ≈ {L_approx:.5f}")
    st.info(f"True f(x₀ + dx, y₀ + dy) = {true_value:.5f}")
    st.warning(f"Linear approximation error ≈ {linear_error_sci}")

    st.info(
        "Summary:\n"
        "- Differential df shows the linear change: df = fx*dx + fy*dy\n"
        "- Linear approximation L(x₀+dx, y₀+dy) uses the tangent plane at (x₀, y₀)\n"
        "- Smaller dx, dy → better approximation"
    )






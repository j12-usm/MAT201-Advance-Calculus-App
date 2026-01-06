import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="Multivariable Calculus App", layout="wide")
st.title("📐 Multivariable Calculus Learning App")

x, y = sp.symbols("x y", real=True)

# -------------------------------------------------
# Improved safe parser
# -------------------------------------------------
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

# -------------------------------------------------
# Domain analyzer (basic but mathematical)
# -------------------------------------------------
def analyze_domain(expr):
    conditions = []

    for arg in expr.atoms(sp.sqrt):
        conditions.append(sp.latex(arg.args[0]) + r" \ge 0")

    for arg in expr.atoms(sp.log):
        conditions.append(sp.latex(arg.args[0]) + r" > 0")

    denom = sp.denom(expr)
    if denom != 1:
        conditions.append(sp.latex(denom) + r" \ne 0")

    if conditions:
        return r"\{(x,y)\in\mathbb{R}^2 \mid " + ",\ ".join(conditions) + r"\}"
    else:
        return r"\mathbb{R}^2"


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
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
        "⚠ Examples of valid input: `sin(x*y)`, `cos(x)^2`, `sqrt(3*x^4)`  \n"
        "⚠ Use `asin(x)` for sin⁻¹(x)"
    )

    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function syntax.")
        st.stop()

    st.latex(f"f(x,y) = {sp.latex(f)}")

    # User-defined evaluation point
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0)
    with col2:
        y0 = st.number_input("y₀", value=1.0)

    # Domain
    st.subheader("Domain")
    st.latex(analyze_domain(f))

    # Plot
    f_np = sp.lambdify((x, y), f, "numpy")

    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_np(X, Y)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.8)
    ax.scatter(x0, y0, f_np(x0, y0), color="red", s=50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    st.pyplot(fig)

    st.success(f"f({x0}, {y0}) = {f_np(x0, y0):.3f}")

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

    st.caption(
        "Valid examples: `x^2 + x*y`, `sin(x*y)`, `sqrt(x^2+y^2)`"
    )

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

    # Separate rate-of-change plots
    t = np.linspace(-3, 3, 100)
    f_np = sp.lambdify((x, y), f, "numpy")

    fig_x, ax_x = plt.subplots()
    ax_x.plot(t, f_np(t, y0))
    ax_x.axvline(x0, linestyle="--")
    ax_x.set_title("Rate of Change with Respect to x (y fixed)")
    ax_x.set_xlabel("x")
    ax_x.set_ylabel("f(x, y₀)")
    st.pyplot(fig_x)

    fig_y, ax_y = plt.subplots()
    ax_y.plot(t, f_np(x0, t))
    ax_y.axvline(y0, linestyle="--")
    ax_y.set_title("Rate of Change with Respect to y (x fixed)")
    ax_y.set_xlabel("y")
    ax_y.set_ylabel("f(x₀, y)")
    st.pyplot(fig_y)

# =================================================
# 3. Differentials (Improved Step-by-Step)
# =================================================
elif topic == "Differentials":
    st.header("Differentials and Linear Approximation")

    expr_input = st.text_input(
        "Enter f(x, y):",
        "x^2 + y^2",
        help="Examples: x^2 + y^2, sqrt(x^2+y^2), exp(x+y)"
    )

    st.caption(
        "Valid examples: `x^2 + y^2`, `sqrt(x^2 + y^2)`, `exp(x+y)`"
    )

    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function.")
        st.stop()

    # Partial derivatives
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.subheader("Partial Derivatives")
    st.latex(r"f_x = " + sp.latex(fx))
    st.latex(r"f_y = " + sp.latex(fy))

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0)
        y0 = st.number_input("y₀", value=1.0)
    with col2:
        dx = st.number_input("dx", value=0.1)
        dy = st.number_input("dy", value=0.1)

    # Symbolic differential
    dx_sym, dy_sym = sp.symbols("dx dy")
    df_symbolic = fx * dx_sym + fy * dy_sym

    st.subheader("Differential Formula")
    st.latex(r"df = f_x\,dx + f_y\,dy")
    st.latex(r"df = " + sp.latex(df_symbolic))

    # Substitute dx, dy
    df_substituted = df_symbolic.subs({dx_sym: dx, dy_sym: dy})

    st.subheader("Substitute dx and dy")
    st.latex(r"df = " + sp.latex(df_substituted))

    # Evaluate at (x0, y0)
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

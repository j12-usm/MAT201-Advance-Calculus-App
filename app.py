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

    for denom in sp.denom(expr).as_ordered_factors():
        conditions.append(sp.latex(denom) + r" \ne 0")

    if conditions:
        return r"$\{(x,y)\in\mathbb{R}^2 : " + ",\ ".join(conditions) + r"\}$"
    else:
        return r"$\mathbb{R}^2$"

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
    ax_x.set_title("Change in f as x varies (y fixed)")
    ax_x.set_xlabel("x")
    ax_x.set_ylabel("f(x, y₀)")
    st.pyplot(fig_x)

    fig_y, ax_y = plt.subplots()
    ax_y.plot(t, f_np(x0, t))
    ax_y.axvline(y0, linestyle="--")
    ax_y.set_title("Change in f as y varies (x fixed)")
    ax_y.set_xlabel("y")
    ax_y.set_ylabel("f(x₀, y)")
    st.pyplot(fig_y)

# =================================================
# 3. Differentials (unchanged)
# =================================================
elif topic == "Differentials":
    st.header("Differentials and Linear Approximation")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0)
        y0 = st.number_input("y₀", value=1.0)
    with col2:
        dx = st.number_input("dx", value=0.1)
        dy = st.number_input("dy", value=0.1)

    f_np = sp.lambdify((x, y), f, "numpy")

    actual_change = f_np(x0 + dx, y0 + dy) - f_np(x0, y0)
    df = fx.subs({x:x0,y:y0})*dx + fy.subs({x:x0,y:y0})*dy

    st.latex("df = f_x dx + f_y dy")
    st.success(f"df ≈ {float(df):.5f}")
    st.info(f"Actual Δf = {actual_change:.5f}")

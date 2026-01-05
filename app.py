import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="Multivariable Calculus App", layout="wide")
st.title("📐 Multivariable Calculus Learning App")

x, y = sp.symbols("x y")

# -------------------------------------------------
# Safe parser
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
                "exp": sp.exp,
                "sqrt": sp.sqrt,
                "ln": sp.log,
                "log": sp.log,
                "e": sp.E,
            },
        )
        return f, None
    except Exception:
        return None, "Invalid function"

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

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function.")
        st.stop()

    st.latex(f"f(x,y) = {sp.latex(f)}")

    col1, col2 = st.columns(2)

    with col1:
        x0 = st.slider("x₀", -4.0, 4.0, 1.0)
        y0 = st.slider("y₀", -4.0, 4.0, 1.0)

    f_np = sp.lambdify((x, y), f, "numpy")

    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_np(X, Y)

    # 3D surface
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.8)
    ax.scatter(x0, y0, f_np(x0, y0), color="red", s=50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    st.pyplot(fig)

    # Contour plot
    fig2, ax2 = plt.subplots()
    contour = ax2.contour(X, Y, Z, levels=15)
    ax2.clabel(contour)
    ax2.scatter(x0, y0, color="red")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    st.pyplot(fig2)

    st.success(f"f({x0:.2f}, {y0:.2f}) = {f_np(x0, y0):.3f}")

    st.info(
        "A function of two variables assigns a value to each point (x, y). "
        "The surface shows height, while contour lines show points with equal values."
    )

# =================================================
# 2. Partial Derivatives
# =================================================
elif topic == "Partial Derivatives":
    st.header("Partial Derivatives as Rates of Change")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + x*y")
    f, error = parse_function(expr_input)
    if error:
        st.error("Invalid function.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(fx)}")
    st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(fy)}")

    x0 = st.slider("x₀", -3.0, 3.0, 1.0)
    y0 = st.slider("y₀", -3.0, 3.0, 1.0)

    fx_val = float(fx.subs({x: x0, y: y0}))
    fy_val = float(fy.subs({x: x0, y: y0}))

    st.success(
        f"At ({x0}, {y0}): ∂f/∂x = {fx_val:.3f},  ∂f/∂y = {fy_val:.3f}"
    )

    # Cross-sections
    t = np.linspace(-3, 3, 100)
    f_np = sp.lambdify((x, y), f, "numpy")

    fig, ax = plt.subplots()
    ax.plot(t, f_np(t, y0), label="x varies, y fixed")
    ax.plot(t, f_np(x0, t), label="y varies, x fixed")
    ax.legend()
    ax.set_xlabel("Variable")
    ax.set_ylabel("f value")
    st.pyplot(fig)

    st.info(
        "∂f/∂x measures change when x varies and y is fixed. "
        "∂f/∂y measures change when y varies and x is fixed."
    )

# =================================================
# 3. Differentials
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
    df = fx.subs({x: x0, y: y0}) * dx + fy.subs({x: x0, y: y0}) * dy

    st.latex("df = f_x dx + f_y dy")
    st.success(f"Differential (df) ≈ {float(df):.5f}")
    st.info(f"Actual change Δf = {actual_change:.5f}")
    st.warning(f"Approximation error = {abs(actual_change - df):.5e}")

    st.info(
        "Differentials give a linear approximation of the actual change in f. "
        "The smaller dx and dy are, the better the approximation."
    )

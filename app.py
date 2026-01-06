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

    # -------------------------------------------------
    # Domain & Range
    # -------------------------------------------------
    st.subheader("Domain and Range")

    st.markdown("**Domain:** All real values of x and y for which the function is defined.")

    Z_finite = Z[np.isfinite(Z)]
    z_min = np.min(Z_finite)
    z_max = np.max(Z_finite)

    st.markdown(
        f"**Approximate Range (from plotted region):** "
        f"[{z_min:.2f}, {z_max:.2f}]"
    )

    st.info(
        "The domain depends on where the formula makes sense. "
        "The range shown here is a numerical estimate based on the plotted surface."
    )

    # -------------------------------------------------
    # 3D surface
    # -------------------------------------------------
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.8)
    ax.scatter(x0, y0, f_np(x0, y0), color="red", s=50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    st.pyplot(fig)

    # -------------------------------------------------
    # Contour plot
    # -------------------------------------------------
    fig2, ax2 = plt.subplots()
    contour = ax2.contour(X, Y, Z, levels=15)
    ax2.clabel(contour)
    ax2.scatter(x0, y0, color="red")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    st.pyplot(fig2)

    st.success(f"f({x0:.2f}, {y0:.2f}) = {f_np(x0, y0):.3f}")

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

    # -------------------------------------------------
    # Separate cross-sections
    # -------------------------------------------------
    t = np.linspace(-3, 3, 100)
    f_np = sp.lambdify((x, y), f, "numpy")

    # Rate of change w.r.t x
    fig_x, ax_x = plt.subplots()
    ax_x.plot(t, f_np(t, y0))
    ax_x.axvline(x0, linestyle="--")
    ax_x.set_title("Rate of Change with Respect to x (y fixed)")
    ax_x.set_xlabel("x")
    ax_x.set_ylabel("f(x, y₀)")
    st.pyplot(fig_x)

    # Rate of change w.r.t y
    fig_y, ax_y = plt.subplots()
    ax_y.plot(t, f_np(x0, t))
    ax_y.axvline(y0, linestyle="--")
    ax_y.set_title("Rate of Change with Respect to y (x fixed)")
    ax_y.set_xlabel("y")
    ax_y.set_ylabel("f(x₀, y)")
    st.pyplot(fig_y)

    st.info(
        "Each graph represents a cross-section of the surface. "
        "The slope at the marked point corresponds to the partial derivative."
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
        "Differentials provide a linear approximation of the actual change in f. "
        "The approximation improves as dx and dy become smaller."
    )

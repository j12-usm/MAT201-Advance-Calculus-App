import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="Multivariable Calculus App", layout="wide")
st.title("📐 Multivariable Calculus Learning App")
st.write("Interactive visualization and explanation of functions of several variables.")

# Symbols
x, y = sp.symbols('x y')

# -------------------------------------------------
# Safe function parser
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
                "e": sp.E,
                "ln": sp.log,
                "log": sp.log,
                "sqrt": sp.sqrt
            }
        )
        return f, None
    except Exception:
        return None, "Invalid function"

# Sidebar
topic = st.sidebar.selectbox(
    "Select Topic",
    [
        "Function of Two Variables",
        "Partial Derivatives",
        "Directional Derivatives",
        "Gradient & Steepest Ascent",
        "Differentials"
    ]
)

# -------------------------------------------------
# 1. Function of Two Variables
# -------------------------------------------------
if topic == "Function of Two Variables":
    st.header("Meaning & Visualization of f(x, y)")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    st.caption("Examples: x^2 + y^2, sin(x)+y, exp(x+y), sqrt(x^2+y^2)")

    f, error = parse_function(expr_input)
    if error:
        st.error("❌ Invalid function. Please check your input.")
        st.stop()

    st.latex(f"f(x,y) = {sp.latex(f)}")

    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    f_np = sp.lambdify((x, y), f, "numpy")

    try:
        Z = f_np(X, Y)
    except Exception:
        st.error("❌ Function cannot be evaluated on this range.")
        st.stop()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    st.pyplot(fig)

    st.info(
        "A function of two variables assigns a value to each point (x, y). "
        "The surface shows how f(x, y) changes as x and y vary."
    )

# -------------------------------------------------
# 2. Partial Derivatives
# -------------------------------------------------
elif topic == "Partial Derivatives":
    st.header("Partial Derivatives as Rates of Change")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + x*y")
    st.caption("Examples: x^2 + xy, sin(xy), exp(x+y)")

    f, error = parse_function(expr_input)
    if error:
        st.error("❌ Invalid function.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    x0 = st.number_input("x₀", value=1.0)
    y0 = st.number_input("y₀", value=1.0)

    st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(fx)}")
    st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(fy)}")

    try:
        fx_val = fx.subs({x: x0, y: y0})
        fy_val = fy.subs({x: x0, y: y0})
        st.success(f"At ({x0}, {y0}): ∂f/∂x = {fx_val},  ∂f/∂y = {fy_val}")
    except Exception:
        st.error("❌ Cannot evaluate derivatives at this point.")

    st.info(
        "Partial derivatives measure how the function changes when one variable "
        "changes while the other is held constant."
    )

# -------------------------------------------------
# 3. Directional Derivatives
# -------------------------------------------------
elif topic == "Directional Derivatives":
    st.header("Directional Derivatives")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    st.caption("Examples: x^2+y^2, sin(xy), exp(x-y)")

    f, error = parse_function(expr_input)
    if error:
        st.error("❌ Invalid function.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    x0 = st.number_input("x₀", value=1.0)
    y0 = st.number_input("y₀", value=1.0)

    vx = st.number_input("Direction vector v₁", value=1.0)
    vy = st.number_input("Direction vector v₂", value=1.0)

    if vx == 0 and vy == 0:
        st.error("❌ Direction vector cannot be zero.")
        st.stop()

    v = np.array([vx, vy], dtype=float)
    v_unit = v / np.linalg.norm(v)

    try:
        grad = np.array(
            [fx.subs({x: x0, y: y0}), fy.subs({x: x0, y: y0})],
            dtype=float
        )
        Dv = grad.dot(v_unit)
        st.latex("D_v f = \\nabla f \\cdot \\hat{v}")
        st.success(f"Directional derivative = {Dv:.3f}")
    except Exception:
        st.error("❌ Cannot compute directional derivative.")

    st.info(
        "The directional derivative gives the rate of change of f "
        "in a specified direction."
    )

# -------------------------------------------------
# 4. Gradient & Steepest Ascent
# -------------------------------------------------
elif topic == "Gradient & Steepest Ascent":
    st.header("Gradient and Direction of Steepest Ascent")

    expr_input = st.text_input("Enter f(x, y):", "x*y")
    st.caption("Examples: xy, x^2+y^2, sin(x)+cos(y)")

    f, error = parse_function(expr_input)
    if error:
        st.error("❌ Invalid function.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    x0 = st.number_input("x₀", value=1.0)
    y0 = st.number_input("y₀", value=2.0)

    try:
        grad = sp.Matrix([fx, fy])
        grad_val = grad.subs({x: x0, y: y0})
        st.latex(f"\\nabla f = {sp.latex(grad)}")
        st.success(f"Gradient at ({x0}, {y0}) = {grad_val}")
    except Exception:
        st.error("❌ Cannot compute gradient at this point.")

    st.info(
        "The gradient vector points in the direction of steepest ascent. "
        "Its magnitude represents the maximum rate of increase."
    )

# -------------------------------------------------
# 5. Differentials
# -------------------------------------------------
elif topic == "Differentials":
    st.header("Differentials")

    expr_input = st.text_input("Enter f(x, y):", "x^2 + y^2")
    st.caption("Examples: x^2+y^2, sin(xy), exp(x+y)")

    f, error = parse_function(expr_input)
    if error:
        st.error("❌ Invalid function.")
        st.stop()

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    dx = st.number_input("dx", value=0.1)
    dy = st.number_input("dy", value=0.1)

    try:
        df = fx * dx + fy * dy
        st.latex("df = f_x dx + f_y dy")
        st.latex(f"df = {sp.latex(df)}")
    except Exception:
        st.error("❌ Cannot compute differential.")

    st.info(
        "Differentials give a linear approximation of the change in the function "
        "for small changes in x and y."
    )

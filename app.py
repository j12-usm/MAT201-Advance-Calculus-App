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
# Custom LN function class
# -----------------------------
class LN(sp.Function):
    """Custom natural logarithm function to distinguish from base-10 log"""
    
    nargs = 1
    
    @classmethod
    def eval(cls, arg):
        return None
    
    def fdiff(self, argindex=1):
        return 1/self.args[0]
    
    def _eval_evalf(self, prec):
        return sp.log(self.args[0])._eval_evalf(prec)

# -----------------------------
# Custom log10 function class
# -----------------------------
class Log10(sp.Function):
    """Custom base-10 logarithm function"""
    
    nargs = 1
    
    @classmethod
    def eval(cls, arg):
        return None
    
    def fdiff(self, argindex=1):
        return 1/(self.args[0] * LN(10))
    
    def _eval_evalf(self, prec):
        return sp.log(self.args[0], 10)._eval_evalf(prec)

# -----------------------------
# Custom LaTeX Printer
# -----------------------------
from sympy.printing.latex import LatexPrinter

class CustomLatexPrinter(LatexPrinter):
    def _print_Mul(self, expr):
        """Override multiplication to show implicit multiplication"""
        coeff, rest = expr.as_coeff_Mul()
        
        if coeff == 1 and all(isinstance(arg, sp.Symbol) or 
                              (isinstance(arg, sp.Pow) and isinstance(arg.base, sp.Symbol))
                              for arg in rest.args):
            return ''.join(self._print(arg) for arg in rest.args)
        
        return super()._print_Mul(expr)
    
    def _print_LN(self, expr):
        """Print natural logarithm as ln"""
        arg = expr.args[0]
        return r"\ln\!\left(%s\right)" % self._print(arg)
    
    def _print_Log10(self, expr):
        """Print base-10 logarithm as log"""
        arg = expr.args[0]
        return r"\log_{10}\!\left(%s\right)" % self._print(arg)

def latex_with_mixed_ln_log(expr):
    """Convert expression to LaTeX with proper ln/log distinction"""
    return CustomLatexPrinter().doprint(expr)

# -----------------------------
# Safe function parser
# -----------------------------
def parse_function(expr_input):
    try:
        expr_input = expr_input.replace("^", "**")
        expr_input = re.sub(r"\bln\s*\(", "LN(", expr_input)
        
        f = sp.sympify(expr_input, locals={
            "x": x, "y": y,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "exp": sp.exp, "sqrt": sp.sqrt,
            "log": Log10,
            "LN": LN,
            "e": sp.E,
        })
        
        return f, None
    
    except Exception as e:
        return None, str(e)

# -----------------------------
# Convert LN and Log10 to sp.log for numerical evaluation
# -----------------------------
def convert_for_numpy(expr):
    """Convert LN and Log10 to sp.log for lambdify"""
    expr = expr.replace(lambda e: isinstance(e, LN), lambda e: sp.log(e.args[0]))
    expr = expr.replace(lambda e: isinstance(e, Log10), lambda e: sp.log(e.args[0], 10))
    return expr

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
    for arg in expr.atoms(LN):
        conditions.append(sp.latex(arg.args[0]) + r" > 0")
    for arg in expr.atoms(Log10):
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

    st.latex(rf"f(x,y) = {latex_with_mixed_ln_log(f)}")

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
    f_eval = convert_for_numpy(f)
    f_np = sp.lambdify((x, y), f_eval, "numpy")
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
        x=1.15,
        len=0.75,
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
    st.latex(r"\frac{\partial f}{\partial x} = " + latex_with_mixed_ln_log(fx))
    st.latex(r"\frac{\partial f}{\partial y} = " + latex_with_mixed_ln_log(fy))


    col1, col2 = st.columns(2)
    with col1: x0 = st.number_input("x₀", value=1.0)
    with col2: y0 = st.number_input("y₀", value=1.0)

    fx_eval = convert_for_numpy(fx)
    fy_eval = convert_for_numpy(fy)
    fx_val = float(fx_eval.subs({x:x0, y:y0}))
    fy_val = float(fy_eval.subs({x:x0, y:y0}))
    st.success(f"At ({x0}, {y0}): ∂f/∂x = {fx_val:.3f}, ∂f/∂y = {fy_val:.3f}")

    f_eval = convert_for_numpy(f)
    if uses_x and uses_y:
        f_np = sp.lambdify((x, y), f_eval, "numpy")
        fx_np = sp.lambdify((x, y), fx_eval, "numpy")
        fy_np = sp.lambdify((x, y), fy_eval, "numpy")
    elif uses_x:
        f_np = sp.lambdify(x, f_eval, "numpy")
        fx_np = sp.lambdify(x, fx_eval, "numpy")
    elif uses_y:
        f_np = sp.lambdify(y, f_eval, "numpy")
        fy_np = sp.lambdify(y, fy_eval, "numpy")

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
        ax1.scatter(x0, f_np(x0, y0) if uses_y else f_np(x0), color="green", s=50)
        ax1.text(
            x0,
            f_np(x0, y0) if uses_y else f_np(x0),
            f"({x0:.2f},{(f_np(x0, y0) if uses_y else f_np(x0)):.2f})",
            fontsize=9, ha="left", va="bottom", color="green"
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x, y₀) and ∂f/∂x")
        ax1.set_title("Rate of Change w.r.t x")
        ax1.legend()
        ax1.grid(True)

        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.info("ℹ️ No x-direction graph (function does not depend on x).")

    
       # w.r.t y
    if uses_y:
        y_vals = t
        x_fixed = np.full_like(y_vals, x0) if uses_x else None

        f_vals_y = f_np(x_fixed, y_vals) if uses_x else f_np(y_vals)
        fy_vals = fy_np(x_fixed, y_vals) if uses_x else fy_np(y_vals)

        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(y_vals, f_vals_y, label=f"f({x0}, y)", color="blue")
        ax2.plot(y_vals, fy_vals, label=f"∂f/∂y at x={x0}", color="red", linestyle="--")
        ax2.scatter(
            y0,
            f_np(x0, y0) if uses_x else f_np(y0),
            color="green",
            s=50
        )
        ax2.text(
            y0,
            f_np(x0, y0) if uses_x else f_np(y0),
            f"({y0:.2f},{(f_np(x0, y0) if uses_x else f_np(y0)):.2f})",
            fontsize=9,
            ha="left",
            va="bottom",
            color="green"
        )

        ax2.set_xlabel("y")
        ax2.set_ylabel("f(x₀, y) and ∂f/∂y")
        ax2.set_title("Rate of Change w.r.t y")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("ℹ️ No y-direction graph (function does not depend on y).")

# =================================================
# 3. Differentials
# =================================================
elif topic == "Differentials":
    st.header("Differentials and Linear Approximation")

    # Helper function for 4sf with trailing zeros
    def format_4sf(val):
        if val == 0:
            return "0.000"
        digits_before = int(np.floor(np.log10(abs(val)))) + 1
        decimal_places = max(0, 4 - digits_before)
        return f"{val:.{decimal_places}f}"

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

    # -----------------------------
    # Partial derivatives
    # -----------------------------
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    st.subheader("Symbolic Partial Derivatives")
    st.latex(r"f_x = " + latex_with_mixed_ln_log(fx))
    st.latex(r"f_y = " + latex_with_mixed_ln_log(fy))

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
    # Step 1: Evaluate fx and fy at (x0,y0)
    # -----------------------------
    st.markdown("### Step 1: Evaluate partial derivatives at $(x_0, y_0)$")

    # fx
    fx_sub = fx.subs({x: x0, y: y0})
    fx_numeric = float(fx_sub)
    fx_numeric_str = format_4sf(fx_numeric)
    fx_formula = sp.latex(fx)
    fx_sub_formula = fx_formula.replace('x', f'({x0})').replace('y', f'({y0})')
    st.latex(
        rf"f_x(x_0, y_0) = f_x({x0},{y0}) = {fx_sub_formula} = {fx_numeric_str}"
    )

    # fy
    fy_sub = fy.subs({x: x0, y: y0})
    fy_numeric = float(fy_sub)
    fy_numeric_str = format_4sf(fy_numeric)
    fy_formula = sp.latex(fy)
    fy_sub_formula = fy_formula.replace('x', f'({x0})').replace('y', f'({y0})')
    st.latex(
        rf"f_y(x_0, y_0) = f_y({x0},{y0}) = {fy_sub_formula} = {fy_numeric_str}"
    )

    # -----------------------------
    # Step 2: Differential df and Δf
    # -----------------------------
    st.markdown("### Step 2: Differential df = f_x*dx + f_y*dy and change in f")

    # Actual change Δf
    f_x0_y0 = f.subs({x: x0, y: y0})
    f_x0_y0_str = format_4sf(float(f_x0_y0))
    actual_x = x0 + dx
    actual_y = y0 + dy
    f_actual_x_y = f.subs({x: actual_x, y: actual_y})
    f_actual_str = format_4sf(float(f_actual_x_y))

    # Show f(x0, y0)
    f_x0_y0_formula = sp.latex(f).replace('x', f'({x0})').replace('y', f'({y0})')
    st.latex(
        rf"f(x_0, y_0) = f({x0},{y0}) = {f_x0_y0_formula} = {f_x0_y0_str}"
    )

    # Show f(x0+dx, y0+dy)
    f_actual_formula = sp.latex(f).replace('x', f'({actual_x})').replace('y', f'({actual_y})')
    st.latex(
        rf"f(x_0+dx, y_0+dy) = f({actual_x},{actual_y}) = {f_actual_formula} = {f_actual_str}"
    )

    # Differential increment with detailed calculation
    df_x = fx_numeric * dx
    df_y = fy_numeric * dy
    df_numeric = df_x + df_y

    df_x_str = format_4sf(df_x)
    df_y_str = format_4sf(df_y)
    df_numeric_str = format_4sf(df_numeric)

    st.latex(
        rf"df = f_x*dx + f_y*dy = ({fx_numeric_str})*({dx}) + ({fy_numeric_str})*({dy}) = {df_x_str} + {df_y_str} = {df_numeric_str}"
    )

    # Δf calculation
    delta_f = float(f_actual_x_y - f_x0_y0)
    delta_f_str = format_4sf(delta_f)
    st.latex(
        rf"\Delta f = f(x_0+dx, y_0+dy) - f(x_0, y_0) = {f_actual_str} - {f_x0_y0_str} = {delta_f_str}"
    )

    # Error
    error_value = abs(delta_f - df_numeric)
    if error_value != 0:
        error_sci = f"{error_value/10**int(np.floor(np.log10(error_value))):.3g}*10^{int(np.floor(np.log10(error_value)))}"
    else:
        error_sci = "0"
    st.warning(f"Error of differential approximation = |Δf - df| ≈ {error_sci}")

    # -----------------------------
    # Linear approximation (separate)
    # -----------------------------
    st.markdown("### Linear Approximation (Tangent Plane)")

    # General L(x,y) formula
    st.latex(r"L(x, y) = f(x_0, y_0) + f_x(x_0, y_0) \cdot (x - x_0) + f_y(x_0, y_0) \cdot (y - y_0)")

    # Linear approximation increment shown with calculation
    L_increment_x = fx_numeric * dx
    L_increment_y = fy_numeric * dy
    L_increment = L_increment_x + L_increment_y

    L_increment_x_str = format_4sf(L_increment_x)
    L_increment_y_str = format_4sf(L_increment_y)
    L_increment_str = format_4sf(L_increment)

    st.latex(
        rf"Increment = f_x*dx + f_y*dy = ({fx_numeric_str})*({dx}) + ({fy_numeric_str})*({dy}) = {L_increment_x_str} + {L_increment_y_str} = {L_increment_str}"
    )

    # Linear approximation at (x0+dx,y0+dy)
    L_approx = float(f_x0_y0) + L_increment
    L_approx_str = format_4sf(L_approx)
    st.latex(
        rf"L(x_0 + dx, y_0 + dy) = f(x_0, y_0) + Increment = {f_x0_y0_str} + {L_increment_str} = {L_approx_str}"
    )

    # Linear approximation error shown as f(x0+dx,y0+dy) - L(x0+dx,y0+dy)
    linear_error = float(f_actual_x_y) - L_approx
    linear_error_str = format_4sf(linear_error)
    st.latex(
        rf"\text{{Linear Approximation Error}} = f(x_0+dx, y_0+dy) - L(x_0+dx, y_0+dy) = {f_actual_str} - {L_approx_str} = {linear_error_str}"
    )

    st.info(
        "Summary:\n"
        "- Differential df shows the linear change: df = f_x*dx + f_y*dy\n"
        "- Linear approximation L(x₀+dx, y₀+dy) uses the tangent plane at (x₀, y₀)\n"
        "- Smaller dx, dy → better approximation"
    )

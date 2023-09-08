import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.markdown("[![CryptoEconLab](./app/static/cover.png)](https://cryptoeconlab.io)")

st.sidebar.success("Select a Page above.")

st.markdown(
    """
### Filecoin Baseline Function Explorer
In this app, we explore variations of the baseline function and their effects on Filecoin's KPI's.

### How to use this app

**👈 Select "Baseline Exploration" from the sidebar** to get started. 

In this app, to explore variants of the baseline, enter a sympy expression in the text box, and then click "forecast."
The app will then run mechaFIL (Filecoin Economy Digital Twin) with the baseline function entered, as well as the original. Results
for network KPI's are then displayed.

An example of a sympy expression where the baseline function is defined such that it's growth rate is half of the current baseline function,
that is, it doubles every two years rather than every year, is given below:
```
g = np.log(2)/365.0
sym_t = sympy.symbols('t')
sympy_info = {
    'var': sym_t,
    'expr': sympy.exp(g*sym_t/2.0)*baseline_at_start,
}
```
A brief explanation is as follows:
- `g` is the growth rate of the baseline function. In this case, it is defined as half of the current baseline function. We can define it as a normal variable.
- `sym_t` is a sympy symbol that represents time. We define it as a sympy symbol, so that we can express new baseline functions that we want to try out, as a function of t.
- `sympy_info` is a dictionary that contains the sympy symbol `sym_t` and the sympy expression that defines the new baseline function. Notice here that we need to use sympy functions,
since they operate on a symbolic variable.  The backend executes this into a numerical vector that is then passed into the simulation.
- The variable `baseline_at_start` is a numerical value that represents the baseline function at the start of the simulation. 
    This can be used to scale the sympy expression to the correct value, as is shown in this example.

Refer to sympy for further details on how to enter expressions.

### Want to learn more?

- Check out [CryptoEconLab](https://cryptoeconlab.io)

- Engage with us on [X](https://x.com/cryptoeconlab)

- Read more of our research on [Medium](https://medium.com/cryptoeconlab) and [HackMD](https://hackmd.io/@cryptoecon/almanac/)

### Disclaimer
CryptoEconLab designed this application for informational purposes only. CryptoEconLab does not provide legal, tax, financial or investment advice. No party should act in reliance upon, or with the expectation of, any such advice.
"""
)
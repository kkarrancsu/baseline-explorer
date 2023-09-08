#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import date, timedelta
import pandas as pd

import numpy as np
import jax.numpy as jnp

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.date_utils as du
import mechafil_jax.minting as minting

import scenario_generator.utils as u

import sympy

import streamlit as st
import streamlit.components.v1 as components
from streamlit_ace import st_ace

import altair as alt

@st.cache_data
def get_offline_data(start_date, current_date, end_date):
    PUBLIC_AUTH_TOKEN='Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ'
    offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)

    _, hist_rbp = u.get_historical_daily_onboarded_power(current_date-timedelta(days=180), current_date)
    _, hist_rr = u.get_historical_renewal_rate(current_date-timedelta(days=180), current_date)
    _, hist_fpr = u.get_historical_filplus_rate(current_date-timedelta(days=180), current_date)

    smoothed_last_historical_rbp = float(np.median(hist_rbp[-30:]))
    smoothed_last_historical_rr = float(np.median(hist_rr[-30:]))
    smoothed_last_historical_fpr = float(np.median(hist_fpr[-30:]))

    return offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr

def compute_full_baseline(historical_baseline, start_date, current_date, end_date, code_str):
    historical_len = (current_date - start_date).days
    historical_baseline_until_start = historical_baseline[0:historical_len]
    
    ##########################################
    # These variables are defined by the codebase and can be used in the sympy expression
    baseline_at_start = historical_baseline_until_start[-1]
    ##########################################

    d = dict(locals(), **globals())
    exec(code_str, d, d)
    sympy_info_dict = d['sympy_info']
    sym_var = sympy_info_dict['var']
    b_t_expr = sympy_info_dict['expr']
    np_b_t = sympy.lambdify(sym_var, b_t_expr, 'numpy')
    
    arr_len = (end_date - current_date).days
    t_days = jnp.arange(0, arr_len)
    new_b_t_eval = np_b_t(t_days)
    
    return jnp.concatenate([historical_baseline_until_start, new_b_t_eval])

def make_plots(t, simulation_results_status_quo, simulation_results_configured_baseline, historical_baseline, configured_baseline):
    t_plot = pd.to_datetime(t)

    # setup df's for plotting
    power_dff = pd.DataFrame()
    power_dff['RBP'] = simulation_results_status_quo['rb_total_power_eib']
    power_dff['QAP'] = simulation_results_status_quo['qa_total_power_eib']
    power_dff['b(t)'] = historical_baseline
    power_dff['b*(t)'] = configured_baseline
    power_dff['date'] = t_plot

    minting_dff = pd.DataFrame()
    minting_dff['b(t)'] = simulation_results_status_quo['day_network_reward']
    minting_dff['b*(t)'] = simulation_results_configured_baseline['day_network_reward']
    minting_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    cs_dff = pd.DataFrame()
    cs_dff['b(t)'] = simulation_results_status_quo['circ_supply'] / 1e6
    cs_dff['b*(t)'] = simulation_results_configured_baseline['circ_supply'] / 1e6
    cs_dff['date'] = t_plot

    locked_dff = pd.DataFrame()
    locked_dff['b(t)'] = simulation_results_status_quo['network_locked'] / 1e6
    locked_dff['b*(t)'] = simulation_results_configured_baseline['network_locked'] / 1e6
    locked_dff['date'] = t_plot

    pledge_dff = pd.DataFrame()
    pledge_dff['b(t)'] = simulation_results_status_quo['day_pledge_per_QAP']
    pledge_dff['b*(t)'] = simulation_results_configured_baseline['day_pledge_per_QAP']
    pledge_dff['date'] = t_plot

    roi_dff = pd.DataFrame()
    roi_dff['b(t)'] = simulation_results_status_quo['1y_sector_roi'] * 100
    roi_dff['b*(t)'] = simulation_results_configured_baseline['1y_sector_roi'] * 100
    roi_dff['date'] = t_plot[0:len(simulation_results_status_quo['1y_sector_roi'])]
    
    col1, col2, col3 = st.columns(3)

    with col1:
        power_df = pd.melt(power_dff, id_vars=["date"], 
                           value_vars=[
                               "b(t)", "b*(t)", 
                               "RBP", "QAP",],
                           var_name='Power', 
                           value_name='EIB')
        power_df['EIB'] = power_df['EIB']
        power = (
            alt.Chart(power_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                    y=alt.Y("EIB").scale(type='log'), color=alt.Color('Power', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Network Power")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(power.interactive(), use_container_width=True) 

        # NOTE: adding the tooltip here causes the chart to not render for some reason
        # Following the directions here: https://docs.streamlit.io/library/api-reference/charts/st.altair_chart
        roi_df = pd.melt(roi_dff, id_vars=["date"], 
                         value_vars=["b(t)", "b*(t)"],
                         var_name='Scenario', 
                         value_name='%')
        roi = (
            alt.Chart(roi_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                    y=alt.Y("%"), color=alt.Color('Scenario', legend=alt.Legend(orient="top", title=None)))
            .properties(title="1Y Sector FoFR")
            .configure_title(fontSize=14, anchor='middle')
            # .add_params(hover)
        )
        st.altair_chart(roi.interactive(), use_container_width=True)

    with col2:
        # pledge_per_qap_df = my_melt(cil_df_historical, cil_df_forecast, 'day_pledge_per_QAP')
        pledge_per_qap_df = pd.melt(pledge_dff, id_vars=["date"],
                                    value_vars=["b(t)", "b*(t)"],
                                    var_name='Scenario', value_name='FIL')
        day_pledge_per_QAP = (
            alt.Chart(pledge_per_qap_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                    y=alt.Y("FIL"), color=alt.Color('Scenario', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Pledge/32GiB QAP")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(day_pledge_per_QAP.interactive(), use_container_width=True)

        minting_df = pd.melt(minting_dff, id_vars=["date"],
                             value_vars=["b(t)", "b*(t)"],
                             var_name='Scenario', value_name='FILRate')
        minting = (
            alt.Chart(minting_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                    y=alt.Y("FILRate", title='FIL/day'), color=alt.Color('Scenario', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Minting Rate")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(minting.interactive(), use_container_width=True)

    with col3:
        cs_df = pd.melt(cs_dff, id_vars=["date"],
                             value_vars=["b(t)", "b*(t)"],
                             var_name='Scenario', value_name='cs')
        cs = (
            alt.Chart(cs_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                    y=alt.Y("cs", title='M-FIL'), color=alt.Color('Scenario', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Circulating Supply")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(cs.interactive(), use_container_width=True)

        locked_df = pd.melt(locked_dff, id_vars=["date"],
                             value_vars=["b(t)", "b*(t)"],
                             var_name='Scenario', value_name='cs')
        locked = (
            alt.Chart(locked_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
                    y=alt.Y("cs", title='M-FIL'), color=alt.Color('Scenario', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Network Locked")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(locked.interactive(), use_container_width=True)


def forecast_economy(start_date=None, current_date=None, end_date=None, forecast_length_days=365*6):
    rb_onboard_power_pib_day =  st.session_state['rbp_slider']
    renewal_rate_pct = st.session_state['rr_slider']
    fil_plus_rate_pct = st.session_state['fpr_slider']
    
    lock_target = 0.3
    sector_duration_days = 360
    
    # get offline data
    offline_data, _, _, _ = get_offline_data(start_date, current_date, end_date)

    # run the simulation w/ the status-quo baseline
    rbp_val = rb_onboard_power_pib_day
    rr_val = max(0.0, min(1.0, renewal_rate_pct / 100.))
    fpr_val = max(0.0, min(1.0, fil_plus_rate_pct / 100.))
    
    rbp = jnp.ones(forecast_length_days) * rbp_val
    rr = jnp.ones(forecast_length_days) * rr_val
    fpr = jnp.ones(forecast_length_days) * fpr_val
    
    simulation_results_status_quo = sim.run_sim(
        rbp,
        rr,
        fpr,
        lock_target,

        start_date,
        current_date,
        forecast_length_days,
        sector_duration_days,
        offline_data
    )

    # run the simulation w/ the configured baseline
    # this should put the correct var (sympy_info) into the local namespace
    historical_baseline = minting.compute_baseline_power_array(
        np.datetime64(start_date), np.datetime64(end_date), offline_data['init_baseline_eib'],
    )
    code_str = st.session_state['sympy_code_block']
    if code_str is None:
        code_str = sympy_example
    
    b_t_eval = compute_full_baseline(historical_baseline, start_date, current_date, end_date, code_str)
    simulation_results_configured_baseline = sim.run_sim(
        rbp,
        rr,
        fpr,
        lock_target,

        start_date,
        current_date,
        forecast_length_days,
        sector_duration_days,
        offline_data,
        baseline_function_EIB = b_t_eval
    )
    t = du.get_t(start_date, end_date=end_date)

    make_plots(t, simulation_results_status_quo, simulation_results_configured_baseline, historical_baseline, b_t_eval) #, new_bt_label)
    

st.set_page_config(
    page_title="Filecoin Baseline Explorer",
    page_icon="🚀",  # TODO: can update this to the FIL logo
    layout="wide",
)
current_date = date.today() - timedelta(days=3)
mo_start = min(current_date.month - 1 % 12, 1)
start_date = date(current_date.year, mo_start, 1)
forecast_length_days=365*3
end_date = current_date + timedelta(days=forecast_length_days)
forecast_kwargs = {
    'start_date': start_date,
    'current_date': current_date,
    'end_date': end_date,
    'forecast_length_days': forecast_length_days,
}

_, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr = get_offline_data(start_date, current_date, end_date)
smoothed_last_historical_renewal_pct = int(smoothed_last_historical_rr * 100)
smoothed_last_historical_fil_plus_pct = int(smoothed_last_historical_fpr * 100)

with st.sidebar:
    st.title('Filecoin Economics Explorer')

    st.slider("Raw Byte Onboarding (PiB/day)", min_value=3., max_value=50., value=smoothed_last_historical_rbp, step=.1, format='%0.02f', key="rbp_slider",
            on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
    st.slider("Renewal Rate (Percentage)", min_value=10, max_value=99, value=smoothed_last_historical_renewal_pct, step=1, format='%d', key="rr_slider",
            on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
    st.slider("FIL+ Rate (Percentage)", min_value=10, max_value=99, value=smoothed_last_historical_fil_plus_pct, step=1, format='%d', key="fpr_slider",
            on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
    
    st.button("Forecast", on_click=forecast_economy, kwargs=forecast_kwargs, key="forecast_button")

# add code-block for baselne function configuration
sympy_example = """# `baseline_at_start` is the value of the baseline at the start of the forecast period
# and can be used to test out new trajectories that do not have discontinuities. This variable is predefined
# by the codebase and can be used in any expressions.  Note that the value of t over which the symbolic expression
# is evaluated is: t=[0:forecast_length_days], so configure the expression accordingly!

# Below is an example of a baseline function w/ half the growth rate of Filecoin's current b(t)
g = np.log(2)/365.0
sym_t = sympy.symbols('t')
sympy_info = {
    'var': sym_t,
    'expr': sympy.exp(g*sym_t/2.0)*baseline_at_start,
}
"""
content = st_ace(
    value=sympy_example,
    placeholder=sympy_example,
    height=300,
    language="python",
    auto_update=True,
    key="sympy_code_block",
    wrap=True,
)
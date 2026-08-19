from datetime import date

import pandas as pd
import plotly.graph_objects as go


def plot_conformal_intervals(
    df: pd.DataFrame, 
    shock_start: date, 
    shock_end: date, 
    shock_multiplier: float,
    title: str = "Energy demand forecast with conformal prediction intervals"
) -> go.Figure:
    """
    Plots historical series, point forecast, and adaptive conformal prediction intervals.

    Expects df to carry true_value, prediction, lower_bound, upper_bound on a
    DatetimeIndex. Shades the shock window only when shock_multiplier != 1.0.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['upper_bound'],
        mode='lines',
        line={"width": 0},
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['lower_bound'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line={"width": 0},
        name='Prediction interval'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['true_value'],
        mode='lines',
        line={"color": 'black', "width": 1},
        name='Observed'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['prediction'],
        mode='lines',
        line={"color": 'rgba(255, 127, 14, 1)', "width": 2, "dash": 'dot'},
        name='Point forecast'
    ))

    if shock_multiplier != 1.0:
        graph_start = df.index.min().date()
        graph_end = df.index.max().date()
        
        shade_start = max(shock_start, graph_start)
        shade_end = min(shock_end, graph_end)
        
        if shade_start <= shade_end:
            fig.add_vrect(
                x0=shade_start, x1=shade_end,
                fillcolor="rgba(255, 0, 0, 0.1)", 
                opacity=0.5, 
                layer="below", 
                line_width=0,
                annotation_text=f"Shock multiplier: {shock_multiplier}x", 
                annotation_position="top left",
                annotation_font_color="red"
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Energy Demand (kWh)",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
    )

    return fig
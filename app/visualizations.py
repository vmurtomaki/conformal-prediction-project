import plotly.graph_objects as go
import pandas as pd

def plot_conformal_intervals(df: pd.DataFrame, title: str = "Electricity Demand Forecast with Conformal Intervals") -> go.Figure:
    """
    Renders a Plotly chart showing point forecasts, true values, and dynamic uncertainty bounds.
    """
    fig = go.Figure()

    # Upper Bound (Invisible line to set the fill boundary)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['upper_bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Lower Bound (Fills upward to the Upper Bound)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['lower_bound'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(width=0),
        name='90% Prediction Interval'
    ))

    # True Values
    fig.add_trace(go.Scatter(
        x=df.index, y=df['true_value'],
        mode='lines',
        line=dict(color='black', width=1),
        name='True Demand (Actual)'
    ))

    # Point Prediction
    fig.add_trace(go.Scatter(
        x=df.index, y=df['prediction'],
        mode='lines',
        line=dict(color='rgba(255, 127, 14, 1)', width=2, dash='dot'),
        name='EnbPI Point Forecast'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Energy Consumption (kWh)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
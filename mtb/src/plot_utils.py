import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser" # notebook


def plotly_multi_dfs(
    dfs_dict, 
    x_col='elapsed_seconds',
    y_col='ax_uc_device',
    x_range=None,
    y_range=None,
    title=None,
    height=600,
    width=1000,
    colors=None,
    markers=True,
    marker_size=4,
    line_width=1,
    show_legend=True,
    connectgaps=True
):
    """
    Create a Plotly line graph with markers showing same y-axis from different dataframes.
    Useful for comparing original vs resampled/aligned data.
    
    Parameters:
        dfs_dict (dict): Dictionary of {name: dataframe} pairs to plot
        x_col (str): Column name for x-axis values
        y_col (str): Column name for y-axis values
        x_range (list, optional): [min, max] for x-axis range
        y_range (list, optional): [min, max] for y-axis range
        title (str, optional): Plot title
        height (int): Plot height
        width (int): Plot width
        colors (dict, optional): Dictionary of {name: color} pairs
        markers (bool): Whether to show markers
        marker_size (int): Size of markers
        line_width (int): Width of lines
        show_legend (bool): Whether to show the legend
        connectgaps (bool): Whether to connect gaps in the data
        
    Returns:
        fig: Plotly figure object
    """
    import plotly.graph_objects as go
    
    # Default title if none provided
    if title is None:
        title = f"Comparison of {y_col} signals"
    
    # Create the figure
    fig = go.Figure()
    
    # Default colors if none provided
    if colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = {name: default_colors[i % len(default_colors)] 
                 for i, name in enumerate(dfs_dict.keys())}
    
    # Mode based on markers setting
    mode = 'lines+markers' if markers else 'lines'
    
    # Add each dataframe's trace
    for name, df in dfs_dict.items():
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Warning: {x_col} or {y_col} not found in dataframe '{name}'. Skipping.")
            continue
        
        # Add the trace
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode=mode,
                name=name,
                line=dict(
                    color=colors.get(name, None),
                    width=line_width
                ),
                marker=dict(
                    size=marker_size,
                    opacity=0.7
                ),
                connectgaps=connectgaps
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=height,
        width=width,
        showlegend=show_legend,
        template='plotly_white',
        hovermode='closest'
    )
    
    # Set axis ranges if provided
    if x_range:
        fig.update_xaxes(range=x_range)
    if y_range:
        fig.update_yaxes(range=y_range)
    
    # Return and show the figure
    fig.show()
    return None


def plotly_multi_y(
    df: pd.DataFrame, 
    x_col='elapsed_seconds',
    y_cols: list[str] = ['gx_device', 'gy_device', 'gz_device'],
    x_range=None,
    y_range=None,
    title=None,
    height=600,
    width=1000,
    colors=None,
    markers=True,
    marker_size=4,
    line_width=1,
    show_legend=True,
    connectgaps=True
):
    """
    Create a Plotly line graph with markers showing same many y-values from same dataframe.
    Useful for comparing original vs resampled/aligned data.
    
    Parameters:
        dfs_dict (dict): Dictionary of {name: dataframe} pairs to plot
        x_col (str): Column name for x-axis values
        y_col (str): Column name for y-axis values
        x_range (list, optional): [min, max] for x-axis range
        y_range (list, optional): [min, max] for y-axis range
        title (str, optional): Plot title
        height (int): Plot height
        width (int): Plot width
        colors (dict, optional): Dictionary of {name: color} pairs
        markers (bool): Whether to show markers
        marker_size (int): Size of markers
        line_width (int): Width of lines
        show_legend (bool): Whether to show the legend
        connectgaps (bool): Whether to connect gaps in the data
        
    Returns:
        fig: Plotly figure object
    """
    import plotly.graph_objects as go
    
    # Create the figure
    fig = go.Figure()
    
    # Default colors if none provided
    if colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = {y_col: default_colors[i % len(default_colors)] 
                 for i, y_col in enumerate(y_cols)}
    
    # Mode based on markers setting
    mode = 'lines+markers' if markers else 'lines'
    
    # Add each dataframe's trace
    for y_col in y_cols:
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Warning: {x_col} or {y_col} not found in dataframe. Skipping.")
            continue
        
        # Add the trace
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode=mode,
                name=y_col,
                line=dict(
                    color=colors.get(y_col, None),
                    width=line_width
                ),
                marker=dict(
                    size=marker_size,
                    opacity=0.7
                ),
                connectgaps=connectgaps
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=height,
        width=width,
        showlegend=show_legend,
        template='plotly_white',
        hovermode='closest'
    )
    
    # Set axis ranges if provided
    if x_range:
        fig.update_xaxes(range=x_range)
    if y_range:
        fig.update_yaxes(range=y_range)
    
    # Return and show the figure
    fig.show()
    return None


def get_interactive_config():
    """Get interactive configuration for plotly graphs with mouse wheel zoom"""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'gold_price_chart',
            'height': 600,
            'width': 800,
            'scale': 1
        },
        'responsive': True,
        'scrollZoom': True,  # Enable mouse wheel zoom
        'doubleClick': 'reset+autosize',
        'showTips': True,
        'editable': False
    }

def apply_clean_layout(fig, title, height=500):
    """Apply clean layout styling to plotly figures"""
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        height=height,
        margin=dict(l=120, r=80, t=140, b=120),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        plot_bgcolor='rgba(255,255,255,0.95)',  # Clean white background
        paper_bgcolor='rgba(255,255,255,0.95)'
    )
    
    # Clean grid styling
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='rgba(0,0,0,0.1)',
        title_standoff=25,
        title_font_size=12
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='rgba(0,0,0,0.1)',
        title_standoff=25,
        title_font_size=12
    )
    
    return fig
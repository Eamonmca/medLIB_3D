def dash_ct_view(Scan):
    
    name = Scan.getName()
    fig = Scan.display_3D_volume()
    
    import plotly.graph_objects as go # or plotly.express as px # or any Plotly Express function e.g. px.bar(...)
    # fig.add_trace( ... )
    fig.update_layout(title=go.layout.Title(text=name))

    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    app = dash.Dash()
    app.layout = html.Div([
    dcc.Graph(figure=fig)
    ])

    return app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

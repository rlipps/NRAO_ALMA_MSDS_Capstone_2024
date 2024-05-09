# Packages
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import os
from dash import Dash, html, dcc, Input, Output, callback, dash_table

# Read data
topic_measurement = pd.read_csv('../data/model_outputs/topic_measurement.csv').set_index(['topic', 'measurement'])
topic_cluster = pd.read_csv('../data/model_outputs/topic_cluster.csv').set_index(['topic', 'cluster'])

# Dash code
# initialize app
app = Dash(__name__)

# define layout
app.layout = html.Div([
    html.H1('ALMA Spectral Line Measurement Explorer', style={'font-family':'arial'}),
    html.H2('This dashboard allows you to explore the measurements made for accepted\
            projects, grouped by project topic.', style={'font-family':'arial'}),
    html.H3('Some of the histogram lines are very narrow, so please refer to the scatterplot to observe all clusters before exploring the histogram.',
            style={'font-family':'arial'}),
    html.H3('You can click and drag areas on the plots to zoom in on specific regions.',
            style={'font-family':'arial'}),
    html.H3('Select a topic to explore with the dropdown menu below',
            style={'text-align': 'left', 'font-family': 'arial', 'color': 'black'}),
    dcc.Dropdown(
        id='topic-cluster-options',
        options=[{'label': str(i), 'value': i} for i in range(51)],
        value=0
    ),
    dcc.Graph(
        id='scatterplot'
    ),
    html.Div([
        dcc.RadioItems(
            id='y-axis-option',
            options=[
                {'label': 'Count of Measurements', 'value': 'count_meas'},
                {'label': 'Count of Projects', 'value': 'count_proj'}
            ],
            value='count_meas',  # Default option
            labelStyle={'display': 'inline-block'},
            style={'text-align': 'center'}
        )
    ], style={'margin': '10px auto', 'textAlign': 'left', 'color': 'black', 'font-family': 'arial'}),
    dcc.Graph(
        id='histogram'
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='cluster-histogram'),
        ], style={'width': '49%', 'height':'400px', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.H3('Selected Cluster Data', style={'text-align': 'center', 'black': 'white', 'font-family': 'arial', 'vertical_align':'top'}),
            html.Div(id='datatable-container', style={'display':'block', 'maxHeight': '400px', 'overflowY': 'auto', 'vertical_align':'bottom', 'horizontal_align':'middle'})
        ], style={'width': '49%', 'height':'400px', 'display': 'inline-block', 'vertical-align': 'bottom'})
    ], style={'height':'400px'})
])

# define callbacks
@app.callback([
    Output('histogram', 'figure'),
    Output('scatterplot', 'figure'),
    Output('cluster-histogram', 'figure'),
    Output('datatable-container', 'children')],
    [Input('topic-cluster-options', 'value'),
     Input('histogram', 'clickData'),
     Input('y-axis-option', 'value')])

def update_graph(inspect_topic, click_data, y_axis_option):

    inspect_topic_frame = topic_measurement.loc[inspect_topic]

    inspect_topic_frame = inspect_topic_frame.sort_values('cluster_label', ascending=False)
    inspect_topic_frame.cluster_label = inspect_topic_frame.cluster_label.astype('str')

    symbols = list(np.zeros(np.unique(inspect_topic_frame.cluster_label).shape[0], 'int'))
    symbols[-1] = 'x'

    # Add noise binary column for plot symbol
    inspect_topic_frame['noise'] = np.where(inspect_topic_frame.cluster_label == '-1', 1, 0)

    # Noise and Signal
    itf_noise = inspect_topic_frame.noise.sum()
    itf_signal = inspect_topic_frame.shape[0] - itf_noise

    # Create filtered dataframe to power histogram
    # We use query because we want to exclude noise
    filtered_df = topic_cluster.query(f'topic== {inspect_topic} and cluster != -1')

    hist = go.Figure()
    hist.add_trace(go.Bar(
        x=filtered_df.med_freq,
        y=filtered_df[y_axis_option],  # Use selected y-axis option
        marker=dict(color=filtered_df.count_proj, colorscale='bluered'),
        width=filtered_df.width.to_list(),
        name=f'Cluster for Topic {inspect_topic}',
        hovertemplate=
        '<i>Cluster Number</i>: %{customdata[0]}<br>' +
        '<i>Count of Projects</i>: %{customdata[1]}<br>' +
        '<i>Count of Measurements</i>: %{customdata[2]}<br>' +
        '<i>Minimum Frequency</i>: %{customdata[3]}<br>' +
        '<i>Median Frequency</i>: %{customdata[4]}<br>' +
        '<i>Maximum Frequency</i>: %{customdata[5]}',
        customdata=list(zip(filtered_df.index.get_level_values(level=1),
                            filtered_df.count_proj,
                            filtered_df.count_meas,
                            filtered_df.min_freq,
                            filtered_df.med_freq,
                            filtered_df.max_freq))
    )),

    hist.update_layout(
        title=(f'Areas of Interest for Topic {inspect_topic}<br><sup>Click and drag to zoom in on areas<br>Click a bin to see underlying measurement distribution</sup>'),
        xaxis_title = 'Frequency (GHz)',
        yaxis_title = y_axis_option.replace('_', ' ').title(),  # Use selected y-axis option
        #coloraxis_colorbar=dict(title='Count of Projects'),
        transition_duration=500,
    )

    hist.layout.template = 'plotly'

    scatter = px.scatter(inspect_topic_frame,
                         x='med_freq',
                         y='cluster_label',
                         color='cluster_label',
                         symbol='cluster_label',
                         symbol_sequence=symbols,
                         title=f"HDBSCAN Generated Clusters for Topic {inspect_topic} <br><sup>{itf_signal} Clustered Measurements with {itf_noise} Noise Measurements</sup>",
                         labels={'med_freq': 'Median Frequency (GHz)',
                                 'index': 'Index',
                                 'cluster_label': 'Cluster Label'
                                 })
    scatter.update_traces(marker={'size': 15, 'opacity': 0.5})

    scatter.layout.template = 'plotly'

    # Default cluster label for cluster histogram
    cluster_label = 0

    # If click data is available, update cluster_label
    if click_data:
        cluster_label = click_data['points'][0]['pointNumber']

    # Create histogram for selected cluster_label
    cluster_hist = px.histogram(inspect_topic_frame[inspect_topic_frame['cluster_label'] == str(cluster_label)],
                                x='med_freq',
                                title=f"Histogram of Measurements for Cluster {cluster_label} (Topic {inspect_topic})",
                                labels={'med_freq': 'Median Frequency (GHz)', 'count': 'Count'})
    cluster_hist.update_layout(yaxis_title='Count of Measurements')
    cluster_hist.layout.template = 'plotly'

    # Create data table for selected cluster_label
    data_table = dash_table.DataTable(
        id='datatable',
        columns=[{'name': i, 'id': i} for i in inspect_topic_frame.columns],
        data=inspect_topic_frame[inspect_topic_frame['cluster_label'] == str(cluster_label)].sort_values('med_freq', ascending=True).to_dict('records'),
        style_table={'overflowX': 'scroll'},
    style_data={
        'color': 'black',
        'backgroundColor': 'white',
        'font_family':'arial'
    },
    style_as_list_view=True,
    style_cell={
        'text_align':'center'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'Silver',
        }
    ],
    style_header={
        'backgroundColor': 'white',
        'color': 'black',
        'font_family':'arial',
        'fontWeight': 'bold'
    }
    )

    return hist, scatter, cluster_hist, data_table

# run app
if __name__ == '__main__':
    app.run(jupyter_mode='inline', debug=True)

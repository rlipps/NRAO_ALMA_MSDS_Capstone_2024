# Packages
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import os
from dash import Dash, html, dcc, Input, Output, callback, dash_table

# Read data
topic_words = pd.read_csv('../data/model_outputs/topic_words.csv').set_index(['topic', 'word_number'])
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
    html.Div(id='topic-words-container', style={'text-align': 'left', 'font-family': 'arial', 'color': 'black'}),
    html.Div([
        html.H3('Select band(s)'),
        dcc.Checklist(
        id='band-selector',
        options=[{'label': band, 'value': band} for band in np.sort(topic_cluster['min_band'].unique())],
        value=topic_cluster['min_band'].unique(),  # Default value
        labelStyle={'display': 'inline-block'}
        ), 
    ], style={'text-align': 'left', 'font-family': 'arial', 'color': 'black'}
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

# Define a callback function to update the topic words based on the selected topic
@app.callback(
    Output('topic-words-container', 'children'),
    [Input('topic-cluster-options', 'value')]
)
def update_topic_words(selected_topic):
    # Filter topic_words based on the selected topic
    selected_topic_words = topic_words.query(f'topic == {selected_topic}').word.values
    
    # If no words found for the selected topic, display no words found
    if len(selected_topic_words) == 0:
        return html.P('No words available for the selected topic.')
    
    # Join the words into a single string and display them
    topic_words_str = ', '.join(selected_topic_words)
    return html.P(f'Topic {selected_topic} Top Words: {topic_words_str}')


# define callbacks
@app.callback([
    Output('histogram', 'figure'),
    Output('scatterplot', 'figure'),
    Output('cluster-histogram', 'figure'),
    Output('datatable-container', 'children')],
    [Input('topic-cluster-options', 'value'),
     Input('histogram', 'clickData'),
     Input('y-axis-option', 'value'),
     Input('band-selector', 'value')])

def update_graph(inspect_topic, click_data, y_axis_option, selected_bands):

    inspect_topic_frame = topic_measurement.loc[inspect_topic].query(f'band in {selected_bands}')

    inspect_topic_frame = inspect_topic_frame.sort_values('cluster_label', ascending=False)

    # Map cluster_labels to symbols for plot
    def get_symbol(cl):
        if cl == -1:
            return 'x'
        else:
            return 'circle'
    symbol_map = {}
    for cl in inspect_topic_frame.cluster_label.unique():
        symbol_map[cl.astype('str')] = get_symbol(cl)

    inspect_topic_frame.cluster_label = inspect_topic_frame.cluster_label.astype('str')

    # Add noise binary column for plot symbol
    inspect_topic_frame['noise'] = np.where(inspect_topic_frame.cluster_label == '-1', 1, 0)
    #inspect_topic_frame['symbol'] = np.where(inspect_topic_frame.cluster_label == '-1', 'x', 'circle')

    # Noise and Signal
    itf_noise = inspect_topic_frame.noise.sum()
    itf_signal = inspect_topic_frame.shape[0] - itf_noise

    # Create filtered dataframe to power histogram
    # We use query because we want to exclude noise
    filtered_df = topic_cluster.query(f'topic == {inspect_topic} and cluster != -1 and min_band in {selected_bands}')

    hist = go.Figure()
    hist.add_trace(go.Bar(
        x=filtered_df.med_freq,
        y=filtered_df[y_axis_option],  # Use selected y-axis option
        marker=dict(color=filtered_df.count_proj, colorscale='bluered', line=dict(width=0.1, color='black')),
        width=filtered_df.width.to_list(),
        name=f'Cluster for Topic {inspect_topic}',
        hovertemplate=
        '<i>Cluster Number</i>: %{customdata[0]}<br>' +
        '<i>Count of Projects</i>: %{customdata[1]}<br>' +
        '<i>Count of Measurements</i>: %{customdata[2]}<br>' +
        '<i>Band</i>: %{customdata[3]}<br>' +
        '<i>Minimum Frequency</i>: %{customdata[4]}<br>' +
        '<i>Median Frequency</i>: %{customdata[5]}<br>' +
        '<i>Maximum Frequency</i>: %{customdata[6]}',
        customdata=list(zip(filtered_df.index.get_level_values(level=1),
                            filtered_df.count_proj,
                            filtered_df.count_meas,
                            filtered_df.min_band,
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
                         symbol_map=symbol_map,
                         custom_data=['cluster_label', 'band', 'low_freq', 'med_freq', 'high_freq', 'project_code'],
                         title=f"HDBSCAN Generated Clusters for Topic {inspect_topic} <br><sup>{itf_signal} Clustered Measurements with {itf_noise} Noise Measurements</sup>",
                         labels={'med_freq': 'Median Frequency (GHz)',
                                 'index': 'Index',
                                 'cluster_label': 'Cluster Label'},
                        )
    scatter.update_traces(marker={'size': 15, 'opacity': 0.5},
                          hovertemplate =
                            '<i>Cluster Label</i>: %{customdata[0]}<br>' +
                            '<i>Band</i>: %{customdata[1]}<br>' +
                            '<i>Low Frequency</i>: %{customdata[2]}<br>' +
                            '<i>Median Frequency</i>: %{customdata[3]}<br>' +
                            '<i>High Frequency</i>: %{customdata[4]}<br>' +
                            '<i>Project Code</i>: %{customdata[5]}<br>')

    scatter.layout.template = 'plotly'

    # Default cluster label for cluster histogram
    cluster_label = 0

    # If click data is available, update cluster_label
    if click_data:
        cluster_label = click_data['points'][0]['customdata'][0]

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

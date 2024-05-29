# Model Packages
import pandas as pd
import numpy as np
import nltk
import re
import string
from joblib import load
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Dashboard Packages
import plotly_express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback, dash_table

# NRAO Stopword list
nrao_stops = ['0','1','2','3','4','5','6','7','8','9',"=",'<','>','~',"/",'`',".",",", 'observation'
                       'alma','resolution','source','show','sample','high','use','observe'
                       'figure','low','image','propose', 'also','use', 'large', 'study'
                       'reference', 'detect','see','well', 'time', 'however', 'expect'
                       'provide','datum', 'model', 'result','sensitivity','scale'
                       'find','allow','scientific','target','compare','resolve','first',
                       'leave', 'estimate','suggest', 'due', 'obtain', 'small', 'measure',
                       'include','property','justification', 'right ', 'understand', 'similar',
                       'detection', 'require', 'indicate', 'order', 'range', 'make','map','thus',
                       'follow','fig','goal','proposal','field','determine','therefore', 'reveal',
                       'give', 'process','total', 'important', 'know','constrain', 'ratio','even',
                       'case','et', 'al', 'pc','kpc','apj','km','mm','m','one','two', 'data', 'us',
                       'mnras', 'left', 'right', 'may', 'within','would','need','request','mjy',
                       'different','assume','recent','good','since','still','previous','science',
                       'ghz','could','object','much','survey','three','whether','likely','several',
                       'like','able','identify','new','best','number','analysis','confirm','predict',
                       'le','evidence','select','example','take','recently','combine','exist','value',
                       'fit','objective','comparison','investigate','respectively','many','although',
                       'achieve','cm','jy','need','enough','search','yr','explain','au','apjl','per','arxiv',
                       'a&a', 'aa', 'apj', 'apjl', 'mnras', 'pasp', 'aj', 'cycle', 'band',
                       'emission', 'free', 'anticipate', 'originate', 'success', 'separate', 'uv', 'significance',
                       'hot', 'frequency', 'wavelength', 'realistic', 'mas', 'mg', 'minute', 'ii', 'ad', 'hd',
                       'occurrence', 'event', 'myr', 'ra', 'dec',  'ly', 'tau', 'cn',
                       'arc', 'ori', 'hh', 'iii', 'cha', 'ab', 'tw', 'ms', 'ngc', 'pds',
                       'jwst','hcn','hco+','oh','xray','aca', 'vla', 'gbt', 'proto', 'noema',
                       'quiescent','nir', 'heating', 'sb','temperature','cr','hya','liu','warm',
                       'nh','extent','spitzer', 'co','yang']

# Text processing functions
#convert to lowercase, strip and remove punctuations and remove ALMA, case insensitive
def preprocess(text):
    text = text.lower()                                                         # Make everything lower case
    text = text.strip()                                                         # Strip leading and trailing whitespace
    text = re.compile('<.*?>').sub('', text)                                    # Remove things like html tags 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)    # Remove punctuation
    text = re.sub(r'(?i)alma', '', text)                                        # Remove case insensitive 'alma'
    text = re.sub('\s+', ' ', text)                                             # Convert whitespace to single space
    text = re.sub(r'\[[0-9]*\]',' ',text)                                       # Remove things like citations e.g. [9]
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())                    # Remove non alphanumeric characters
    text = re.sub(r'\d',' ',text)                                               # Remove digits
    text = re.sub(r'\s+',' ',text)                                              # Collapse any created whitespace into single space
    return text
 
# STOPWORD REMOVAL
def stopword(string):
    sws = stopwords.words('english')
    sws.extend(nrao_stops)
    a= [i for i in string.split() if i not in sws]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

# Import models from joblib files
logreg_tfidf_vectorizer = load('models/line_continuum_classifier/tfidf_vectorizer_logreg.joblib')
logreg_classifier = load('models/line_continuum_classifier/log_model.joblib')
naive_bayes_tfidf_vectorizer = load('models/band_classification/tfidf_vectorizer_naive_bayes.joblib')
naive_bayes_model = load('models/band_classification/naive_bayes_model.joblib')
lda_count_vectorizer = load('models/topic_mining/lda_count_vectorizer.joblib')
lda_model = load('models/topic_mining/lda_model.joblib')

# Prompt user for title and abstract
print('\nWelcome to the ALMA Project Explorer!')
# title = input('Enter project title:')
# abstract = input('Enter project abstract:')
# raw_text = title + '. ' + abstract
with open('input.txt', 'r') as infile:
    raw_text = infile.read()

# Create dataframe of input with various text processing required for models
df = pd.DataFrame({
    'raw_text':raw_text
}, index=[1])

df['std_text'] = df.raw_text.apply(lambda x: preprocess(x))
df['std_text_sw_removed'] = df.std_text.apply(lambda x: stopword(x))
df['std_text_sw_removed_lemmatized'] = df.std_text_sw_removed.apply(lambda x: lemmatizer(x))

# Predict from models
# Logistic Regression for Line/Continuum classification
logreg_pred = logreg_classifier.predict_proba(logreg_tfidf_vectorizer.transform(df.std_text))
print(f'\nPredicted probability of only continuum measurements: {round(logreg_pred[0][0]*100, 3)}')
print(f'Predicted probability of at least one line measurement: {round(logreg_pred[0][1]*100, 3)}')

# # Topic assignment
lda_pred = np.argmax(lda_model.transform(lda_count_vectorizer.transform(df.std_text_sw_removed_lemmatized)))
print(f'Predicted project topic number: {lda_pred}')

# Band prediction
band_pred = naive_bayes_model.predict_proba(naive_bayes_tfidf_vectorizer.transform(df.std_text_sw_removed_lemmatized))
sorted_indices = np.argsort(band_pred)          # Get indices for all band predictions
sorted_indices = np.flip(sorted_indices)        # Sort in descending probability
for prediction in range(len(sorted_indices)):   # Manage class relabeling to match AMLA bands
    for band in range(len(sorted_indices[prediction])):
        if sorted_indices[prediction][band] != 0:
            sorted_indices[prediction][band] += 2 # need to add 2 to index to equal the band that was predicted band (3, 4, 5, 6, 7, 8, 9, or 10)
        else:
            sorted_indices[prediction][band] += 1 # need to add 1 to index to equal the band that was predicted (band 1)
print(f'Top two predicted bands: {sorted_indices[0][:2]}\n')

# Read data
topic_words = pd.read_csv('data/model_outputs/topic_words.csv').set_index(['topic', 'word_number'])
topic_measurement = pd.read_csv('data/model_outputs/topic_measurement.csv').set_index(['topic', 'measurement'])
topic_cluster = pd.read_csv('data/model_outputs/topic_cluster.csv').set_index(['topic', 'cluster'])

# Dash code
# initialize app
app = Dash(__name__)

# define layout
app.layout = html.Div([
    html.H1('ALMA Spectral Line Measurement Explorer', style={'font-family':'arial'}),
    html.H2('This dashboard allows you to explore the measurements made for accepted\
            projects, grouped by project topic.', style={'font-family':'arial'}),
    html.H3('Some of the histogram lines are very narrow, so please cross-reference the scatterplot with the histogram.',
            style={'font-family':'arial'}),
    html.H3('You can click and drag areas on the plots to zoom in on specific regions.',
            style={'font-family':'arial'}),
    html.H3('The topic below is your predicted topic, but feel free to explore with the dropdown menu below',
            style={'text-align': 'left', 'font-family': 'arial', 'color': 'black'}),
    dcc.Dropdown(
        id='topic-cluster-options',
        options=[{'label': str(i), 'value': i} for i in range(51)],
        value=lda_pred  # Default value from lda prediction
    ),
    html.Div(id='topic-words-container', style={'text-align': 'left', 'font-family': 'arial', 'color': 'black'}),
    html.Div([
        html.H3('Select band(s)'),
        dcc.Checklist(
        id='band-selector',
        options=[{'label': band, 'value': band} for band in np.sort(topic_cluster['min_band'].unique())],
        value=sorted_indices[0][:2],  # Default value from band_prediction
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
    app.run(jupyter_mode='inline', debug=True, use_reloader=False)

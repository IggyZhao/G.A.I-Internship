import numpy as np
import pandas as pd


import dash
import dash_auth
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64

import pymongo
import dns
import json
import dash_bootstrap_components as dbc
import plotly
from plotly.offline import plot
import random
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.figure_factory as ff





#---------------------------------------------- import data from csv ----------------------------------------------
# stock fundamental info
df_fundinfo = pd.read_csv('appdata/df_fundinfo.csv',index_col=0)

# LSTM model prediction result
df_res = pd.read_csv("appdata/df_res.csv")

# LSTM model (with SDG scores as risk factors) prediction result 
df_ress = pd.read_csv('appdata/df_ress.csv')

# COVID-19 key dates and events
df_covid = pd.read_csv('appdata/covid_timeline.csv')

# SDG, SDG_adj scores change at COVID-19 key dates (by sector, by SDG, by overall)
df_SDG_cv = pd.read_csv("appdata/df_SDG_cv.csv")
del df_SDG_cv['Unnamed: 0']
df_SDG_adj_cv = pd.read_csv("appdata/df_SDG_adj_cv.csv")

# dates for SDG, SDG_adj, Sentiment score outliers
df_outlier = pd.read_csv('appdata/df_outlier.csv')
del df_outlier['Unnamed: 0']
df_adj_outlier = pd.read_csv('appdata/df_adj_outlier.csv')
del df_adj_outlier['Unnamed: 0']
df_stm_outlier = pd.read_csv('appdata/df_stm_outlier.csv')
del df_stm_outlier['Unnamed: 0']

# rank of SDG, SDG_adj, Sentiment scores changes at key dates
df_SDG_rank = pd.read_csv('appdata/df_SDG_rank.csv')
df_SDG_adj_rank = pd.read_csv('appdata/df_SDG_adj_rank.csv')
df_stm_rank = pd.read_csv('appdata/df_stm_rank.csv')

# SDG, SDG_adj, Sentiment scores change of major company per sector
SDG_maj = pd.read_csv('appdata/SDG_maj.csv')
del SDG_maj['Unnamed: 0']
SDG_adj_maj = pd.read_csv('appdata/SDG_adj_maj.csv')
del SDG_adj_maj['Unnamed: 0']
stm_maj = pd.read_csv('appdata/stm_maj.csv')
del stm_maj['Unnamed: 0']








#-------------------------------------------------- app data --------------------------------------------------

### FOR EVERYBODY
## stock fundamental info
sf_ticker = 'AAL'
df_sf = df_fundinfo.loc[df_fundinfo['Ticker']==sf_ticker][['Timestamp','Adj. Close','P/Book','P/E']]
sf = []
sf.append(go.Scatter(x=df_sf['Timestamp'].values,y=df_sf['Adj. Close'].values,name='adj price',mode='lines'))
sf.append(go.Scatter(x=df_sf['Timestamp'].values,y=df_sf['P/Book'].values,name='P/B',mode='lines'))
sf.append(go.Scatter(x=df_sf['Timestamp'].values,y=df_sf['P/E'].values,name='P/E',mode='lines'))

sf_fig = {'data':sf,
            'layout':go.Layout(xaxis={'title':'date'},yaxis={'title':'stock price'},
                               title = 'adj price, P/B and P/E for '+sf_ticker, hovermode='closest')}


## garch result
image_filename0 = 'appdata/garch.png'
encoded_image0 = base64.b64encode(open(image_filename0, 'rb').read())
image0 = 'data:image/png;base64,{}'.format(encoded_image0.decode())

## lstm result
n_train = int(df_res.shape[0]*0.9)
ts = []
ts.append(go.Scatter(x=df_res['date'][:n_train].values,y=df_res['actual'][:n_train].values,name='Train',mode='lines'))
ts.append(go.Scatter(x=df_res['date'][n_train:].values,y=df_res['actual'][n_train:].values,name='Test',mode='lines'))
ts.append(go.Scatter(x=df_res['date'][n_train:].values,y=df_res['model'][n_train:].values,name='Forecasted',mode='lines'))

line_fig = {'data':ts,
            'layout':go.Layout(xaxis={'title':'date'},yaxis={'title':'stock price'},
                               title = 'LSTM model result', hovermode='closest')}
# with SDG as risk factors
# train & test loss
image_filename0s = 'appdata/lstm_loss.png'
encoded_image0s = base64.b64encode(open(image_filename0s, 'rb').read())
image0s = 'data:image/png;base64,{}'.format(encoded_image0s.decode())
# model result
n_trains = int(df_ress.shape[0]*0.9)
tss = []
tss.append(go.Scatter(x=df_ress['date'][:n_trains].values,y=df_ress['actual'][:n_trains].values,name='Train',mode='lines'))
tss.append(go.Scatter(x=df_ress['date'][n_trains:].values,y=df_ress['actual'][n_trains:].values,name='Test',mode='lines'))
tss.append(go.Scatter(x=df_ress['date'][n_trains:].values,y=df_ress['model'][n_trains:].values,name='Forecasted',mode='lines'))

line_figs = {'data':tss,
            'layout':go.Layout(xaxis={'title':'date'},yaxis={'title':'stock price'},
                               title = 'LSTM model result (with SDG scores as risk factors)', hovermode='closest')}


### DATA SCIENCE
## dropdown choices
SDG_choice = ['SDG','SDG_adj','sentiment']

## SDG change at covid-19 key dates
image_filename1 = 'appdata/by_sector.png'
encoded_image1 = base64.b64encode(open(image_filename1, 'rb').read())
by_sector = 'data:image/png;base64,{}'.format(encoded_image1.decode())

image_filename1a = 'appdata/by_sector_adj.png'
encoded_image1a = base64.b64encode(open(image_filename1a, 'rb').read())
by_sector_adj = 'data:image/png;base64,{}'.format(encoded_image1a.decode())

image_filename2 = 'appdata/by_SDG.png'
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())
by_SDG = 'data:image/png;base64,{}'.format(encoded_image2.decode())

image_filename2a = 'appdata/by_SDG_adj.png'
encoded_image2a = base64.b64encode(open(image_filename2a, 'rb').read())
by_SDG_adj = 'data:image/png;base64,{}'.format(encoded_image2a.decode())

image_filename3 = 'appdata/by_overall.png'
encoded_image3 = base64.b64encode(open(image_filename3, 'rb').read())
by_overall = 'data:image/png;base64,{}'.format(encoded_image3.decode())

image_filename3a = 'appdata/by_overall_adj.png'
encoded_image3a = base64.b64encode(open(image_filename3a, 'rb').read())
by_overall_adj = 'data:image/png;base64,{}'.format(encoded_image3a.decode())


# the major companies we select for the 11 sectors are as follows

# communication services: FB
# consumer discretionary: AMZN
# consumer staples: WMT
# energy: XOM
# financials: GS
# health care: CVS
# industrials: GE
# information technology: AAPL
# materials: ECL
# real estate: AMT
# utilities:NEE

## major companies figure
major_ticker = ['FB','AMZN','WMT','XOM','GS','CVS','GE','AAPL','ECL','AMT','NEE']
industry_ticker = ['Communication Services','Consumer Discretionary','Consumer Staples','Energy','Financials',
                  'Health Care','Industrials','Information Technology','Materials','Real Estate','Utilities']
# SDG
sdgmaj_fig = tools.make_subplots(rows=3, cols=4, subplot_titles=industry_ticker,
                                 specs=[[{'type': 'xy'}]*4]*3 )
for i in range(len(major_ticker)):
    sdgmaj_fig.append_trace(
        go.Scatter(
            name = major_ticker[i], mode = 'lines',
            x=SDG_maj[SDG_maj['Ticker']==major_ticker[i]]['Timestamp'].values,
            y=SDG_maj[SDG_maj['Ticker']==major_ticker[i]]['SDG_Mean_Chg'].values
        ),(i//4)+1, (i%4)+1
    )
sdgmaj_fig['layout'].update(title='major companies per sector',title_x=0.5)

# SDG_adj
sdgadjmaj_fig = tools.make_subplots(rows=3, cols=4, subplot_titles=industry_ticker,
                                 specs=[[{'type': 'xy'}]*4]*3 )
for i in range(len(major_ticker)):
    sdgadjmaj_fig.append_trace(
        go.Scatter(
            name = major_ticker[i], mode = 'lines',
            x=SDG_adj_maj[SDG_adj_maj['Ticker']==major_ticker[i]]['Timestamp'].values,
            y=SDG_adj_maj[SDG_adj_maj['Ticker']==major_ticker[i]]['SDG_Mean_ADJ_Chg'].values
        ),(i//4)+1, (i%4)+1
    )
sdgadjmaj_fig['layout'].update(title='major companies per sector',title_x=0.5)

# Sentiment
stmmaj_fig = tools.make_subplots(rows=3, cols=4, subplot_titles=industry_ticker,specs=[[{'type': 'xy'}]*4]*3)
for i in range(len(major_ticker)):
    stmmaj_fig.append_trace(
        go.Scatter(
            name = major_ticker[i], mode = 'lines',
            x=stm_maj[stm_maj['Ticker']==major_ticker[i]]['Timestamp'].values,
            y=stm_maj[stm_maj['Ticker']==major_ticker[i]]['Sentiment_Chg'].values
        ),(i//4)+1, (i%4)+1
    )
stmmaj_fig['layout'].update(title='major companies per sector',title_x=0.5)







app = dash.Dash()

#--------------------------------------------- main page layout ---------------------------------------------
app.layout = html.Div([
    html.Span(id="vir_span_view", style={"display": "none"}, children=0),
    html.H1(
        'Stock Analysis',
        style={
            'textAlign': 'center', 'color': '#333', 'font-size': '30px',
            'height': '20px', 'line-height': '30px', 'padding-top': "20px"
        }
    ),
    html.Div([
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='For Everybody', value='tab-1',
                    style={"height": 60,'font-size': '20px'},
                    selected_style={"height": 60,'font-size': '20px'}),
            dcc.Tab(label='Data Science', value='tab-2',
                    style={"height": 60,'font-size': '20px'},
                    selected_style={"height": 60,'font-size': '20px'}),
        ]),
    ],style={"height": "40px"}),
    html.Div(style=dict(clear="both")),
    html.Div(id='tabs-content')
],style={"padding": "0 40px", "background": "#F2F2F2"})

#---------------------------------------- main page callback function ----------------------------------------
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1_layout
    elif tab == 'tab-2':
        return tab2_layout



    
#------------------------------------------------ tab1 layout ------------------------------------------------
tab1_layout = html.Div([  # whole page
    
    # part1 stock fundamental information
    html.Div([
        html.Div([html.H2('Part 1: Stock Fundamental Information')],
                 style = {'textAlign': 'center','padding-top': "25px"})
        ]),
    
    # fundamental information of stocks
    # title
    html.Div([
        html.Div([html.H3('fundamental information of stocks (including P/E, P/B)')],
                 style = {'width':'52%','display':'inline-block','margin-top':5}),
        html.Div([html.H3('stock price, P/E and P/B')],
                 style = {'width':'48%','display':'inline-block','margin-top':5})
        ]),
    # table
    html.Div([
        html.Div([
            dash_table.DataTable(
                id='datatable-paging',
                columns=[{"name": i, "id": i} for i in list(df_fundinfo.columns)],
                page_current=0, page_size=10, page_action='custom', fixed_columns={'headers': True,'data': 1},
                style_table={'height': '380px','overflowY': 'auto','width': '100%','minWidth': '100%'})   
        ], style = {'margin-top':5,'width':'48%','display':'inline-block',}),
        
        html.Div([dcc.Graph(id='stock-fund-info',figure = sf_fig,
                            style={"height":"90%","width":"90%","padding-left":"5%"})],
                 style={"float": "right", "width": "52%",'display':'inline-block'}),
        
    ],style={'padding-top': "1px","background": "#F2F2F2"}),
    

    # part2 time series and machine learning prediction
    html.Div([
        html.Div([html.H2('Part 2: Time Series and Machine Learning Prediction')],
                 style = {'textAlign': 'center','padding-top': "25px"})
        ]),
    # Garch model result and LSTM model result
    # title
    html.Div([
        html.Div([html.H3('time series prediction result')],
                 style = {'width':'52%','display':'inline-block','margin-top':5}),
        html.Div([html.H3('machine learning prediction result')],
                 style = {'width':'48%','display':'inline-block','margin-top':5})
        ]),
    # image and figure
    html.Div([
        html.Div([html.Img(src=image0,style={"height":"95%","width":"95%","padding-left":"5%"})],
                 style={"float": "left", "width": "45%"}),
        html.Div([dcc.Graph(id='stock-lstm',figure = line_fig,
                            style={"height":"90%","width":"90%","padding-left":"5%"})],
                 style={"float": "right", "width": "52%"}),
    ],style={'padding-top': "20px","padding-bottom":"10px","background": "#F2F2F2"}),
    
    html.Div(style=dict(clear="both")),
    
    # LSTM model result with SDG as risk factors
    # title
    html.Div([
        html.Div([html.H3('lstm model train & validation loss')],
                 style = {'width':'52%','display':'inline-block','margin-top':20}),
        html.Div([html.H3('machine learning prediction result ')],
                 style = {'width':'48%','display':'inline-block','margin-top':20})
        ]),
    # image and figure
    html.Div([
        html.Div([html.Img(src=image0s,style={"height":"90%","width":"80%","padding-left":"10%"})],
                 style={"float": "left", "width": "45%"}),
        html.Div([dcc.Graph(id='stock-lstm-sdg',figure = line_figs,
                            style={"height":"90%","width":"90%","padding-left":"5%"})],
                 style={"float": "right", "width": "52%"}),
    ],style={'padding-top': "20px","padding-bottom":"10px","background": "#F2F2F2"}),
    
    html.Div(style=dict(clear="both")),
    html.Div([html.Div([html.H3('')],style = {'width':'52%','display':'inline-block'})]),
    
])


#---------------------------------------- tab1 callback function ----------------------------------------
@app.callback(
    Output('datatable-paging', 'data'),
    [Input('datatable-paging', "page_current"),
     Input('datatable-paging', "page_size")])
def update_1(page_current,page_size):
    return df_fundinfo.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
    


    
    

#--------------------------------------------- tab2 layout ---------------------------------------------
tab2_layout = html.Div([ # whole page
    
    html.Div([
        html.Div([
            html.Div([html.H3('Select sentiment score type')],
                     style = {'width':'52%','display':'inline-block','margin-top':10})
        ]),
        
        html.Div([
            html.Div([
                dcc.Dropdown(id='ticker',options = [{'label':i,'value':i} for i in SDG_choice],value = 'SDG')
            ],style = {'width':'48%','display':'inline-block'}),
            html.Div([
                html.Button(id = 'submit_button',n_clicks = 1,children = 'Submit',
                            style = {'height':'40px','margin-left':80})
            ],style = {'width':'48%','display':'inline-block','float':'right'})
        ])
    ],style={'padding-top': "10px","background": "#F2F2F2"}),


    # phase1
    html.Div([
        html.Div([html.H2('Phase 1: COVID-19 Timeline and SDG Scores Change at Key Dates')],
                 style = {'textAlign': 'center','padding-top': "25px"})
        ]),
    
    # COVID-19 timeline table and SDG change table
    # title
    html.Div([
        html.Div([html.H3('COVID-19 key dates and events')],
                 style = {'width':'52%','display':'inline-block','margin-top':5}),
        html.Div([html.H3('SDG scores change at COVID-19 key dates')],
                 style = {'width':'48%','display':'inline-block','margin-top':5})
        ]),
    # tables
    html.Div([
            html.Div([dash_table.DataTable(
                id='covid-timeline-table',
                columns=[{"name": i, "id": i} for i in list(df_covid.columns)],
                page_current=0, page_size=10, page_action='custom', fixed_columns={'headers': True,'data': 1},
                style_table={'height': '360px','overflowY': 'auto','width': '100%','minWidth': '100%'})   
                     ],style={"float": "left", "width": "45%"}),
        
            html.Div([dash_table.DataTable(
                id='SDG-change-table',
                page_current=0, page_size=10, page_action='custom', fixed_columns={'headers': True,'data': 1},
                style_table={'height': '360px','overflowY': 'auto','padding-right':"5%",
                             'width': '100%','minWidth': '100%'})
                     ],style={"float": "right", "width": "50%"}),
        
        ],style={'padding-top': "5px","padding-bottom":"10px","background": "#F2F2F2"}),
    html.Div(style=dict(clear="both")),
    
    # highlight SDG change at key dates
    # title
    html.Div([
        html.Div([html.H3('highlight SDG change at COVID-19 key dates')],
                 style = {'width':'52%','display':'inline-block'})
        ]),
    # images
    html.Div([
        html.Div([html.Img(id ='sdg-change-img1',style={'height':'300px','width':'600px','padding-right':'30%'})],
                 style={'width':'30%','display':'inline-block'}),
        html.Div([html.Img(id ='sdg-change-img2',style={'height':'300px','width':'600px','padding-left':'15%','padding-right':'15%'})],
                 style={'width':'30%','display':'inline-block'}),
        html.Div([html.Img(id='sdg-change-img3',style={'height':'300px','width':'600px','padding-left':'30%'})],
                 style={'width':'30%','display':'inline-block'}),
    ],style={'padding-top': "20px","padding-bottom":"10px","background": "#F2F2F2"}),
    html.Div(style=dict(clear="both")),
    
    
    # phase2
    html.Div([
        html.Div([html.H2('Phase 2: Identify Outlier, Rank Stocks and Plot Major Companies')],
                 style = {'textAlign': 'center','padding-top': "25px"})
        ]),
    # identify dates for score outlier & rank score changes at key dates and show top 10 stocks
    # title
    html.Div([
        html.Div([html.H3('Identify dates for SDG score outlier')],
                 style = {'width':'52%','display':'inline-block','margin-top':5}),
        html.Div([html.H3('Top 10 stocks ranked by SDG score change at COVID-19 key dates')],
                 style = {'width':'48%','display':'inline-block','margin-top':5})
        ]),
    # tables
    html.Div([
            html.Div([dash_table.DataTable(
                id='SDG-outlier-table',
                page_current=0, page_size=16, page_action='custom', fixed_rows={'headers': True,'data': 0},
                style_table={'height': '400px','overflowY': 'auto','overflowX': 'scroll','width': '100%','minWidth': '100%'})  
                     ],style={"float": "left", "width": "45%"}),
        
            html.Div([dash_table.DataTable(
                id='top10-stock-table',
                page_current=0, page_size=16, page_action='custom', fixed_rows={'headers': True,'data': 0},
                style_table={'height': '400px','overflowY': 'auto','padding-right':"5%",
                             'width': '100%','minWidth': '100%'})
                     ],style={"float": "right", "width": "50%"}),
        
        ],style={'padding-top': "5px","padding-bottom":"10px","background": "#F2F2F2"}),
    html.Div(style=dict(clear="both")),
    
    
    # show graphs of major companies per sector
    # title
    html.Div([
        html.Div([html.H3('show graphs of major companies per sector')],
                 style = {'width':'52%','display':'inline-block'})
        ]),
    # figure
    html.Div([
        html.Div([dcc.Graph(id='major-company-fig')])
    ],style={'padding-top': "20px","background": "#F2F2F2"}),

    html.Div([html.Span(id="vir_span2", style={"display": "none"})
             ],style={'padding-top': "20px","background": "#F2F2F2"})
    
])

#---------------------------------------- tab2 callback function ----------------------------------------
@app.callback(
    Output('covid-timeline-table', 'data'),
    [Input('covid-timeline-table', "page_current"),
     Input('covid-timeline-table', "page_size")])
def update_2(page_current,page_size):
    return df_covid.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')

@app.callback(
    [Output('SDG-change-table', 'data'),
    Output('SDG-change-table','columns')],
    [Input('SDG-change-table', "page_current"),
     Input('SDG-change-table', "page_size"),
     Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_sdg_chg_table(page_current,page_size,n_clicks,ticker):
    # sentiment.csv does not include per sector data, so if ticker==sentiment, it make no sense
    if ticker == 'SDG':
        data = df_SDG_cv.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_SDG_cv.columns)]
    else:
        data = df_SDG_adj_cv.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_SDG_adj_cv.columns)]
    return data,columns

@app.callback(
    [Output('sdg-change-img1', 'src'),
    Output('sdg-change-img2', 'src'),
    Output('sdg-change-img3', 'src')],
    [Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_sdg_chg_img(n_clicks,ticker):
    # sentiment.csv does not include per sector data, so if ticker==sentiment, it make no sense
    if ticker == 'SDG':
        return by_sector, by_SDG, by_overall
    else:
        return by_sector_adj, by_SDG_adj, by_overall_adj


@app.callback(
    [Output('SDG-outlier-table', 'data'),
     Output('SDG-outlier-table', 'columns')],
    [Input('SDG-outlier-table', "page_current"),
     Input('SDG-outlier-table', "page_size"),
     Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_outlier_table(page_current,page_size,n_clicks,ticker):
    if ticker == 'SDG':
        data = df_outlier.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_outlier.columns)]
    elif ticker == 'SDG_adj':
        data = df_adj_outlier.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_adj_outlier.columns)]
    else:
        data = df_stm_outlier.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_stm_outlier.columns)]
    return data, columns

@app.callback(
    [Output('top10-stock-table', 'data'),
     Output('top10-stock-table','columns')],
    [Input('top10-stock-table', "page_current"),
     Input('top10-stock-table', "page_size"),
     Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_top10_table(page_current,page_size,n_clicks,ticker):
    if ticker == 'SDG':
        data = df_SDG_rank.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_SDG_rank.columns)]
    elif ticker == 'SDG_adj':
        data = df_SDG_adj_rank.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_SDG_adj_rank.columns)]
    else:
        data = df_stm_rank.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')
        columns = [{"name": i, "id": i} for i in list(df_stm_rank.columns)]
    return data, columns

@app.callback(
    Output('major-company-fig','figure'),
    [Input('submit_button','n_clicks')],
    [State('ticker','value')])
def update_maj_company_fig(n_clicks,ticker):
    if ticker == 'SDG':
        return sdgmaj_fig
    elif ticker == 'SDG_adj':
        return sdgadjmaj_fig
    else:
        return stmmaj_fig


if __name__ == '__main__':
    app.run_server(port=9583, host='127.0.0.1', debug=False)
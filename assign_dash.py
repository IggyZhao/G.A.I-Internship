import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import date
from datetime import datetime as dt

import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import dns
import json
import plotly
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import plot
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")

#####-------------------------------------------------- Data part --------------------------------------------------#####

# import data
tickers = ['SPY','AAPL','FB','NFLX','MSFT']
start_date = '2015-05-01'
end_date = date.today()
stocks = yf.download(tickers,start_date,end_date)['Adj Close']

# calculate descriptive statistics
def cal_stat1(stocks):
    tickers = stocks.columns
    stocks_stat = stocks.copy()
    for i in tickers:
        stocks_stat[i+'_returns'] = stocks[i].pct_change()
        stocks_stat[i+'_std'] = stocks[i].std()
        stocks_stat[i+'_momentum_10d'] = stocks[i] - stocks[i].shift(10)
        stocks_stat[i+'_differences'] = stocks[i].diff()
        stocks_stat[i+'_MA_10d'] = stocks[i].rolling(10).mean()
    return stocks_stat
stock_stat1 = cal_stat1(stocks)

# data used in dash
stock_stat = stock_stat1.dropna()
desp_stats = ['returns','momentum_10d','differences','MA_10d']
color_gauge = ['coral','khaki','lightgoldenrodyellow','lightsalmon','lightpink']
colors = ['aqua', 'aquamarine','lightseagreen','mediumorchid','orangered']
all_date = pd.DatetimeIndex(stock_stat.index)
start_d, end_d = all_date[0].date(),all_date[-1].date()
start_y, end_y = str(start_d)[0:4], str(end_d)[0:4]
len_of_years = int(end_y) - int(start_y) + 1
mark = dict(
    zip([x for x in np.arange(1,len_of_years+1,1).tolist()], [y for y in np.arange(int(start_y), int(end_y)+1).tolist()]))




#####-------------------------------------------------- App part --------------------------------------------------#####

app = dash.Dash()

#----------------------------------------------- main page layout ---------------------------------------------
app.layout = html.Div([
    html.Span(id="vir_span_view", style={"display": "none"}, children=0),
    html.H1(
        'Stock Analysis',
        style={
            'textAlign': 'center',
            'color': '#333',
            'font-size': '30px',
            'height': '20px',
            'line-height': '30px',
            'padding-top': "20px"
        }
    ),
    html.Div([
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Price & Return Analysis', value='tab-1',
                    style={"height": 60,'font-size': '20px'},
                    selected_style={"height": 60,'font-size': '20px'}),
            dcc.Tab(label='Descriptive Statistics Analysis', value='tab-2',
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

tab1_layout = html.Div([  # whole page: upper part, lower part
    
    html.Div([  # upper part: dropdown, date_picker_range, button
        html.Div([
            html.Div([html.H3('Select stock tickers')],
                     style = {'width':'52%','display':'inline-block','margin-top':10}),
            html.Div([html.H3('Select date ranges')],
                     style = {'width':'48%','display':'inline-block','margin-top':10})
        ]),
        
        html.Div([
            html.Div([
                dcc.Dropdown(
                id='ticker',
                options = [{'label':i,'value':i} for i in tickers],
                value = tickers,
                multi = True
                )
            ],style = {'width':'48%','display':'inline-block'}),
            html.Div([
                dcc.DatePickerRange(
                id='date_range',
                min_date_allowed = start_d,
                max_date_allowed = end_d,
                start_date = start_d,
                end_date = end_d,
                style = {'height':'40px'}
                ),

                html.Button(
                id = 'submit_button',
                n_clicks = 1,
                children = 'Submit',
                style = {'height':'40px','margin-left':80}
                )
            ],style = {'width':'48%','display':'inline-block','float':'right'})
        ])
    ],style={'padding-top': "10px","background": "#F2F2F2"}),
    
    
    html.Div([ # lower part: line chart, displot, heatmap, speedometer
        # time series line chart
        html.Div([
            html.Div([dcc.Graph(id='stock-ts-line',style={'margin-top':'20px'})])
        ],style={'padding-top': "10px","background": "#F2F2F2"}),

        # histogram-displot and heatmap
        html.Div([
            html.Div([dcc.Graph(id='stock-histogram')],style={"float": "left", "width": "59%"}),
            html.Div([dcc.Graph(id='stock-heatmap')],style={"float": "right", "width": "40%"}),
        ],style={'padding-top': "20px","background": "#F2F2F2"}),

        html.Div(style=dict(clear="both")),

        # speedometer figure
        html.Div([
            html.Div([
                html.Div(children='Stock Price Speedometer Figure',
                               style={'textAlign':'center','padding-bottom':'40px',
                                      'fontSize':'21px','fontFamily':'Old Standard TT'}
                        ),
                dcc.Graph(id='stock-speedometer')
            ],style={'padding-top':'100px','padding-bottom':'100px',
                     'padding-left':'80px','padding-right':'80px','background':'white'}
            )
        ],style={'padding-top': "20px","background": "#F2F2F2"}),

        html.Div([html.Span(id="vir_span1", style={"display": "none"})
                 ],style={'padding-top': "20px","background": "#F2F2F2"})
    ])
    
])


#------------------------------------------- tab1 callback function -------------------------------------------

@app.callback([Output('stock-ts-line','figure'),Output('stock-histogram','figure'),
               Output('stock-heatmap','figure'),Output('stock-speedometer','figure')],
              [Input('submit_button','n_clicks')],
              [State('ticker','value'),State('date_range','start_date'),State('date_range','end_date')])
def update_graph(n_clicks, ticker, start_date, end_date):
    # time series line chart
    traces = []
    for i in ticker:
        traces.append(
        go.Scatter(
        x = stock_stat.loc[str(start_date):str(end_date),i+'_returns'].index,
        y =  stock_stat.loc[str(start_date):str(end_date),i+'_returns'].values,
        name = i+'_returns',
        mode='lines')
        )
    line_fig = {'data':traces,
                'layout':go.Layout(xaxis={'title':'stocks'},yaxis={'title':'returns'},
                                   title = 'Returns of Selected Stocks', hovermode='closest')
               }
    
    # histogram-displot
    hist_data = []
    group_labels=[]
    for i in ticker:
        hist_data.append(stock_stat.loc[str(start_date):str(end_date),i])
        group_labels.append(i)
    histogram_fig = ff.create_distplot(hist_data, group_labels)
    histogram_fig.update_layout(title='Price Distribution of Selected Stocks',title_x=0.5)
    
    # heatmap
    heatmap_fig = {
        'data':[go.Heatmap(x=ticker,
                           y=ticker,
                           z=stock_stat.loc[str(start_date):str(end_date),ticker].corr(),
                           colorscale='spectral',
                           opacity=0.6,
                          zmin = 0.5, zmax=1)],
        'layout':go.Layout(title='Correlation Matrix of Stock Prices of Selected Stocks')   
    }
    
    # speedometer
    speed_traces = []
    length = len(ticker)
    n = 1/length
    for i in range(length):
        max_value = max(stock_stat.loc[str(start_date):str(end_date),ticker[i]])
        cur_value = stock_stat.loc[str(start_date):str(end_date),ticker[i]][-1]
        speed_traces.append(
            go.Indicator(
                value = cur_value,
                domain = {'x': [i*n, (i+1)*n-0.05], 'y': [0, 1]},
                mode = "gauge+number+delta",
                title = {'text': ticker[i], 'font': {'size': 18}},
                delta = {'reference': 0.8*max_value, 'relative': True,
                         'increasing': {'color': "green"},'decreasing': {'color': "red"}},
                gauge = {
                    'axis':{'range':[None,max_value],
                           'tickwidth': 1, 'tickcolor': 'darkblue'},
                    'bar': {'color': color_gauge[i],'line':{'color':'purple'}},
                    'bgcolor':'purple',
                    'borderwidth':2,
                    'bordercolor':'gray',
                    'steps': [{'range': [0, 0.5*max_value], 'color': 'lightskyblue'},
                              {'range': [0.5*max_value, 0.75*max_value], 'color': 'deepskyblue'},
                             {'range': [0.75*max_value, max_value], 'color': 'dodgerblue'}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75,'value': cur_value}
                }
            )
        )
    speedometer_fig = go.Figure(data = speed_traces)
    speedometer_fig['layout'].update(paper_bgcolor='aliceblue',
                                     title='current stock price vs historical prices range', 
                                     title_x=0.9,font={'size':12})
    
    
    return line_fig, histogram_fig, heatmap_fig, speedometer_fig





#----------------------------------------------- tab2 layout -----------------------------------------------

tab2_layout = html.Div([ # whole page: upper part, middle part, lower part
    
    html.Div([ # upper part: year slider
        html.Div([
            html.P("Select year range:"),
        ], style={"float": "left","width": "10%","margin-top": "17px",'margin-left':'5%'}),
        
        html.Div([
            dcc.Slider(id='year_slider',min=1,max=len_of_years,value=5,
                       marks=mark,step=1)],
            style={"height": "50px", "background": "#fff","border": "1px solid #ccc","width": "82%", 
                   "float": "right","padding": "20px 0 0px 0"}
        )
    ],style={"background": "#fff", "width":"99.9%", "height": "72px","border": "1px solid #ccc",'margin-top': '10px'}),
    html.Div(style=dict(clear="both")),
    
    
    html.Div([ # middle part: dropdown 1, dropdown 2, button
        html.Div([
            html.Div([html.H3('Select stock tickers')],
                     style = {'width':'52%','display':'inline-block','margin-top':10}),
            html.Div([html.H3('Select statistics')],
                     style = {'width':'48%','display':'inline-block','margin-top':10})
        ]),
        
        html.Div([
            html.Div([
                dcc.Dropdown(
                id='ticker2',
                options = [{'label':i,'value':i} for i in tickers],
                value = tickers,
                multi = True
                )
            ],style = {'width':'48%','display':'inline-block'}),

            html.Div([
                dcc.Dropdown(
                id='select_stat',
                options = [{'label':i,'value':i} for i in desp_stats],
                value = 'MA_10d',
                style = {'width': '60%','display':'inline-block','height':'40px'}
                ),

                html.Button(
                id = 'submit_button2',
                n_clicks = 1,
                children = 'Submit',
                style = {'height':'40px','margin-left':80}
                )
            ],style = {'width':'48%','display':'inline-block','float':'right'})
        ])
    ],style={'padding-top': "10px","background": "#F2F2F2"}),
    
    
    html.Div([ # lower part: polar/radar chart, box chart, horizontal bar chart, scatterplot
        # polar chart
        html.Div([
            html.Div([dcc.Graph(id='polar-chart',style={'margin-top':'20px'})])
        ],style={'padding-top': "10px","background": "#F2F2F2"}),
        

        # box and horizontal bar chart
        html.Div([
            html.Div([dcc.Graph(id='box-chart')],
                     style={"width": "55%", "float": "left", "padding": "20px 1%",'height':'420px',
                           'background':'#fff'}),
            html.Div([dcc.Graph(id='horizontal-bar')],
                     style={"float": "right", "width": "38%","padding": "0px 1%",'height':'460px','background':'#fff'}),
        ],style={'padding-top': "20px","background": "#F2F2F2"}),

        html.Div(style=dict(clear="both")),

        # scatterplot
        html.Div([
            html.Div([dcc.Graph(id='stock-scatterplot')])
        ],style={'padding-top': "20px","background": "#F2F2F2"}),

        html.Div([html.Span(id="vir_span2", style={"display": "none"})
                 ],style={'padding-top': "20px","background": "#F2F2F2"})
    ])
    
])


#---------------------------------------- tab2 callback function ----------------------------------------

@app.callback([Output('polar-chart','figure'),Output('box-chart','figure'),
               Output('horizontal-bar','figure'),Output('stock-scatterplot','figure')],
              [Input('submit_button2','n_clicks')],
              [State('ticker2','value'),State('select_stat','value'),State('year_slider','value')])
def update_graph2(n_clicks, ticker, stat, select_year):
    start_y = '2015'
    
    # polar chart
    scatterpolar_fig = make_subplots(rows=1, cols=len(ticker),subplot_titles=ticker,
                                     specs=[[{'type': 'polar'}]*len(ticker)]*1 )
    for i in range(len(ticker)):
        scatterpolar_fig.add_trace(
            go.Scatterpolar(
                name = ticker[i],
                r=[
                    stock_stat.loc[start_y:str(mark[select_year]+1),ticker[i]+'_returns'].mean()*1000,
                    stock_stat.loc[start_y:str(mark[select_year]+1),ticker[i]+'_std'].mean()/100,
                    stock_stat.loc[start_y:str(mark[select_year]+1),ticker[i]+'_momentum_10d'].mean(),
                    stock_stat.loc[start_y:str(mark[select_year]+1),ticker[i]+'_differences'].mean()*10,
                    stock_stat.loc[start_y:str(mark[select_year]+1),ticker[i]+'_MA_10d'].mean()/100,
                    stock_stat.loc[start_y:str(mark[select_year]+1),ticker[i]+'_returns'].mean()*1000
                ],
                theta = ['returns','std','momentum_10d','differences','MA_10d','returns'],
            ),1,i+1
        )
    scatterpolar_fig.update_traces(fill='toself'),
    scatterpolar_fig['layout'].update(title='descriptive statistics polar chart (with standardization)',title_x=0.5)
    
    # box plot
    box_data = []
    for i in ticker:
        box_data.append(
            go.Box(
                y = stock_stat.loc[start_y:str(mark[select_year]+1),i+'_'+stat],
                name = i,
                boxmean=True,
                whiskerwidth=0.2,
                marker_size=2,
                line_width=1,
                width=0.4
            )
        )
    box_layout = go.Layout(
        title = 'box chart for '+stat, title_x=0.5,autosize=False,margin={'l': 40, 'b': 20, 't': 40, 'r': 10},
        xaxis={"showticklabels": True,'showgrid':False}, yaxis={"showticklabels": True,'showgrid':False},
        height=400, plot_bgcolor = '#fff'
    )
    box_fig = go.Figure(data = box_data, layout = box_layout)
    
    # horizontal bar chart
    colors = ['aqua', 'aquamarine','lightseagreen','mediumorchid','orangered']
    bar_x_data = []
    for i in ticker:
        bar_x_data.append(stock_stat.loc[start_y:str(mark[select_year]+1),i+'_'+stat].mean())
    bar_data = go.Bar(x=bar_x_data, y=ticker, orientation='h',marker_color=colors[0:len(ticker)],opacity=0.6)
    bar_layout = go.Layout(title = 'horizontal bar chart for '+stat, title_x=0.5,plot_bgcolor = '#fff')
    bar_fig = go.Figure(data=bar_data,layout = bar_layout)
    
    # scatterplot
    scatter_fig = tools.make_subplots(rows=1, cols=len(ticker),subplot_titles=ticker)
    for i in range(len(ticker)):
        scatter_fig.append_trace(
            go.Scatter(
                x=stock_stat.loc[start_y:str(mark[select_year]+1),ticker[i]],
                y=stock_stat.loc[start_y:str(mark[select_year]+1),ticker[i]+'_'+stat],
                mode='markers',name=ticker[i]
            ),1,i+1
        )
    scatter_fig['layout'].update(title='stock price(x) v.s. '+stat+' (y)',title_x=0.5)
    
    
    return scatterpolar_fig, box_fig, bar_fig, scatter_fig



# run app on website
if __name__ == '__main__':
    app.run_server(port=9583, host='127.0.0.1',debug=False)
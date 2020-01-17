#Basics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

#Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
#Plotly
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import plotly as py

import os
import csv

import gunicorn

from ipywidgets import interact

df = pd.read_csv('data2.csv')
print(df.head())
df3=pd.read_excel('data3.xlsx')

#World
fig1 = go.Figure(go.Scattermapbox(mode = "markers+lines",lon = [-51, -8, 45, 90],lat = [-14, 39, 40, 23],marker = {'size': 10}))
fig1.update_layout( margin ={'l':0,'t':0,'b':0,'r':0},mapbox = { 'center': {'lon': 13, 'lat': 10},'style': "stamen-terrain",'center': {'lon': -20, 'lat': -20},'zoom': 0.9})
fig1.show()

#Population grouth
#plt.figure(figsize = [10,10])
fig2=px.line(df, x="year", y="Pop_Growth",color="country",template='simple_white',width=800, height=400,
labels = {'Pop_Growth': 'Population Growth (%)'}#,animation_group='country',
         #    ,title = 'Population Growth in four country between 2005-2016')
             )
fig2.update_layout(
  template='none',
  margin=dict(r=10, t=25, b=40, l=60),title_font_size=24
)
fig2.show()
#GDP per capita
fig3=px.bar(df3, x='Year', y='GDP', color='Country',#title = 'GDP per capita (current US$) in four country between 2005-2016',
                  template="none",width=800, height=400,
            labels = {'GDP': 'GDP per capita (current US$)'})



#plt.figure(figsize=(6,4))
#plt.xlim(-12, 12)
#bar_plot = sns.barplot(x="portugal3",y="years", color='#9933ff', data = Pyramid_Data2,label="Portugal",orient='h',hue_order=True)
#bar_plot = sns.barplot(x="armenia3",y="years", color='#3abfbf',data = Pyramid_Data2,label="Armenia",orient='h',hue_order=True  )
#plt.legend(loc=4, borderaxespad=0.,prop={'size':10})
#plt.ylabel('Years', fontsize=12)
#plt.xlabel('Population - Millions', fontsize=12)
#plt.title('Armenia and Portugal population betweem the years 2005-2016')

#Alcohol
fig7 = px.scatter(df, x="year", y="Drug_Alc", color="country",size="Drug_Alc",width=800, height=400,
                 labels = {'Drug_Alc': 'Alcohol and Drug addiction (%)'},
                  #title = 'Alcohol and Drug addiction (%) in four country between 2005-2016 ',
                  template='simple_white'
                  )

#fig8 = px.area(df, x="year", y="Drug_Alc", color="country",size="Drug_Alc",width=800, height=400,
                 #labels = {'Drug_Alc': 'Alcohol and Drug addiction (%)'},
                  #title = 'Alcohol and Drug addiction (%) in four country between 2005-2016 ',
                  #template='simple_white'
                  #)
sns.set_style("white")
x=['2005', '2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
Armenia=[4.352,4.382,5.064,5.559,4.360,4.217,4.917,5.694,5.496,5.529,4.957,5.340]
Bangladesch=[39.478,43.541,44.381,49.511,53.736,59.937,63.413,67.505,69.709,73.189,56.440,72.886]
Brazil=[347.308,347.668,363.212,387.631,367.147,419.754,439.412,470.028,503.677,529.808,417.564,514.565]
Portugal=[65.279,59.816,60.149,55.624,54.176,48.136,47.623,46.013,45.426,45.052,52.729,50.839]
# Basic stacked area chart.
#fig7, ax = plt.subplots()
#ax=plt.stackplot(x,Armenia, Portugal, Bangladesch,Brazil, labels=['Armenia','Portugal','Bangladesch','Brazil'])
#fig = px.area(df, x="year", y="pop", color="continent",
	      #line_group="country")
#ax.legend(loc='upper left')
#plt.show()


fig11=px.bar(df, x='year', y='Educ_Tert', color='country',#title = 'Pop. >25 years wiht Education at least Bsc. in four country between the years 2005-2016 (%)',
           barmode="group",
            template='simple_white'
            ,width=800, height=400,  facet_col='country',
            labels = {'Educ_Tert': '% Population > 25 years at least Bsc.'})

fig12 = px.scatter_geo(df, locations="country_code", color="country",width=800, height=400,
                     hover_name="country", size="Depression",
                     projection="natural earth", animation_frame="year")
#layout3 = [dict]

title2 = 'Fertility Rate'
labels2 = ['ARMENIA', 'BANGLADESH', 'BRAZIL', 'PORTUGAL']
colors2 = ['rgb(0,0,255)', 'rgb(255,165,0)', 'rgb(0,128,0)', 'rgb(106,13,173)']
mode_size = [6, 6, 6, 6]
line_size = [2, 2, 2, 2]
x_data = np.vstack((np.arange(2005, 2016),)*4)
y_data = np.array([
    [1.40,1.30,1.40,1.40,1.60,1.55,1.50,1.73,1.71,1.69,1.66,1.63],
    [2.61,2.52,2.44,2.38,2.32,2.28,2.24,2.21,2.18,2.16,2.13,2.10],
    [1.98,1.93,1.88,1.85,1.82,1.81,1.79,1.78,1.77,1.75,1.74,1.73],
    [1.42,1.40,1.38,1.36,1.34,1.33,1.31,1.29,1.28,1.26,1.25,1.25],
])
fig6 = go.Figure()
for i in range(0, 4):
    fig6.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',name=labels2[i],line=dict(color=colors2[i], width=line_size[i]),connectgaps=True))
 # endpoints
    fig6.add_trace(go.Scatter(x=[x_data[i][0], x_data[i][-1]],y=[y_data[i][0], y_data[i][-1]],mode='markers',marker=dict(color=colors2[i], size=mode_size[i])))
fig6.update_layout(xaxis=dict(showline=True, showgrid=False,showticklabels=True,linecolor='rgb(204, 204, 204)',linewidth=2,
ticks='outside',tickfont=dict(family='Arial',size=8,color='rgb(82, 82, 82)',),),yaxis=dict( showgrid=False,zeroline=False, showline=False,showticklabels=False,
 ), autosize=False,margin=dict( autoexpand=False,l=100,r=20, t=110,), showlegend=False, plot_bgcolor='white'
)
annotations = []

# Adding labels
for y_trace, label, color in zip(y_data, labels2, colors2):
    # labeling the left_side of the plot
    annotations.append(dict(xref='paper', x=0.05, y=y_trace[1],
                                  xanchor='right', yanchor='middle',
                                  text=label + ' {}%'.format(y_trace[1]),
                                  font=dict(family='Arial',
                                            size=10),
                                  showarrow=False))
    # labeling the right_side of the plot
    annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],
                                  xanchor='left', yanchor='middle',
                                  text='{}%'.format(y_trace[11]),
                                  font=dict(family='Arial',
                                            size=10),
                                  showarrow=False))
# Title
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              #text='Fertility Rate in four country between 2005-2016',
                              font=dict(family='Arial',
                                        size=15,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
# Source
annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                              xanchor='center', yanchor='top',
                              text='Source: World Bank ',
                              font=dict(family='Arial',
                                        size=15,
                                        color='rgb(150,150,150)'),
                              showarrow=False))

fig6.update_layout(annotations=annotations)



labels = ['ARMENIA', 'BANGLADESH', 'BRAZIL', 'PORTUGAL']
colors = ['rgb(0,0,255)', 'rgb(255,165,0)', 'rgb(0,128,0)', 'rgb(106,13,173)']

mode_size = [6, 6, 6, 6]
line_size = [2, 2, 2, 2]

x_data = np.vstack((np.arange(2005, 2016),)*4)

y_data = np.array([
    [1.46,1.48,1.72,1.911,1.50,1.465,1.70,1.97,1.89,1.89,1.914,1.72],
    [0.28,0.308,0.31,0.343,0.36,0.40,0.424,0.447,0.456,0.47,0.50,0.39],
    [1.86,1.84,1.91,2.01,1.89,2.14,2.22,2.35,2.50,2.61,2.69,2.18],
    [6.21,5.68,5.70,5.26,5.12,4.55,4.51,4.37,4.34,4.33,4.04,4.92],
])
fig10 = go.Figure()

for i in range(0, 4):
    fig10.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
        name=labels[i],
        line=dict(color=colors[i], width=line_size[i]),
        connectgaps=True,
    ))

    # endpoints
    fig10.add_trace(go.Scatter(
        x=[x_data[i][0], x_data[i][-1]],
        y=[y_data[i][0], y_data[i][-1]],
        mode='markers',
        marker=dict(color=colors[i], size=mode_size[i])
    ))

fig10.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
    ),
    autosize=False,
    margin=dict(
        autoexpand=False,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=False,
    plot_bgcolor='white'
)

annotations = []

# Adding labels
for y_trace, label, color in zip(y_data, labels, colors):
    # labeling the left_side of the plot
    annotations.append(dict(xref='paper', x=0.05, y=y_trace[1],
                                  xanchor='right', yanchor='middle',
                                  text=label + ' {}%'.format(y_trace[1]),
                                  font=dict(family='Arial',
                                            size=10),
                                  showarrow=False))
    # labeling the right_side of the plot
    annotations.append(dict(xref='paper', x=0.95, y=y_trace[5],
                                  xanchor='left', yanchor='middle',
                                  text='{}%'.format(y_trace[11]),
                                  font=dict(family='Arial',
                                            size=10),
                                  showarrow=False))
# Title
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='CO2 emissions (metric tons per capita) four country between 2005-2016',
                              font=dict(family='Arial',
                                        size=15,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
# Source
annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                              xanchor='center', yanchor='top',
                              text='Source: World Bank ',
                              font=dict(family='Arial',
                                        size=15,
                                        color='rgb(150,150,150)'),
                              showarrow=False))

fig10.update_layout(annotations=annotations)




app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,'assets/style.css'])

server = app.server

colors = {
    'background': '#fdf4ca',
    'background2': '#FF0',
    'text': 'gray'
    }

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
html.H1(
children='Visualizing Environmental, Economic, Educational and Depression indicators of Armenia, Bangladesh, Brazil & Portugal',
 style={'textAlign': 'center', 'color': colors['text']} ),



html.Div(className='container', children=[
    html.Img(src=app.get_asset_url('flags.png'))], style={'textAlign': 'center'}),
    html.Div([
        html.H4('Group Members:'),
        html.Div([
            html.P('Ali Sabbir - M20190580'),
            html.P('Lilit Tonoyan – M20190930'),
           html.P('Eliane Maria Zanlorense – M20190802'),
            html.P('Fernando Afonso Ribeiro – M20190815'),

        ])

    ],style={'text-align' : 'center'}),

    html.H2('Map visualization', style={'text-align': 'center'}),

    html.Div(dbc.Container(dcc.Graph(figure=fig1),style={'height': '70vh', "width" : "100%"}),),

    html.H3('Population Growth (Annual %) between 2005-2016', style={'text-align': 'center'}),
    html.Div(dbc.Container(dcc.Graph(figure=fig2),style={'height': '70vh', "width" : "100%"}),),

    html.H4('GDP per capita (current US$) between 2005-2016', style={'text-align': 'center'}),
    html.Div(dbc.Container(dcc.Graph(figure=fig3),style={'height': '70vh', "width" : "100%"}),),

    #html.Div(children = ['Education'],style={'text-align':'center'}),
    #html.Div(dcc.Graph(id = 'Graph2',figure = fig6),className = 'container',),
    html.H5('Fertility Rate (%) between 2005-2016', style={'text-align': 'center'}),
    html.Div(dbc.Container(dcc.Graph(figure=fig6),style={'height': '70vh', "width" : "100%"}),),
    html.H6('Alcohol and Drug addiction (%) between 2005-2016', style={'text-align': 'left'}),
    html.Div(dbc.Container(dcc.Graph(figure=fig7),style={'height': '70vh', "width" : "100%"}),),

    html.Div(dbc.Container(dcc.Graph(figure=fig10),style={'height': '70vh', "width" : "100%"}),),
    #html.H0('Pop. >25 years with Education at least Bsc. between 2005-2016 (%)', style={'text-align': 'center'}),
    html.Div(dbc.Container(dcc.Graph(figure=fig11),style={'height': '70vh', "width" : "100%"}),),
    html.Div(dbc.Container(dcc.Graph(figure=fig12))),
    #html.Div(children=['GDP']),

    html.Div(

        #dbc.Container(
          #  dcc.Graph(figure = dict(data=[go.Scattermapbox(lat=[-14, 39, 40, 23], #title = 'Map of the countries',
        #lon = [-51, -8, 45, 90], mode ='markers+lines', marker = {'size': 10})],layout=go.Layout(autosize=True, hovermode='closest', mapbox = {
        #'center': {'lon': 13, 'lat': 10},
        #'style': "stamen-terrain",
        #'center': {'lon': -20, 'lat': -20},
        #'zoom': 0.9})))
        #)
    ),




])

# @app.callback(
#     [
#         Output('graph3','figure')
#     ],
#     [
#         Input('slider','value')
#     ],
# )
# def plot(years):
#     df2 = df.loc[df['year'].isin(years)]
#     data_ag= []
#     for year in years:
#         data_ag.append(dict(type = 'bar',
#                             x = df2['year'],
#                             y = df2['GDP'],
#                             name = df2['country'],
#                             mode = 'markers')
#
#         )
#     return go.Figure(data=data_ag)


if __name__ == '__main__':
    app.run_server(debug=True)

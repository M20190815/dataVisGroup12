import dash
import gunicorn
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import os
import numpy as np
import plotly.express as px
import plotly
import plotly.io as pio

df = pd.read_csv('data2.csv')
# x=['2005', '2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
# Armenia=[4.352,4.382,5.064,5.559,4.360,4.217,4.917,5.694,5.496,5.529,4.957,5.340]
# Bangladesch=[39.478,43.541,44.381,49.511,53.736,59.937,63.413,67.505,69.709,73.189,56.440,72.886]
# Brazil=[347.308,347.668,363.212,387.631,367.147,419.754,439.412,470.028,503.677,529.808,417.564,514.565]
# Portugal=[65.279,59.816,60.149,55.624,54.176,48.136,47.623,46.013,45.426,45.052,52.729,50.839]
#
# data =dict(x=x,y=[Armenia,Bangladesch,Brazil,Portugal],
#             mode='lines',
#             stackgroup='one',
#             line=dict(
#                 width=1
#             ))
# fig11 = dict(data=data)
#
# plotly.offline.plot(fig11)

fig7 = px.scatter(df, x="year", y="Drug_Alc", color="country",size="Drug_Alc",
                 labels = {'Drug_Alc': 'Alcohol and Drug addiction (%)'},
                  title = 'Alcohol and Drug addiction (%) in four country between 2005-2016 ',
                  #template='simple_white'
                  )

fig6=px.bar(df, x='year', y='Educ_Tert', color='country',title = 'Pop. >25 years wiht Education at least Bsc. in four country between the years 2005-2016 (%)',
           barmode="group",
            #template='simple_white',
            height=400,  facet_col='country',
            labels = {'Educ_Tert': '% Population > 25 years at least Bsc.'})
df3=pd.read_excel('data3.xlsx')
fig5=px.bar(df3, x='Year', y='GDP', color='Country',title = 'GDP per capita (current US$) in four country between 2005-2016',
                #  template='simple_white',
            labels = {'GDP': 'GDP per capita (current US$)'})

fig3 = px.scatter_geo(df, locations="country_code", color="country",
                     hover_name="country", size="Depression",
                     projection="natural earth")
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

fig2 = go.Figure()

for i in range(0, 4):
    fig2.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
        name=labels2[i],
        line=dict(color=colors2[i], width=line_size[i]),
        connectgaps=True,
    ))

    # endpoints
    fig2.add_trace(go.Scatter(
        x=[x_data[i][0], x_data[i][-1]],
        y=[y_data[i][0], y_data[i][-1]],
        mode='markers',
        marker=dict(color=colors2[i], size=mode_size[i])
    ))

fig2.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=8,
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
                              text='Fertility Rate in four country between 2005-2016',
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

fig2.update_layout(annotations=annotations)
# title = 'Fertility Rate'
# labels = ['ARMENIA', 'BANGLADESH', 'BRAZIL', 'PORTUGAL']
# colors = ['rgb(0,0,255)', 'rgb(255,165,0)', 'rgb(0,128,0)', 'rgb(106,13,173)']
#
# mode_size = [6, 6, 6, 6]
# line_size = [2, 2, 2, 2]
#
# x_data = np.vstack((np.arange(2005, 2016),)*4)
#
# y_data = np.array([
#     [1.40,1.30,1.40,1.40,1.60,1.55,1.50,1.73,1.71,1.69,1.66,1.63],
#     [2.61,2.52,2.44,2.38,2.32,2.28,2.24,2.21,2.18,2.16,2.13,2.10],
#     [1.98,1.93,1.88,1.85,1.82,1.81,1.79,1.78,1.77,1.75,1.74,1.73],
#     [1.42,1.40,1.38,1.36,1.34,1.33,1.31,1.29,1.28,1.26,1.25,1.25],
# ])
#
# fig = go.Figure()
#
# for i in range(0, 4):
#     fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
#         name=labels[i],
#         line=dict(color=colors[i], width=line_size[i]),
#         connectgaps=True,
#     ))
#
#     # endpoints
#     fig.add_trace(go.Scatter(
#         x=[x_data[i][0], x_data[i][-1]],
#         y=[y_data[i][0], y_data[i][-1]],
#         mode='markers',
#         marker=dict(color=colors[i], size=mode_size[i])
#     ))
#
# fig.update_layout(
#     xaxis=dict(
#         showline=True,
#         showgrid=False,
#         showticklabels=True,
#         linecolor='rgb(204, 204, 204)',
#         linewidth=2,
#         ticks='outside',
#         tickfont=dict(
#             family='Arial',
#             size=8,
#             color='rgb(82, 82, 82)',
#         ),
#     ),
#     yaxis=dict(
#         showgrid=False,
#         zeroline=False,
#         showline=False,
#         showticklabels=False,
#     ),
#     autosize=False,
#     margin=dict(
#         autoexpand=False,
#         l=100,
#         r=20,
#         t=110,
#     ),
#     showlegend=False,
#     plot_bgcolor='white'
# )
#
# annotations = []
#
# # Adding labels
# for y_trace, label, color in zip(y_data, labels, colors):
#     # labeling the left_side of the plot
#     annotations.append(dict(xref='paper', x=0.05, y=y_trace[1],
#                                   xanchor='right', yanchor='middle',
#                                   text=label + ' {}%'.format(y_trace[1]),
#                                   font=dict(family='Arial',
#                                             size=10),
#                                   showarrow=False))
#     # labeling the right_side of the plot
#     annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],
#                                   xanchor='left', yanchor='middle',
#                                   text='{}%'.format(y_trace[11]),
#                                   font=dict(family='Arial',
#                                             size=10),
#                                   showarrow=False))
# # Title
# annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
#                               xanchor='left', yanchor='bottom',
#                               text='Fertility Rate in four country between 2005-2016',
#                               font=dict(family='Arial',
#                                         size=15,
#                                         color='rgb(37,37,37)'),
#                               showarrow=False))
# # Source
# annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
#                               xanchor='center', yanchor='top',
#                               text='Source: World Bank ',
#                               font=dict(family='Arial',
#                                         size=15,
#                                         color='rgb(150,150,150)'),
#                               showarrow=False))
#
# fig.update_layout(annotations=annotations)
#
# fig.show()
# py.plot(fig, filename='fertility.html')

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,'assets/style.css'])

server = app.server

colors = {
    'background': 'rgb(255,255,255)',
    'text': '#7FDBFF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Overview',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    # html.Div(children='Birth rate by Country by year', style={
    #     'textAlign': 'center',
    #     'color': colors['text']
    #}),
    #dbc.Container(
        #html.H1('Birth rate',style={'text-align' : 'center'}),
    # dcc.Graph(
    #     className='graph',
    #     id='Graph1',
    #
    #     figure={
    #         'data': [
    #             {'x': df['year'], 'y': df['Birth_rate'][:12], 'type': 'bar', 'name': df['country'].unique()[0]},
    #             {'x': df['year'], 'y': df['Birth_rate'][12:24], 'type': 'bar', 'name': df['country'].unique()[1]},
    #             {'x': df['year'], 'y': df['Birth_rate'][24:36], 'type': 'bar', 'name': df['country'].unique()[2]},
    #             {'x': df['year'], 'y': df['Birth_rate'][36:], 'type': 'bar', 'name': df['country'].unique()[3]},
    #
    #         ],
    #         'layout': {
    #             'plot_bgcolor': colors['background'],
    #             'paper_bgcolor': colors['background'],
    #             'font': {
    #                 'color': colors['text']
    #             }
    #         }
    #     }
    # ),#),
    html.Div(
        dbc.Container(
            dcc.Graph(figure=fig5)
    )),
    html.Div(children = ['Education'],style={'text-align':'center'}),
    html.Div(
        dcc.Graph(
            id = 'Graph2',

            figure = fig6
                # 'data' : [
                #     {'x': df['year'],'y': df['Educ_Tert'][:12], 'type':'line','name': df['country'].unique()[0]},
                #     {'x': df['year'], 'y': df['Educ_Tert'][12:24], 'type': 'line', 'name': df['country'].unique()[1]},
                #     {'x': df['year'], 'y': df['Educ_Tert'][24:36], 'type': 'line', 'name': df['country'].unique()[2]},
                #     {'x': df['year'], 'y': df['Educ_Tert'][36:], 'type': 'line', 'name': df['country'].unique()[3]},
                # ]

        ),
        className = 'container',
    ),
    #html.Div(children=['GDP']),
    html.Div(
        dcc.Graph(id = 'graph3',
            figure = px.bar(df, x='year', y='GDP', color='country',title='Evolution of GDP')),# id = 'graph3'),
        # dcc.Slider(
        #     #id='slider',
        #     min = df['year'].min(),
        #     max = df['year'].max(),
        #     marks = {str(i): '{}'.format(str(i)) for i in [2005,2009, 2013, 2016]},
        #     value = df['year'].min(),
        #     step = 1,
        #            ),
    ),
    # html.Div(
    #     data2=[go.Scattermapbox(lat=[-14, 39, 40, 23],
    #     lon = [-51, -8, 45, 90], mode ='markers', marker = {'size': 10})],
    #
    #     layout2 = go.Layout(autosize=True, hovermode='closest', mapbox = {
    #     'center': {'lon': 13, 'lat': 10},
    #     'style': "stamen-terrain",
    #     'center': {'lon': -20, 'lat': -20},
    #     'zoom': 0.9}),
    #
    #     ),
    html.H1('Map visualization', style={'text-align': 'center'}),
    html.Div(

        dbc.Container(
            dcc.Graph(figure = dict(data=[go.Scattermapbox(lat=[-14, 39, 40, 23], #title = 'Map of the countries',
        lon = [-51, -8, 45, 90], mode ='markers+lines', marker = {'size': 10})],layout=go.Layout(autosize=True, hovermode='closest', mapbox = {
        'center': {'lon': 13, 'lat': 10},
        'style': "stamen-terrain",
        'center': {'lon': -20, 'lat': -20},
        'zoom': 0.9})))
        )
    ),
    html.Div(
dcc.Graph(
            figure = {
                'data' : [
                    {'x': df['year'],'y': df['Educ_Tert'][:12], 'type':'bar','name': df['country'].unique()[0]},
                    {'x': df['year'], 'y': df['Educ_Tert'][12:24], 'type': 'bar', 'name': df['country'].unique()[1]},
                    {'x': df['year'], 'y': df['Educ_Tert'][24:36], 'type': 'bar', 'name': df['country'].unique()[2]},
                    {'x': df['year'], 'y': df['Educ_Tert'][36:], 'type': 'bar', 'name': df['country'].unique()[3]},
                ]
            }
        ),
    ),
    html.Div(
        dbc.Container(
            dcc.Graph(figure = fig3)

            ),
        ),
    html.Div(
        dbc.Container(
            dcc.Graph(figure = fig2)
        )
    ),
    html.Div(
        dbc.Container(
            dcc.Graph(figure = fig7)
        )
    ),
    # html.Div(
    #     dbc.Container(
    #         dcc.Graph(figure = fig11)
    #     )
    # )
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

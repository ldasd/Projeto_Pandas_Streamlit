import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import plotly.express as px
import datetime as dt
import streamlit as st  # versão 0.87
from bokeh.plotting import figure
from plotly.subplots import make_subplots

import glob, os    

def plot_media_movel(data):
    # Média simples de 3 dias
    data['MM_3'] = data.Close.rolling(window=3).mean()

    # Média simples de 9 dias
    data['MM_9'] = data.Close.rolling(window=9).mean()

    # Média simples de 17 dias
    data['MM_17'] = data.Close.rolling(window=17).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.Date,
        y=data.Close,
        name = 'Close',
        line = dict(color = '#330000'),
        opacity = 0.8))

    fig.add_trace(go.Scatter(
        x=data.Date,
        y=data['MM_3'],
        name = "Média Móvel 3 Períodos",
        line = dict(color = '#FF8000'),
        opacity = 0.8))

    fig.add_trace(go.Scatter(
        x=data.Date,
        y=data['MM_9'],
        name = "Média Móvel 9 Períodos",
        line = dict(color = '#B2FF66'),
        opacity = 0.8))

    fig.add_trace(go.Scatter(
        x=data.Date,
        y=data['MM_17'],
        name = "Média Móvel 17 Períodos",
        line = dict(color = '#FF00FF'),
        opacity = 0.8))

    st.plotly_chart(fig)


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Open'], name='Open'))

    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'], name='Close'))

    fig.layout.update(title_text='Abertatura e Fechamento',
                     xaxis_rangeslider_visible=True)

    fig.layout.update(
        xaxis=dict(
                showline=True,
                showticklabels=True,
                linecolor='rgb(0, 0, 0)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Roboto',
                    size=12,
                    color='rgb(82,82,82)'),
            ),
        yaxis=dict(
                showline=True,
                showgrid=False,
                zeroline=True,
                showticklabels=True,
                ticks='outside',
                linecolor='rgb(0, 0, 0)',
                linewidth=2
            ),
        plot_bgcolor='rgb(255,255,255)',
        
    )
    st.plotly_chart(fig)
    
    
def plot_volume_data(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'], name='Close'), secondary_y=False,)

    fig.add_trace(go.Bar(
        x=data['Date'], y=data['Volume'], name='Volume'), secondary_y=True,)

    fig.layout.update(title_text='Volume',
                     xaxis_rangeslider_visible=True)

    fig.layout.update(
        xaxis=dict(
                showline=True,
                showticklabels=True,
                linecolor='rgb(0, 0, 0)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Roboto',
                    size=12,
                    color='rgb(82,82,82)'),
            ),
        yaxis=dict(
                showline=True,
                showgrid=False,
                zeroline=True,
                showticklabels=True,
                ticks='outside',
                linecolor='rgb(0, 0, 0)',
                linewidth=2
            ),
        plot_bgcolor='rgb(255,255,255)',
        
    )
    st.plotly_chart(fig)
    
def plot_candle_data(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
 
    fig.add_trace(go.Candlestick(x=df_cripto['Date'], open=df_cripto['Open'], high=df_cripto['High'], low=df_cripto['Low'], close=df_cripto['Close'], name='Preços') , secondary_y=False,)
   
    fig.add_trace(go.Bar(
        x=data['Date'], y=data['Volume'], name='Volume'), secondary_y=True,)
    fig.update_layout(xaxis_rangeslider_visible = False)
    fig.layout.update(title_text='Evolução de Preços',
                     xaxis_rangeslider_visible=True)

    fig.layout.update(
        xaxis=dict(
                showline=True,
                showticklabels=True,
                linecolor='rgb(0, 0, 0)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Roboto',
                    size=12,
                    color='rgb(82,82,82)'),
            ),
        yaxis=dict(
                showline=True,
                showgrid=False,
                zeroline=True,
                showticklabels=True,
                ticks='outside',
                linecolor='rgb(0, 0, 0)',
                linewidth=2
            ),
        plot_bgcolor='rgb(255,255,255)',
        
    )
    st.plotly_chart(fig)
    
def grafico_candle_2(df_ltop5):
    fig = go.Figure()
    INCREASING_COLOR = '#14B900'
    DECREASING_COLOR = '#FF0000'
    fig.update_layout(
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
        )
    )
    data = [ dict(
        type = 'candlestick',
        open = df_ltop5['Open'],
        high = df_ltop5['High'],
        low = df_ltop5['Low'],
        close = df_ltop5['Close'],
        x = df_ltop5['Date'],
        yaxis = 'y2',
        name = cripto_box,
        increasing = dict( line = dict( color = INCREASING_COLOR ) ),
        decreasing = dict( line = dict( color = DECREASING_COLOR ) ),
    )]

    layout=dict()

    fig = dict( data=data, layout=layout )
    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict( rangeselector = dict( visible = True ) )
    fig['layout']['yaxis'] = dict( domain = [0, 0.2], showticklabels = False )
    fig['layout']['yaxis2'] = dict( domain = [0.2, 0.8] )
    fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
    fig['layout']['margin'] = dict( t=40, b=40, r=40, l=40 )

    rangeselector=dict(
        visible = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 13 ),
        buttons=list([
            dict(count=1,
                 label='reset',
                 step='all'),
            dict(count=1,
                 label='1yr',
                 step='year',
                 stepmode='backward'),
            dict(count=3,
                label='3 mo',
                step='month',
                stepmode='backward'),

    dict(count=1,
                label='1 mo',
                step='month',
                stepmode='backward'),
            dict(step='all')
        ]))

    fig['layout']['xaxis']['rangeselector'] = rangeselector
    
    def movingaverage(interval, window_size=10):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    
    mv_y = movingaverage(df_ltop5['Close'])
    mv_x = list(df_ltop5['Date'])
    mv_x = mv_x[5:-5]
    mv_y = mv_y[5:-5]

    fig['data'].append( dict( x=mv_x, y=mv_y, type='scatter', mode='lines', 
                             line = dict( width = 1 ),
                             marker = dict( color = '#E377C2' ),
                             yaxis = 'y2', name='Média Móvel' ) )

    colors = []

    for i in range(len(df_ltop5['Close'])):
        if i != 0:
            if (df_ltop5['Close'][i] > df_ltop5['Close'][i-1]):
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)
    fig['data'].append( dict( x=df_ltop5['Date'], 
                             y=df_ltop5['Volume'],                         
                             marker=dict( color=colors ),
                             type='bar', 
                             yaxis='y', 
                             name='Volume' ) 
                      )
    
    def bbands(price, window_size=10, num_of_std=5):
        rolling_mean = price.rolling(window=window_size).mean()
        rolling_std  = price.rolling(window=window_size).std()
        upper_band = rolling_mean + (rolling_std*num_of_std)
        lower_band = rolling_mean - (rolling_std*num_of_std)
        return rolling_mean, upper_band, lower_band

    bb_avg, bb_upper, bb_lower = bbands(df_ltop5['Close'])

    fig['data'].append( dict( x=df_ltop5['Date'], y=bb_upper, type='scatter', yaxis='y2', 
                             line = dict( width = 1 ),
                             marker=dict(color='#ccc'), hoverinfo='none', 
                             legendgroup='Bandas de Bollinger', name='Bandas de Bollinger') )

    fig['data'].append( dict( x=df_ltop5['Date'], y=bb_lower, type='scatter', yaxis='y2',
                             line = dict( width = 1 ),
                             marker=dict(color='#ccc'), hoverinfo='none',
                             legendgroup='Bandas de Bollinger', showlegend=False ) )
   

    st.plotly_chart(fig)
    
def ajusta_df(df,df_scater,moeda):
    df_scater['Name']=moeda
    #df=df_scater.set_index('Date')
    #display(df)
    
    df_scater = df_scater.join(df, how='left', lsuffix='sc')

    df_scater=df_scater.reset_index()

    df_scater = df_scater[['index', 'Namesc','Close', 'Volume', 'Marketcap']]
    df_scater['Close']=df_scater.Close.fillna(0)
    df_scater['Volume']=df_scater.Volume.fillna(0)   
    df_scater['Marketcap']=df_scater.Volume.fillna(0)
    df_scater=df_scater.rename(columns={'index': 'Date', 'Namesc':'Name'})
    df_scater['Year'] =  pd.DatetimeIndex( df_scater['Date']).year
    #print(df_scater)
    return(df_scater)

def define_df_ajustado(df_raw):
    datas = df_raw.Date.unique()
    df_datas = df_raw.Date.unique()
    df_datas = np.sort(datas)
    df_scater = pd.DataFrame(index=df_datas,columns=['Name','Close','Volume', 'Marketcap'])
    df_final =  pd.DataFrame(columns=['Date','Name','Close','Volume','Year','Marketcap'])
    moedas = df_raw.Name.unique()
    for i in moedas:
        #df = df_cripto[(df_cripto['Name']== i)]
        df = df_raw[(df_raw['Name']== i)]
        df = df.set_index('Date')
        #display(df)
        df_ajustado = ajusta_df(df,df_scater,i)
        df_final = df_final.append(df_ajustado)
    return(df_final)
    
df_raw = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "archive/coin*.csv"))))

df_raw['Year'] =  pd.DatetimeIndex(df_raw['Date']).year
df_raw['Data'] =  pd.DatetimeIndex(df_raw['Date']).date

cripto = list(df_raw.Name.unique())

periodo = df_raw.Year.unique()
periodo.sort()
qtd_moedas = len(df_raw.Name.unique())

st.header('Projeto Data Science Degree')
st.title('Análise de Cripto Moedas')
st.subheader('Serão analisados a Evolução do Volume e dos Preços de negociação das Criptos')

per = 'De '+str(periodo[0])+ ' até '+ str(periodo[-1])

col1, col2 = st.columns(2)
col1.metric("Qtd Moedas", qtd_moedas)
col2.metric("Periodo Analisado", per)

# Select box

cripto_box = st.selectbox(
    'Selecione a Moeda',
    (cripto)  # Opções
)
df_cripto = df_raw[(df_raw['Name']==cripto_box)]
df_cripto = df_cripto.sort_values('Date')

plot_raw_data(df_cripto)
plot_candle_data(df_cripto)
plot_volume_data(df_cripto)

st.subheader('Avaliando a evolução do Marketcap de todas as Moedas, é possível elencar as 5 que alcançaram o maior patamar de Marketcap')
fig = px.line(df_raw, x="Date", y="Marketcap", color="Name",
              line_group="Name")
st.plotly_chart(fig)

df_top5 = df_raw.groupby(['Name'], sort=True)['Marketcap'].max().sort_values(ascending=False).head(5)
top5 = df_top5.index
df_topPrice = df_raw[df_raw['Name'].isin(top5)]
df_lowprice = df_topPrice.groupby(['Name'], sort=True)['Marketcap'].min().sort_values(ascending=True).head()
df_top5.sort_index()
df_lowprice.sort_index()
ltop5 = []
for i in zip(df_top5.index,df_top5,df_lowprice):
    ltop5.append(i)

st.subheader('As 5 moedas com maior evolução de Marketcap são:')
col1, col2 = st.columns(2)
col1.metric(ltop5[0][0],np.round(ltop5[0][1]),ltop5[0][2] )
col2.metric(ltop5[1][0],np.round(ltop5[1][1]),ltop5[1][2] )


col3, col4 = st.columns(2)
col3.metric(ltop5[2][0],np.round(ltop5[2][1]),ltop5[2][2] )
col4.metric(ltop5[3][0],np.round(ltop5[3][1]),ltop5[3][2] )

st.metric(ltop5[4][0],np.round(ltop5[4][1]),ltop5[4][2] )


# Select box

cripto_box = st.selectbox(
    'Selecione a Moeda para analisar a média móvel dos preços',
    (top5)  # Opções
)
df_ltop5 = df_raw[(df_raw['Name']==cripto_box)]
df_ltop5 = df_ltop5.sort_values('Date')

grafico_candle_2(df_ltop5)

df_final = define_df_ajustado(df_raw)
df_final = df_final[(df_final['Name'].isin(top5))]

range_ano = [ 2017,2018, 2019, 2020, 2021]
df_final = df_final[(df_final['Year'].isin(range_ano))]

st.subheader('Segue a evolução do Marketcap de transações das principais moedas')

fig = px.bar(df_final,
          x='Name',
          y='Marketcap'
                 ,
       #  size='Volume',
          color='Name',
          # hover_name='Name',
        #  log_x=True,
        #  size_max=100,
          animation_frame='Date',
          animation_group='Name',
                 #,
          #range_x=['2017', '2022'],
          range_y=[-10_000, 80_000_000_000],
        #  height=500
                )
st.plotly_chart(fig)

st.subheader('E por fim, a evolução dos preços em relação ao volume de transações das principais moedas')

range_ano = [2018, 2019, 2020, 2021]
df_final = df_final[(df_final['Year'].isin(range_ano))]

fig = px.scatter(df_final,
          x='Date',
          y='Close'
                 ,
         size='Volume',
          color='Name',
           hover_name='Name',
        #  log_x=True,
          size_max=100,
          animation_frame='Date',
          animation_group='Name',
                 #,
          range_x=['2017', '2022'],
          range_y=[-10_000, 80_000],
          height=500
                )
st.plotly_chart(fig)


    
import pandas as pd
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
#import technical_indicators as ts
import pandas as pd
import pandas
import xlsxwriter
import plotly
import quandl
from plotly import tools
import datetime
import glob

def EMA(df, base, target, period, alpha=False):
    """
    Function to compute Exponential Moving Average (EMA)

    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)

    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

    if (alpha == True):
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()

    df[target].fillna(0, inplace=True)
    return df



def ATR(df, period, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute Average True Range (ATR)

    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])

    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR)
            ATR (ATR_$period)
    """
    atr = 'ATR_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'TR' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df,'TR', atr, period, alpha=True)

    return df


def SuperTrend(df, period, multiplier, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute SuperTrend

    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])

    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR), ATR (ATR_$period)
            SuperTrend (ST_$period_$multiplier)
            SuperTrend Direction (STX_$period_$multiplier)
    """

    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)

    """
    SuperTrend Algorithm :

        BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
        BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR

        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)

        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    """

    # Compute basic upper and lower bands
    df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                         df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else \
        df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                         df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else \
        df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[
            i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] > \
                                     df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= \
                                         df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] < \
                                             df['final_lb'].iat[i] else 0.00

        # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down', 'up'), np.NaN)
    df
    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return df


#Step 2: Bring in data from AKK and read into df


data=pd.read_excel('ProjectUdaan.xlsx')
#data9=pd.read_excel('ConsolidatedData.xlsx')
#data1=data1.iloc[2:]
#print(list(data1))
data9.columns=['Symbol', 'Series', 'date', 'Prev Close', 'Open Price', 'High', 'Low', 'Last', 'Close', 'Average Price', 'Total Traded Quantity', 'Turnover', 'No. of Trades']
data9=data9.drop([ 'Series', 'Prev Close', 'Open Price',  'Last',  'Average Price', 'Total Traded Quantity', 'Turnover', 'No. of Trades'],axis=1)
#EMA(data,'open','new',7,alpha=True)
print(data9.head())
r= SuperTrend(data9,14,2)
r=r.reset_index()
#r=r.iloc[:2]
r=r.iloc[14:]
#year2=[2018,2019]
r['date']=pd.to_datetime(r['date'])
#r=r[r['date'].dt.year.isin(year2)]
#df=df[df.Year.isin(years)]
print("rhead")
print(r.head())
r = pd.melt(r, id_vars=['date','STX_14_2'], var_name='Type', value_name='values')

#print(r.head())

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
styles = {
    'pre': {
        'border': 'thin lightgrey solid',

    }
}

opt = ['Open', 'High', 'Low','Close']
data1 = r[r.Type.isin(opt)]
x = data1['date']

trace1 = go.Box(
        y=data1['values'],
        x=x,
        name='Price Range',
        marker=dict(
            color='#3D9970'
        )
    )

opt2 = ['ST_14_2']
data2 = r[r.Type.isin(opt2)]

data2 = data2.loc[data2['STX_14_2'] == 'up']
data2 = data2.drop_duplicates('date')
print(data2.head())
trace2 = go.Scatter(
        y=data2['values'],
        x=data2['date'],
        mode='markers',
        name='SuperTrend Low',
        connectgaps=False,
        marker=dict(
            color='rgba(152, 0, 0, .8)'
        )
    )

opt4 = ['ST_14_2']
data4 = r[r.Type.isin(opt4)]

data4 = data4.loc[data4['STX_14_2'] == 'down']
data4 = data4.drop_duplicates('date')

trace4 = go.Scatter(
        y=data4['values'],
        x=data4['date'],
        mode='markers',
        name='SuperTrend High',
        connectgaps=False,
        marker=dict(
            color='#3D9970'
        )
    )

opt5 = ['ST_14_2']
data5 = r[r.Type.isin(opt5)]

# data5=data4.loc[data4['STX_7_3']=='down']
    # data4=data4.drop_duplicates('date')
    # print(data4.head())
trace5 = go.Scatter(
        y=data5['values'],
        x=data5['date'],
        mode='lines',
        name='SuperTrend',
        connectgaps=False,
        marker=dict(
            color='#3D9970'
        )
    )

opt6 = ['ATR_14']
data6 = r[r.Type.isin(opt6)]
print('data6 heads')
print(data6.head())
    # data5=data4.loc[data4['STX_7_3']=='down']
    # data4=data4.drop_duplicates('date')
    # print(data4.head())
trace6 = go.Scatter(
        y=data6['values'],
        x=data6['date'],
        mode='lines',
        name='ATR',
        connectgaps=False,
        marker=dict(
            color='rgb(107,174,214)'
        )
    )



opt3 = ['Close']
data3 = r[r.Type.isin(opt3)]

trace3 = go.Scatter(
        y=data3['values'],
        x=x,
        name='Closing price',
        mode='markers',
        marker=dict(
            color='rgb(214, 12, 140)'
        )
    )
data = [trace1,trace2,trace4,trace3,trace5]
layout = go.Layout(
        yaxis=dict(
            title='SuperTrend',
            zeroline=False,

        ),xaxis=dict(range=[8000,11500]),height=800)


q=data9.Symbol.unique()



#figure = dict( data=data, layout=layout )
app.layout = html.Div([html.Div(
    [
        dcc.Markdown(
            '''
            ### Live Dashboard showing super trend computed along with the high and low prices for 50 stocks.
            '''.replace('  ', ''),
            className='eight columns offset-by-three'
        )
    ], className='row',
    style={'text-align': 'center', 'margin-bottom': '10px'}
),

    html.Div([
        html.Div(dcc.Dropdown(id='DropDown', options=[{'label': i
, 'value': i} for i in q],value='ADANIPORTS')),
        html.Div(id='container-button-basic',
             children='Select a stock in the above'),
        dcc.Graph(id='SuperTrend'),
    ],style={ 'width': '100%','float':'left','height':'800px'})

])

@app.callback(
    dash.dependencies.Output('SuperTrend', 'figure'),
    [dash.dependencies.Input('DropDown', 'value')])

def update_fig(value):

    data9=pd.read_excel('ConsolidatedData.xlsx')
    # data1=data1.iloc[2:]
    # print(list(data1))
    data9.columns = ['Symbol', 'Series', 'date', 'Prev Close', 'Open Price', 'High', 'Low', 'Last', 'Close',
                     'Average Price', 'Total Traded Quantity', 'Turnover', 'No. of Trades']
    data9 = data9.drop(
        ['Series', 'Prev Close', 'Open Price', 'Last', 'Average Price', 'Total Traded Quantity', 'Turnover',
         'No. of Trades'], axis=1)
    print("data9")
    print(data9.head())

    data9=data9[data["Symbol"]==value]
    #df = df[df.Year.isin(years)]
    r = SuperTrend(data9, 14, 2)
    #r = r.iloc[:2]
    r = r.iloc[14:]
    # year2=[2018,2019]
    #r['date'] = pd.to_datetime(r['date'])
    #r = r[r['date'].dt.year.isin(year2)]
    # df=df[df.Year.isin(years)]
    print("rhead")
    #print(r.head())
    r = pd.melt(r, id_vars=['Symbol','date', 'STX_14_2'], var_name='Type', value_name='values')
    'Final r'
    print(r.head())

    # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


    opt = ['Open', 'High', 'Low', 'Close']
    data1 = r[r.Type.isin(opt)]
    x = data1['date']

    trace1 = go.Box(
        y=data1['values'],
        x=x,
        name='Price Range',
        marker=dict(
            color='#3D9970'
        )
    )

    opt2 = ['ST_14_2']
    data2 = r[r.Type.isin(opt2)]

    data2 = data2.loc[data2['STX_14_2'] == 'up']
    data2 = data2.drop_duplicates('date')
    print(data2.head())
    trace2 = go.Scatter(
        y=data2['values'],
        x=data2['date'],
        mode='markers',
        name='SuperTrend Low',
        connectgaps=False,
        marker=dict(
            color='rgba(152, 0, 0, .8)'
        )
    )

    opt4 = ['ST_14_2']
    data4 = r[r.Type.isin(opt4)]

    data4 = data4.loc[data4['STX_14_2'] == 'down']
    data4 = data4.drop_duplicates('date')

    trace4 = go.Scatter(
        y=data4['values'],
        x=data4['date'],
        mode='markers',
        name='SuperTrend High',
        connectgaps=False,
        marker=dict(
            color='#3D9970'
        )
    )

    opt5 = ['ST_14_2']
    data5 = r[r.Type.isin(opt5)]

    # data5=data4.loc[data4['STX_7_3']=='down']
    # data4=data4.drop_duplicates('date')
    # print(data4.head())
    trace5 = go.Scatter(
        y=data5['values'],
        x=data5['date'],
        mode='lines',
        name='SuperTrend',
        connectgaps=False,
        marker=dict(
            color='#3D9970'
        )
    )

    opt6 = ['ATR_14']
    data6 = r[r.Type.isin(opt6)]
    print('data6 heads')
    print(data6.head())
    # data5=data4.loc[data4['STX_7_3']=='down']
    # data4=data4.drop_duplicates('date')
    # print(data4.head())
    trace6 = go.Scatter(
        y=data6['values'],
        x=data6['date'],
        mode='lines',
        name='ATR',
        connectgaps=False,
        marker=dict(
            color='rgb(107,174,214)'
        )
    )

    opt3 = ['Close']
    data3 = r[r.Type.isin(opt3)]

    trace3 = go.Scatter(
        y=data3['values'],
        x=x,
        name='Closing price',
        mode='markers',
        marker=dict(
            color='rgb(214, 12, 140)'
        )
    )
    data = [trace1, trace2, trace4, trace3, trace5]
    print('final data')
    print(data)
    layout = go.Layout(
        yaxis=dict(
            title='SuperTrend',
            zeroline=False,

        ), height=800)
    return {'data':data,
            'layout':layout}

#update_fig(value)

#data=data+trace6


#plotly.tools.set_credentials_file(username='kanishkan91',api_key='aYeSpFRWLtq4L1a2k6VC')
#py.plot(fig, filename='ver2',validate=False)

if __name__ == '__main__':
    app.run_server(debug=True)

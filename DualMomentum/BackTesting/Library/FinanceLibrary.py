import pandas as pd
import numpy as np

# Portfolio
def Portfolio(data,name="Portfolio"):
    data = data.dropna()  # NaN 제거하여
    Init_Balance = data[0]
    Final_Balace = data[-1]
    cagr = CAGR(data)
    std = STD(data)
    MaxDD = MDD(data)[1][-1]
    sharpe_ratio = SharpeRatio(data)
    # Initial Balance, Final Balance, CAGR, Std, Max. Drawdown, Sharpe Ratio
    inx = ["Initial Balnce", "Final Balance", "CAGR", "Std", "Max.Drawdown", "Sharpe Ratio"]
    port_folio = pd.Series([Init_Balance,Final_Balace,cagr,std,MaxDD,sharpe_ratio], index=inx, name=name)
    return port_folio

# EMA(Exponential Moving Average) : 지수이동평균
def EMA(data, timeperiod):
    # k : smoothing constant
    k = 2/(1+timeperiod)
    data = data.dropna()
    ema = pd.Series(index=data.index)
    ema[timeperiod-1] = data.iloc[0:timeperiod].sum() / timeperiod
    for i in range(timeperiod,len(data)):
        ema[i] = data[i]*k + ema[i-1] * (1-k)
    return ema

# MACD
def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
    macd = EMA(data, fastperiod) - EMA(data, slowperiod)
    macd_signal = EMA(macd, signalperiod)
    macd_osc = macd - macd_signal
    df = pd.concat([macd, macd_signal, macd_osc],axis=1)
    df.columns = ['MACD','Signal','Oscillator']
    return df

# CAGR
def CAGR(data):
    data = data.dropna()  # NaN 제거
    try:
        y = data.index.year.unique()
    except Exception as e:
        y = data.index.levels[0]  #멀티인덱스에서 year값
    result = (data[-1]/data[0])**(1/len(y))-1
    return np.round(result*100,4)  # %단위로 리턴

# DD & MDD
def MDD(data):
    window = len(data)

    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    Roll_Max = data.rolling(window, min_periods=1).max()
    Roll_Max.rename("Roll_Max", inplace=True)
    Drawdown = data/Roll_Max - 1.0
    Drawdown.rename("Drawdown", inplace=True)

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Drawdown = Drawdown.rolling(window, min_periods=1).min()
    Max_Drawdown.rename("Max_Drawdown", inplace=True)
    return np.round(Drawdown*100,4), np.round(Max_Drawdown*100,4)  # %단위로 리턴

# 일별수익
def YesterdayReturn(data):
    result = data/data.shift(1)   # 어제 진입하고 오늘 청산한다. 그러므로 어제 수익률 게산한다.
    return result

# 모멘텀 일별수익
def TomorrowReturn(data):
    # 전날가격/금일가격, 오늘 집입하면 내일청산한다. 그러므로 내일 수익률 게산한다
    result = data.shift(-1)/data
    return result

# 절대모멘텀
def SimpleMomentum(data,period):
    # shift를 사용하여 원화는 달의 수익률을 구한다
    # 당월주가 > n 개월 전 주가(n개월 모멘텀 > 0) --> 주식매수
    # 당월주가 < n 개월 전 주가(n개월 모멘텀 < 0) --> 주식매도
    result = np.where(data > data.shift(period),1,np.NaN)
    return pd.Series(result,index=data.index)

# 평균모멘텀스코어, data는 시리즈타입되고 DataFrame타입은 안됨.
def AverageMomentumScore(data,period=12):
    # shift를 사용하여 원화는 달의 수익률을 구한다
    # 당월주가 > n 개월 전 주가(n개월 모멘텀 > 0) --> 주식매수
    # 당월주가 < n 개월 전 주가(n개월 모멘텀 < 0) --> 주식매도
    result = 0
    for i in range(period+1):
        result += np.where(data > data.shift(i),1,0)
    result = result/period
    return pd.Series(result,index=data.index)

# 리얼모멘텀스코어
def RealMomentumScore(kodex,bond3,period=12):
    # n개월 real 모멘텀 = n개월 Kodex모멘텀 - n개월 bond3 모멘텀
    result = 0
    for i in range(period+1):
        result += np.where(kodex/kodex.shift(i) > bond3/bond3.shift(i), 1, 0)
    result = result/period
    return pd.Series(result,index=kodex.index)

# 평균모멘텀, 스코어 아님
'''
def AverageMomentum(data,period=12):
    # shift를 사용하여 원화는 달의 수익률을 구한다
    # 당월주가 > n 개월 전 주가(n개월 모멘텀 > 0) --> 주식매수
    # 당월주가 < n 개월 전 주가(n개월 모멘텀 < 0) --> 주식매도
    result = 0
    for i in range(period+1):
        result += data / data.shift(i)
    result = result/period
    return pd.Series(result,index=data.index)
'''

# 표준편차
def STD(data):
  return np.std(data)

# Sharpe Ratio
def SharpeRatio(data):
    data_returns = data.pct_change()  # +/- daily returns
    sharpe_ratio = data_returns.mean()/data_returns.std()
    return sharpe_ratio
# 매주 금요일 resample
# df_kodex = df_kodex.resample("W-FRI")._upsample(None).interpolate()

# market score구하기, 이동평균으로 구한다.주간데이타이면 24주 기준으로 작성함
def MarketScore(data):
    ma1 = pd.Series(data.rolling(window=4).mean(), index=data.index)
    ma2 = pd.Series(data.rolling(window=8).mean(), index=data.index)
    ma3 = pd.Series(data.rolling(window=12).mean(), index=data.index)
    ma4 = pd.Series(data.rolling(window=16).mean(), index=data.index)
    ma5 = pd.Series(data.rolling(window=20).mean(), index=data.index)
    ma6 = pd.Series(data.rolling(window=24).mean(), index=data.index)

    score = (np.where(data>ma1,1,0) \
            + np.where(data>ma2,1,0) \
            + np.where(data>ma3,1,0) \
            + np.where(data>ma4,1,0) \
            + np.where(data>ma5,1,0) \
            + np.where(data>ma6,1,0)) \
            / 6  #ma 6개 평균
    return score

def Ratio(kodex, bond10, bond3, cash=1):
    # 투자 비중 = KODEX평균모멘텀 스코어 / ( KODEX 12개월 평균 모멘텀 + BOND10 12개월 평균 모멘텀 + 현금비율)
    # 현금비율, 1(안정), 0.5(중간), 0.25(위험)
    kodex["Ratio"] = kodex["Avg Mo-Score"]/ (kodex["Avg Mo-Score"] + bond10["Avg Mo-Score"]  + cash)
    bond10["Ratio"] = bond10["Avg Mo-Score"]/ (kodex["Avg Mo-Score"] + bond10["Avg Mo-Score"] + cash)
    # 최종적으로 "1 - 사전비율"은 현금성 자산인 BOND3의 비율이 된다.
    bond3["Ratio"] = 1 - (bond10["Ratio"] + kodex["Ratio"])

    return kodex, bond10, bond3
# ATR 구하기
def ATR(High, Low, Close,w=14):
    H_L = High - Low
    H_Cp = np.abs(High - Close.shift(1))
    L_Cp = np.abs(Low - Close.shift(1))
    df = pd.concat([H_L, H_Cp, L_Cp],axis=1)
    TR = df.max(axis=1)
    ATR = TR.rolling(window=w,min_periods=1).mean()
    ATR = (ATR.shift(1) * 13 + TR) / w
    return ATR.dropna().apply(lambda x: int(x))

# Dual ATR은 short=4, long=14으로 구한다.
def DualATR(High, Low, Close, long=14, short=4):
    H_L = High - Low
    H_Cp = np.abs(High - Close.shift(1))
    L_Cp = np.abs(Low - Close.shift(1))
    df = pd.concat([H_L, H_Cp, L_Cp],axis=1)
    TR = df.max(axis=1)
    longATR = TR.rolling(window=long,min_periods=1).mean()
    longATR = (longATR.shift(1) * (long-1) + TR) / long
    shortATR = TR.rolling(window=long,min_periods=1).mean()
    shortATR = (shortATR.shift(1) * (short-1) + TR) / short
    df_atr = pd.concat([longATR, shortATR],axis=1)
    ATR = df_atr.max(axis=1)
    return ATR.dropna().apply(lambda x: int(x))

# 청산전략, 입력변수는 Dataframe으로 받고 컬럼은 하이,로우,종가,샹들리에 배수,요요 배수
def ChandelierYoYo(df,C_ATR=3,Y_ATR=2):
    df["ATR"] = ATR(df["High"],df["Low"],df["Close"],14)  # ATR을 14일 기준으로 뽑는다
    win = 20  # 평균값을 구하기위한 윈도우 값
    df["Max"] = df["Close"].rolling(window=win,min_periods=1).max()
    df["Min"] = df["Close"].rolling(window=win,min_periods=1).min()
    # 샹들리에 청산
    df["Chandelier"] = df["Max"] - df["ATR"] * C_ATR
    df["ChandelierExit"] = np.where(df["Low"] < df["Chandelier"],1,0)
    # 요요 청산
    df["YoYo"] = df["Close"].shift(1) - df["ATR"] * Y_ATR
    df["YoYoExit"] = np.where(df["Low"] < df["YoYo"],2,0)
    return df

# matplotlib _custom ax
from matplotlib.ticker import Formatter

class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''

        return self.dates[ind].strftime(self.fmt)

# PCI 구하기
def PhaseChangeIndicator(data,Period=20):
    P = Period - 1

    #Momentum = AverageMomentumScore(data,Period)
    Momentum = data - data.shift(Period)

    PCI = [0] * P  # PCI 저장할 리스트
    for ind in range(P,len(data)):
        sumUpDi =  0.0
        sumDownDi = 0.0
        Gradient = []
        Deviation = []
        for ind2 in range(Period):
            # ex) 5일 첫 Close + 5일 모멘텀 * 0/4, 1/4, 2/4, 3/4, 4/4
            val2 = data.iloc[ind-P] + (Momentum.iloc[ind]*(ind2)/P)
            Gradient.append(np.round(val2,4))
        for ind3 in range(Period):
            val3 = abs(data.iloc[ind-P+ind3] - Gradient[ind3])
            if data.iloc[ind-P+ind3] > Gradient[ind3]:
                sumUpDi += val3
            elif data.iloc[ind-P+ind3] < Gradient[ind3]:
                sumDownDi += val3
            else:
                None
        #ZeroDivisionError 방지코자 if문 사용함
        if sumUpDi != 0:
            a = sumUpDi / (sumUpDi+sumDownDi) * 100
            PCI.append(np.round(a,2))
        else:
            PCI.append(0)
    return pd.Series(PCI, index=data.index)

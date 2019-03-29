
# coding: utf-8

# ### KODEX, BOND10, BOND3 ETF 일별 데이타 크롤링

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import FinanceDataReader  as fdr

# KODEX200 크롤링 하기
def kodex200():
    try:
        df_kodex = pd.read_csv("..\\Data\\KODEX200_data.csv",encoding="utf-8",sep=",",engine="python")
        df_kodex.index = pd.to_datetime(df_kodex["Date"])
        df_kodex = df_kodex.drop(['Date'], axis=1)
        last_day = df_kodex.index[-1]  # 크롤링된 마지막 날짜 구하기
        get_day = last_day + timedelta(1)  # 마지막날짜에 +1 day
        get_day = get_day.strftime("%Y-%m-%d")  # 지정된 날자를 포함하여 데이터 크롤링

        df_update = fdr.DataReader("069500",get_day)
        if len(df_update) > 0:  # new 데이타가 있으면
            df_kodex = pd.concat([df_kodex,df_update],axis=0, sort=True)
    except FileNotFoundError:
        df_kodex = fdr.DataReader("069500")  # KODEX200 069500
    # 중복색인 삭제, 색인 중복 error 제거
    df_kodex = df_kodex[~df_kodex.index.duplicated(keep='first')]
    # ### csv 파일로 저장하기
    df_kodex.to_csv("..\\Data\\KODEX200_data.csv",sep=",",encoding="utf-8")

# 국고채10년 크롤링 하기
def bond10():
    try:
        df_bond10 = pd.read_csv("..\\Data\\KOSEF국고채10년_data.csv",encoding="utf-8",sep=",",engine="python")
        df_bond10.index = pd.to_datetime(df_bond10["Date"])
        df_bond10 = df_bond10.drop(['Date'], axis=1)
        last_day = df_bond10.index[-1]  # 크롤링된 마지막 날짜 구하기
        get_day = last_day + timedelta(1)  # 마지막날짜에 +1 day
        get_day = get_day.strftime("%Y-%m-%d")  # 지정된 날자를 포함하여 데이터 크롤링

        df_update = fdr.DataReader("069500",get_day)
        if len(df_update) > 0:  # new 데이타가 있으면
            df_bond10 = pd.concat([df_bond10,df_update],axis=0, sort=True)
    except FileNotFoundError:
        df_bond10 = fdr.DataReader("148070")  # KOSEF 국고채10년
    # 중복색인 삭제, 색인 중복 error 제거
    df_bond10 = df_bond10[~df_bond10.index.duplicated(keep='first')]
    # ### csv 파일로 저장하기
    df_bond10.to_csv("..\\Data\\KOSEF국고채10년_data.csv",sep=",",encoding="utf-8")

# 국고채3년 크롤링 하기
def bond3():
    try:
        df_bond3 = pd.read_csv("..\\Data\\KOSEF국고채3년_data.csv",sep=",",engine="python")
        df_bond3.index = pd.to_datetime(df_bond3["Date"])
        df_bond3 = df_bond3.drop(['Date'], axis=1)
        last_day = df_bond3.index[-1]  # 크롤링된 마지막 날짜 구하기
        get_day = last_day + timedelta(1)  # 마지막날짜에 +1 day
        get_day = get_day.strftime("%Y-%m-%d")  # 지정된 날자를 포함하여 데이터 크롤링

        df_update = fdr.DataReader("114470",get_day)
        if len(df_update) > 0:  # new 데이타가 있으면
            df_bond3 = pd.concat([df_bond3,df_update],axis=0, sort=True)
    except FileNotFoundError:
        df_bond3 = fdr.DataReader("114470")  # KOSEF 국고채3년
    # 중복색인 삭제, 색인 중복 error 제거
    df_bond3 = df_bond3[~df_bond3.index.duplicated(keep='first')]
    # ### csv 파일로 저장하기
    df_bond3.to_csv("..\\Data\\KOSEF국고채3년_data.csv",sep=",")

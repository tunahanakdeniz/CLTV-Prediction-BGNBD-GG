import datetime as dt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#!pip install lifetimes
#!pip install sqlalchemy
#!python -m pip install anaconda mysql-connector-python

from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#2010-2011 UK müşterileri için 6 aylık CLTV Prediction

df=retail_mysql_df[retail_mysql_df["Country"]=="United Kingdom"]

df.head()
df.shape

#Veri seti betimsel istatistiklerine bakıldı. Eksik veriler ve iptal edilen faturayı temsil eden "c" içeren invoice row'ları temizlendi.
#Quantity ve Price 0'dan küçük olamayacağı için güncellendi.
#Analiz tarihinin(T) belirlenmesi için today_date değişkeni tarih 12.11.2011 olarak şekilde güncellendi.
#Monetary değerinin temsil eden Total Price değişkeni oluşturuldu.

df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T
df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011, 12, 11)


#Lifetime Veri Yapısının Hazırlanması

#recency: Son satın alma üzerinden geçen zaman. Haftalık. (daha önce analiz gününe göre, burada kullanıcı özelinde)
#T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
#frequency: tekrar eden toplam satın alma sayısı (frequency>1)
#monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

#Yukarıda CLTV calculation için gerekli değişkenler oluşturuldu.

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df = cltv_df[cltv_df["monetary"] > 0]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

cltv_df["frequency"] = cltv_df["frequency"].astype(int)

#Değişken isimleri Recency, T, Frequency ve Monetary olarak güncellendi ve raporlama haftalık istendiğin için haftalığa dönüştürüldü.

#2. BG-NBD Modelinin Kurulması

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

#1 haftalık beklenen satın alma değişkeni oluşturuldu.

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


#1 aylık beklenen satın alma değişkeni oluşturuldu.

cltv_df["expected_purc_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_df.sort_values("expected_purc_6_month", ascending=False).head()

#Yorumlamalar için kullanılmak üzere 6 ay içerisinde satın alma tahminleri için değişken oluşturuldu. En yüksek satın alma sayısına sahip id'ler incelendi.

# 3. GAMMA-GAMMA Modelinin Kurulması


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)


ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.head()

# Gamma-Gamma modeli ile beklenen ortalama kar tahminiyle ilgili değişken oluşturuldu.

# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.


cltv_6months = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # Görev tanımında 6 aylık prediction bekleniyor.
                                   freq="W",  
                                   discount_rate=0.01)
cltv_6months = cltv_df.merge(cltv_6months, on="CustomerID", how="left")

cltv_6months.sort_values(by="clv", ascending=False).head(10)

#2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplama.
#1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişi analizi


cltv_1month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # Görev tanımında 1 aylık prediction bekleniyor.
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv_1month = cltv_1month.reset_index()
cltv_1month = cltv_df.merge(cltv_1month, on="CustomerID", how="left")

cltv_1month.sort_values(by="clv", ascending=False).head(10)

cltv_12month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # Görev tanımında 12 aylık prediction bekleniyor.
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv_12month = cltv_12month.reset_index()
cltv_12month = cltv_df.merge(cltv_12month, on="CustomerID", how="left")

cltv_12month.sort_values(by="clv", ascending=False).head(10)

# 1 aylık cltv ve 12 aylık cltv değerleri karşılaştırıldığında ilk 5 id'nin aynı olduğu ancak diğer id'lerin farklılık gösterdiği görülmektedir.
# Bu farklılığa sebep olarak ilk 1 aylık periyotta cltv değerleri bakımında ilk 10'sıra içerisinde yer alan müşterilerin uzun vadeli müşteriler olmadığı ya da recency ve frequency ve monetary değerleri incelendiğinde şirket için 12 aylık bir tabloda ilk 10 sırada kalacak value değerine ulaşamdığı öngörülüyor.


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  #6aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()


cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])


cltv_final.sort_values(by="scaled_clv", ascending=False).head()

#BG-NBD ve Gamma-Gamma modelleri birlikte kullanılarak 6 aylık Customer Life Time Value hesaplandı. 
#Bu hesabı değerlendirilebilir hale getirmek için stardartlaştırma yapıldı.(scaled_clv).

#Segmentasyon
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()
cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)
cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


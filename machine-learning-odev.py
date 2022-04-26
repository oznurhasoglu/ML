import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #eğitim ve test verilerini ayırmak için

#veri yukleme
veriler= pd.read_csv("odev_tenis.csv")
print(veriler)

outlook= veriler.iloc[:,0:1].values  
le= preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe= preprocessing.OneHotEncoder()
outlook= ohe.fit_transform(outlook).toarray()

"""BAK BURAYA"""
#play ve windy verilerini tek tek labelencoder yapmak yerine aşağıdaki gibi .apply ile aynı anda daha kısa yapabiliriz.
WindyandPlay= veriler.iloc[:,3:].apply(preprocessing.LabelEncoder().fit_transform)
print(WindyandPlay)

parca1= pd.DataFrame(data=outlook, index=range(14), columns=["sunny","overcast","rainy"])
parca2= pd.DataFrame(data= veriler.iloc[:,1:3], index=range(14), columns=["temperature", "humidity"])
parca3= pd.DataFrame(data=WindyandPlay, index=range(14), columns=["windy","play"])

birlestir12 =pd.concat([parca1, parca2], axis=1)
birlestir123 =pd.concat([birlestir12, parca3], axis=1)
print(birlestir123)

#play tahmini
verilen= birlestir123.iloc[:,0:6]
istenilen= birlestir123.iloc[:,-1:]

x_train, x_test, y_train, y_test= train_test_split(verilen, istenilen, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train, y_train)
y_tahmin= lr.predict(x_test) 

#bir de humidity tahmini yapıyım
verilen2= pd.concat([birlestir123.iloc[:,:4], birlestir123.iloc[:,5:]], axis=1)
x_train, x_test, y_train, y_test= train_test_split(verilen2, birlestir123.iloc[:,4:5], test_size=0.33, random_state=0)
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train, y_train)
y_tahmin= lr.predict(x_test) 
a= y_tahmin #daha sonra geri eleme yapınca çıkan sonuçla karşılaştırmak için bu sonucu bir değişkene atadım.

#geri eleme yöntemiyle sonucu iyileştirmeye çalışalım
import statsmodels.api as sm
x= np.append(arr= np.ones((14,1)).astype(int), values= verilen2, axis=1)
degiskenlistesi= verilen2.iloc[:, [0,1,2,3,4,5]].values
degiskenlistesi= np.array(degiskenlistesi, dtype=float)
#OLS raporu oluşturuyoruz
humidity= birlestir123.iloc[:,4:5]
model= sm.OLS(humidity,degiskenlistesi).fit() #OLS(bagımlı deg, bağımsız degs)
print(model.summary())

#4. verinin (windy) p valuesi en yüksek onu çıkarıp tekrar bakıyorum
x= np.append(arr= np.ones((14,1)).astype(int), values= verilen2, axis=1)
degiskenlistesi= verilen2.iloc[:, [0,1,2,3,5]].values
degiskenlistesi= np.array(degiskenlistesi, dtype=float)
#OLS raporu oluşturuyoruz
humidity= birlestir123.iloc[:,4:5]
model= sm.OLS(humidity,degiskenlistesi).fit() #OLS(bagımlı deg, bağımsız degs)
print(model.summary())

#xtest ve xtrainden windyi çıkarıp eğitimi yeniden yapıyoruz
x_test= pd.concat([x_test.iloc[:,:4], x_test.iloc[:,5:]], axis=1)
x_train= pd.concat([x_train.iloc[:,:4], x_train.iloc[:,5:]], axis=1)

lr.fit(x_train, y_train)
y_tahmin= lr.predict(x_test)

#karşılaştıralım
print(a) #6değişken
print(y_tahmin) #5degişken 
print(y_test) #gercek sonuc


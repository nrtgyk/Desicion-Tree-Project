import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")



df.head()  #veri setinin ilk beş satırına bak


duzeltme_mapping = {'Y': 1, 'N': 0}  # veri setinde N ve Y olan yerleri 1 ve 0 olarak değiştiriyoruz.

df['IseAlindi'] = df['IseAlindi'].map(duzeltme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzeltme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzeltme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzeltme_mapping)
duzeltme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}  
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzeltme_mapping_egitim)



df.head()
#print(df.head()) ## kod düzgün çalışıyor mu diye kontrol edildi ve yorum satırına alındı


# Sonuc sütununu ayırıyorıp, eğitmeye başlıyoruz

y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)

#print(y) ##tekrar kod çalışıyor mu kontrolleri yapıldı
#print(X)

X.head()


# Decision Tree'mizi oluşturuyoruz:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X.values,y.values)



# Prediction yapalım şimdi
# 5 yıl deneyimli, hazlihazırda bir yerde çalışan ve 3 eski şirkette çalışmış olan, eğitim seviyesi Lisans
# top-tier-school mezunu değil
print(clf.predict([[5, 1, 3, 0, 0, 0]]))





# Toplam 2 yıllık iş deneyimi, 7 kez iş değiştirmiş çok iyi bir okul mezunu şuan çalışmıyor
print (clf.predict([[2, 0, 7, 0, 1, 0]]))



# Toplam 2 yıllık iş deneyimi, 7 kez iş değiştirmiş çok iyi bir okul mezunu değil şuan çalışıyor
print (clf.predict([[2, 1, 7, 0, 0, 0]]))


# Toplam 20 yıllık iş deneyimi, 5 kez iş değiştirmiş iyi bir okul mezunu şuan çalışmıyor
print (clf.predict([[20, 0, 5, 1, 1, 1]]))





# ## Toplu Öğrenme: Random Forest

# 20 tane decision tree birleşiminden oluşan bir Random Forest kullanarak tahmin yapacağız:


from sklearn.ensemble import RandomForestClassifier




rnd_fr_clf = RandomForestClassifier(n_estimators=20)
rnd_fr_clf = rnd_fr_clf.fit(X.values,y.values)

#Predict employment of an employed 10-year veteran
print (rnd_fr_clf.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (rnd_fr_clf.predict([[10, 0, 4, 0, 0, 0]]))




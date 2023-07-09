import pandas as pd
import re
import string
import nltk
import spacy
import preprocess_kgptalkie as ps
from flask import Flask, request
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

#Data okuma
df1 = pd.read_excel("hb_dataset.xlsx")
df2 = pd.read_excel("negativecomments.xlsx")
df3 = pd.read_excel("positivecomments.xlsx")
df4 = pd.read_excel("urun_yorumlari.xlsx")

data=open("magaza_yorumlari_duygu_analizi2.csv", encoding="utf-16")
df=pd.read_csv(data,sep=';', error_bad_lines=False)

def fonk(x):
    x=x.replace(';',',')
    string2 = ",".join(x.split(",")[:-1]) + " ; " + x.split(",")[-1]
    return string2

df['Görüş,Durum']=df['Görüş,Durum'].apply(lambda x: fonk(x))
a=[]
b=[]
for i in df['Görüş,Durum']:
    a.append(i.split(';')[0])
    b.append(i.split(';')[1])

df['Görüş']=a
df['Durum']=b   
df.drop(['Görüş,Durum'], axis=1,inplace=True) 

#Ön İşleme
def siniflandirma_fonk(x):
    if x>3:
        return 'pozitif'
    elif x==3:
        return 'notr'
    else:
        return 'negatif'

def sinif_df4(x):
    if x==2:
        return 'notr'
    elif x==0:
        return 'negatif'
    else :
        return 'pozitif'

def sinif_df3(x):
    if x=='n':
        return 'notr'
    elif x=='f':
        return 'negatif'
    else :
        return 'pozitif'

def sinif_df(x):
    if x==' Tarafsız':
        return 'notr'
    elif x==' Olumsuz':
        return 'negatif'
    else :
        return 'pozitif'
## DF

df.rename(columns ={'Durum':'Yorum','Görüş':'Değerlendirme'},inplace=True)
df['Yorum']=df['Yorum'].apply(sinif_df)

## DF1

df1.drop(['Ürün Adı'], axis=1,inplace=True)
df1.dropna(inplace=True)
df1['Yorum']=df1['Yorum'].apply(siniflandirma_fonk)

## DF2

df2['Durum'].fillna('negatif',inplace=True)
df2.rename(columns ={'Durum':'Yorum','Metin':'Değerlendirme'},inplace=True)
df2['Yorum'] = df2['Yorum'].replace(2, 'notr')
df2['Yorum'] = df2['Yorum'].replace(1, 'pozitif')
df2['Yorum'] = df2['Yorum'].replace(0, 'negatif')

## DF3

df3['Durum'].fillna('pozitif',inplace=True)
df3.rename(columns ={'Durum':'Yorum','Metin':'Değerlendirme'},inplace=True)
df3['Yorum']=df3['Yorum'].apply(sinif_df3) 

## DF4

df4.rename(columns ={'Durum':'Yorum','Metin':'Değerlendirme'},inplace=True)
df4['Yorum']=df4['Yorum'].apply(sinif_df4)

df=df[~df['Değerlendirme'].astype(str).str.contains('elbise|ruj|parfüm|pantolon|göz|makyaj|krem', regex=True,na=False)]
df4=df4[~df4['Değerlendirme'].astype(str).str.contains('elbise|ruj|parfüm|pantolon|göz|makyaj|krem', regex=True,na=False)]
## Birleştirme

data=pd.concat([df1, df2,df3,df4,df], ignore_index=True)

# Clean

def get_clean(x):
    x = re.sub(r'[^\w\s]', ' ', x)
    x = str(x).lower().replace('\\', ' ').replace('_', ' ')
    x = x.translate(str.maketrans("çğıöşü", "cgiosu"))
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    x = x.translate(str.maketrans('', '', string.digits))
    return x

data["Değerlendirme"]=data["Değerlendirme"].apply(lambda x: get_clean(x))

data.drop_duplicates(subset=["Değerlendirme"],inplace=True)
data.reset_index(inplace=True,drop=True)

#Label Encoder
le=LabelEncoder()
data.Yorum=le.fit_transform(data.Yorum.astype(str))

# Model

X=data['Değerlendirme']
y=data['Yorum']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=64,stratify=data['Yorum'],shuffle=True)

vectorizer = CountVectorizer(max_features=106000,ngram_range=(1, 3),binary=True,strip_accents="ascii").fit(X_train)
X_train= vectorizer.transform(X_train)
X_test= vectorizer.transform(X_test)

smote=SMOTE(sampling_strategy="minority",random_state=24)
X_resampled,y_resampled =smote.fit_resample(X_train,y_train)

mnb_model = MultinomialNB(alpha=1.763).fit(X_resampled,y_resampled)
y_pred=mnb_model.predict(X_test)

def tahmin(x):    
    x = get_clean(x)
    vec = vectorizer.transform([x])
    a = mnb_model.predict(vec)[0]
    b = mnb_model.predict_proba(vec)
    if a == 0 :
        return 'cümleniz negatif\n\n' + 'Doğruluk ihtimali: ' + str(b[0][0]*100)        
    elif a == 2 :        
        return 'cümleniz pozitif\n\n'+ 'Doğruluk ihtimali: ' + str(b[0][2]*100)
    else:
        return 'cümleniz nötr\n\n' + 'Doğruluk ihtimali: ' + str(b[0][1]*100)
    return 

app = Flask(__name__) 
    
@app.route('/nlp_project', methods = ['GET','POST'])

def upload_page():
    if request.method == "GET":
        return 'Bir cümle giriniz'
    elif request.method == "POST":
        x = request.json['cumle']
        return tahmin(x)

app.run(port=9000)





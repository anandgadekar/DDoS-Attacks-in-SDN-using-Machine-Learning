from flask import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
enc=LabelEncoder()
os=RandomOverSampler()
app=Flask(__name__)
global df,path,data

def preproceesing(file):
    file['proto']=enc.fit_transform(file['proto'])
    file['flgs'] = enc.fit_transform(file['flgs'])
    file['saddr'] = enc.fit_transform(file['saddr'])
    file['daddr'] = enc.fit_transform(file['daddr'])
    file['sport'] = enc.fit_transform(file['sport'])
    file['state'] = enc.fit_transform(file['state'])
    file['category'] = enc.fit_transform(file['category'])
    file['subcategory'] = enc.fit_transform(file['subcategory'])
    return file

def drop(file):
    file.drop(['Unnamed: 0'],axis=1,inplace=True)
    # Dropping Unnecessary column from the dataset
    return file
def splitting(file):
    X=file.drop(['attack','sport','dport'],axis=1)
    y=file['attack']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=52)
    return X_train,X_test,y_train,y_test


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('blog.html')



@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predicback',methods=['POST','GET'])
def predicback():
    df=pd.read_csv('ddos.csv')

    if request.method == "POST":
        f1=request.form['f1']
        f2 = request.form['f2']
        print(f2)
        f3 = request.form['f3']
        print(f3)
        f4 = request.form['f4']
        print(f4)
        f5 = request.form['f5']
        print(f5)
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        f11 = request.form['f11']
        f12 = request.form['f12']
        f13 = request.form['f13']
        f14 = request.form['f14']
        f15 = request.form['f15']
        f16 = request.form['f16']
        f17 = request.form['f17']
        f18 = request.form['f18']
        f19 = request.form['f19']
        f20 = request.form['f20']
        f21 = request.form['f21']
        f22 = request.form['f22']
        f23 = request.form['f23']
        f24 = request.form['f24']
        f25 = request.form['f25']
        f26 = request.form['f26']
        f27 = request.form['f27']
        f28 = request.form['f28']
        f29 = request.form['f29']
        f30 = request.form['f30']
        f31 = request.form['f31']
        f32 = request.form['f32']
        f33 = request.form['f33']
        f34 = request.form['f34']
        f35 = request.form['f35']
        f36 = request.form['f36']
        f37 = request.form['f37']
        f38 = request.form['f38']
        f39 = request.form['f39']
        f40 = request.form['f40']
        f41 = request.form['f41']
        f42 = request.form['f42']
        f43 = request.form['f43']
        l = [float(f1), float(f2), float(f3), float(f4), float(f5), float(f6), float(f7), float(f8), float(f9),
             float(f10), float(f11), float(f12), float(f13), float(f14), float(f15), float(f16), float(f17), float(f18),
             float(f19), float(f20), float(f21), float(f22), float(f23),
             float(f24), float(f25), float(f26), float(f27), float(f28), float(f29), float(f30), float(f31), float(f32),
             float(f33), float(f34), float(f35), float(f36), float(f37), float(f38), float(f39), float(f40), float(f41),
             float(f42), float(f43)]
        df = pd.read_csv("ddos.csv")
        df.head()
        df.info()
        df.describe()
        pdf = preproceesing(df)
        print(pdf)
        ddf = drop(df)
        print(ddf)
        # X=splitting(file)
        # y=splitting(file)
        X_train, X_test, y_train, y_test = splitting(df)
        print(X_train)
        print(y_test)
        X = df.drop(['attack', 'sport', 'dport'], axis=1)
        y = df['attack']
        X_train_res, y_train_res = os.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.3, random_state=52)
        print(X_train)
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        dtpred = dt.predict(X_test)
        acs = accuracy_score(y_test, dtpred)
        print(acs)
        dtpred = dt.predict([l])
        print('hhhhhhhhhhhhhhhhh')
        print(dtpred)
        if dtpred == [0]:
            msg = 'This Network is Safe'
            return render_template('prediction.html', msg=msg)
        else:
            msg = 'This Network is Attacked'
        return render_template('prediction.html',msg=msg)
    return render_template('prediction.html')

@app.route('/upback',methods=['POST','GET'])
def upback():
    if request.method=='POST':
        df=request.files['file']
        data=pd.read_csv(df)
        print(data)
        x=data.columns
        print(x)
        y=data.values.tolist()
        print(y)

        return render_template("blog.html",rows=y,col=x)
    return render_template("blog.html")

if __name__=="__main__":
    app.run(debug=True)
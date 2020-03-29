from flask import Flask, render_template, request, flash, session
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
from sklearn.externals import joblib
import requests
import pandas as pd
import flask
import os

def predict(parameters):
    # モデル読み込み
    model = joblib.load('./nn.pkl')
    #params = parameters.reshape(1,-1)
    #pred = model.predict(parameters)
    #return pred

    hospital = pd.read_csv("hospital.csv",encoding='cp932')
    is_complete_duplicate_keep_first = (hospital.duplicated(keep='first',subset=['医療機関名']))

    new_hospital=hospital[~is_complete_duplicate_keep_first]
    new_hospital.count()

    hospital_name=new_hospital.loc[:,"医療機関名"]
    #print(hospital_name)

    hospital_URL=new_hospital.loc[:,"病院URL"]
    #print(hospital_URL)

    hospital_data = new_hospital.drop(["調査日",'医療機関名',"病院URL"],axis=1)
    hospital_data.head()

    data1 = pd.DataFrame(hospital_data)
    new_data1 = pd.get_dummies(data1)
    #new_data1.head()

    new_data1.loc[854]=parameters

    distance, indice = model.kneighbors(new_data1.iloc[new_data1.index== 854].values.reshape(1,-1),n_neighbors=11)
    index=[]
    for i in range(0, len(distance.flatten())):
    	if  i == 0:
        	pass
        	#print('Recommendations if you like the hospital {0}:\n'.format(new_data1[new_data1.index== hospital_status].index[0]))
    	else:
        	index.append(new_data1.index[indice.flatten()[i]])

    return index
        	#print('{0}　 \n病院名：{1}　\n病院URL：{2}\n'.format(i,hospital_name[index],hospital_URL[index],distance.flatten()[i]))



app = Flask(__name__)
app.secret_key = 'hogehoge'

class hospitalForm(Form):
    #SepalLength = flask.request.form.get("SepalLength",
    #[validators.InputRequired("この項目は入力必須です")])

    #SepalWidth = flask.request.form.get("SepalWidth",
    #[validators.InputRequired("この項目は入力必須です")])

    #PetalLength = flask.request.form.get("PetalLength",
    #[validators.InputRequired("この項目は入力必須です")])

    #PetalWidth = flask.request.form.get("PetalWidth",
    #[validators.InputRequired("この項目は入力必須です")])

    SepalLength = FloatField("平日の外来　(ありの場合：1　なしの場合：0を入力してください)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    SepalWidth  = FloatField("土日の外来　(ありの場合：1　なしの場合：0を入力してください)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    PetalLength = FloatField("入院　(ありの場合：1　なしの場合：0を入力してください)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    PetalWidth  = FloatField("救急　(ありの場合：1　なしの場合：0を入力してください)",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    # html側で表示するsubmitボタンの表示
    submit = SubmitField("おすすめの病院を探す")


@app.route('/', methods = ['GET', 'POST'])
def predicts():

    hospital = pd.read_csv("hospital.csv",encoding='cp932')
    is_complete_duplicate_keep_first = (hospital.duplicated(keep='first',subset=['医療機関名']))

    new_hospital=hospital[~is_complete_duplicate_keep_first]
    new_hospital.count()

    hospital_name=new_hospital.loc[:,"医療機関名"]
    #print(hospital_name)

    hospital_URL=new_hospital.loc[:,"病院URL"]
    #print(hospital_URL)

    hospital_day=new_hospital.loc[:,"外来_平日"]
    hospital_holiday=new_hospital.loc[:,"外来_土日"]
    hospital_nyuuinn=new_hospital.loc[:,"入院"]
    hospital_kyuukyuu=new_hospital.loc[:,"救急"]

    form = hospitalForm(request.form)
    geo_request_url = 'https://get.geojs.io/v1/ip/geo.json'
    geo_data = requests.get(geo_request_url).json()
    geo_data_la = geo_data['latitude']
    geo_data_lo = geo_data['longitude']

    x = np.array([geo_data['latitude'],geo_data['longitude']])

    if request.method == 'POST':

        if form.validate() == False:
            flash("全て入力する必要があります")
            return render_template('index.html', form=form)
        else:
            #SepalLength = flask.request.form.get("SepalLength")
            #[validators.InputRequired("この項目は入力必須です")])
            SepalLength = float(request.form["SepalLength"])
            if SepalLength==1:
                x=np.append(x,[0,0,0,1])
            else:
                x=np.append(x,[1,0,0,0])
            #SepalWidth = flask.request.form.get("SepalWidth")
            SepalWidth  = float(request.form["SepalWidth"])
            if SepalWidth==1:
                x=np.append(x,[0,0,0,1])
            else:
                x=np.append(x,[1,0,0,0])
            #PetalLength = flask.request.form.get("PetalLength")
            PetalLength  = float(request.form["PetalLength"])
            if PetalLength==1:
                x=np.append(x,[0,0,0,1])
            else:
                x=np.append(x,[1,0,0,0])
            #PetalWidth = flask.request.form.get("PetalWidth")
            PetalWidth  = float(request.form["PetalWidth"])
            if PetalWidth==1:
                x=np.append(x,[0,0,0,1])
            else:
                x=np.append(x,[1,0,0,0])

            #print(x)

       	    #new_data1.loc[854] = [geo_data['latitude'], geo_data['longitude'], 0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1
            #x = np.array([geo_data['latitude'],geo_data['longitude'],0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1])
            index=predict(x)
            #irisName = getName(pred)
            return render_template('result.html',
            Name1=hospital_name[index[0]], URL1=hospital_URL[index[0]], day1=hospital_day[index[0]],holiday1=hospital_holiday[index[0]],nyuuinn1=hospital_nyuuinn[index[0]],kyuukyuu1=hospital_kyuukyuu[index[0]],
            Name2=hospital_name[index[1]], URL2=hospital_URL[index[1]], day2=hospital_day[index[1]],holiday2=hospital_holiday[index[1]],nyuuinn2=hospital_nyuuinn[index[1]],kyuukyuu2=hospital_kyuukyuu[index[1]],
            Name3=hospital_name[index[2]], URL3=hospital_URL[index[2]], day3=hospital_day[index[2]],holiday3=hospital_holiday[index[2]],nyuuinn3=hospital_nyuuinn[index[2]],kyuukyuu3=hospital_kyuukyuu[index[2]],
            Name4=hospital_name[index[3]], URL4=hospital_URL[index[3]], day4=hospital_day[index[3]],holiday4=hospital_holiday[index[3]],nyuuinn4=hospital_nyuuinn[index[3]],kyuukyuu4=hospital_kyuukyuu[index[3]],
            Name5=hospital_name[index[4]], URL5=hospital_URL[index[4]], day5=hospital_day[index[4]],holiday5=hospital_holiday[index[4]],nyuuinn5=hospital_nyuuinn[index[4]],kyuukyuu5=hospital_kyuukyuu[index[4]],)

    elif request.method == 'GET':

        return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run()

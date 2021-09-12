from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('radientBo.sav','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))


def feature_change(features):
    final = []
    #Sex Age Fare    Embarked    last_name   title   ticket_type class_cabin family_size
    embarked_dict = {'S':2,'C':0,'Q':1}
    gender_dict = {'Male':1,'Female':0}
    cabin_dict = {'B':1,'C':2,'D':3,'E':4,'A':5,'T':6,'F':7,'G':8}
    title_dict = {'Master':0,'Miss':1,'Mr.':2,'Mrs.':3}
    class_cabin = int(str(cabin_dict[str(features[8])]) + str(features[7]))

    final.append(int(gender_dict[features[3]])) #Sex 
    final.append(int(features[4])) #Age
    final.append(int(features[5])) #Fare
    final.append(int(embarked_dict[features[6]])) #Embarked
    final.append(int(0)) #Last name
    final.append(int(title_dict[features[0]])) #Title
    final.append(int(0)) #ticket type
    final.append(int(class_cabin)) #class_cabin
    final.append(int(features[9])) #family size
    # print('before scaling', final)
    # final = scaler.transform([final])
    # print('after scaling', final[0])
    # return final[0]
    return final


@app.route('/')
def hello_world():
    return render_template("titanic_survival.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    features = [x for x in request.form.values()]
    # print(features)
    #['Miss', 'Apurva', 'Sijaria', 'Female', '24', '12', 'C', '3', 'C', '2']
    #['title' 0, 'first name' 1, 'lastname' 2, 'gender' 3, 'age' 4 , 'fare' 5, 'embarked' 6, 'pclass' 7 , 'cabin code' 8, 'family size' 9]
    final=feature_change(features)
    # print(final)
    prediction=model.predict_proba([final])
    # print(prediction[0][1])
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if prediction[0][1]<(0.5):
        return render_template('titanic_survival.html',pred='Sorry, Your Survival Chances are Low. Find a wooden log and hold on for dear life! \nProbability of survival is {}'.format(output))
    else:
        return render_template('titanic_survival.html',pred='Hurray! Your Survival Chances are High.\n Probability of survival is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)


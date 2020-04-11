from flask import Flask, request
import pandas as pd
import numpy as np
import json
import pickle
import os

app = Flask(__name__)

lr_model_path = os.path.join(os.path.pardir, 'models', 'lrModel.pkl') #'../models/lrModel.pkl'
model_pickle_r = open(lr_model_path, 'rb')
lrModelLoaded = pickle.load(model_pickle_r)

# columns
columns = [ u'Age', u'Fare', u'FamilySize', \
       u'IsMother', u'IsMale', u'Deck_A', u'Deck_B', u'Deck_C', u'Deck_D', \
       u'Deck_E', u'Deck_F', u'Deck_G', u'Deck_Z', u'Pclass_1', u'Pclass_2', \
       u'Pclass_3', u'Title_Lady', u'Title_Master', u'Title_Miss', u'Title_Mr', \
       u'Title_Mrs', u'Title_Officer', u'Title_Sir', u'Fare_Bin_very_low', \
       u'Fare_Bin_low', u'Fare_Bin_high', u'Fare_Bin_very_high', u'Embarked_C', \
       u'Embarked_Q', u'Embarked_S', u'AgeState_Adult', u'AgeState_Child'] 


@app.route('/api' , methods=['POST'])
def make_prediction():
    # read json object and convert to json string
    data = json.dumps(request.get_json(force=True))
    df = pd.read_json(data)
    #extract passenger ids
    passenger_ids = df['PassengerId'].ravel()
    survived = df['Survived'].ravel()
    X = df[columns].values().astype('float')
    predictions = lrModelLoaded.predict(X_scaled)
    
    df_response = pd.DataFrame({'PassengerId': passenger_ids, 'Predicted': predictions, 'Actual': survived})
    return df_response.to_json()

if __name__ == '__main__':
    app.run(port=10002, debug = True)
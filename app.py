from flask import Flask, render_template, jsonify
from flask import request
import joblib

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from math import sin, cos, sqrt, atan2, radians
import re
#from tensorflow import keras

app = Flask(__name__)

@app.route("/")
def home():
    
    return render_template("index.html")

@app.route("/",methods=['POST'])
def getvalues():
    bed = request.form['bed']
    full_bath = request.form['full_bath']
    half_bath = request.form['half_bath']
    property_area = request.form['property_area']
    years_old = request.form['years_old']
    distance_downtown = request.form['distance_downtown']
    lot_size = request.form['lot_size']
    option_basement = request.form['option_basement']
    option_garage = request.form['option_garage']
    #walk_score = request.form['walk_score']
    #bike_score = request.form['bike_score']
    #transit_score = request.form['transit_score']
    property_type = request.form['option_ptype']
    #house = request.form['house']
    #condo = request.form['condo']
    #townhouse = request.form['townhouse']
    postal_code = request.form['postal_code']
    #print(garage)

    if property_type == 'house':
        house = 1
        condo = 0
        townhouse = 0
    elif property_type == 'condo':
        house = 0
        condo = 1
        townhouse = 0
    else:
        house = 0
        condo = 0
        townhouse = 1

    if option_basement == "yes":
        basement = 1
    else:
        basement = 0

    if option_garage == "yes":
        garage = 1
    else:
        garage = 0

    
    postal_code = str(postal_code)


    ###### SCRAPE WALKSCORE ######

    scores_walk = []
    scores_bike = []
    scores_transit = []

    for i in postal_code:

        #time.sleep(5)
        
        try:
            postal_code_a = i.replace(" ", "%20")
            url_score = "https://www.walkscore.com/score/" + str(postal_code_a)
            #time.sleep(5)

            # Parse HTML with Beautiful Soup
            response = requests.get(url_score)
            code_soup = BeautifulSoup(response.text, 'html.parser')

            if 'pp.walk.sc/badge/walk/score' in str(code_soup):
                ws = str(code_soup).split('pp.walk.sc/badge/walk/score/')[1][:2].replace('.','')
                scores_walk.append(ws)
            else:
                ws = 0
                scores_walk.append(ws)
            if 'pp.walk.sc/badge/bike/score' in str(code_soup):
                bs = str(code_soup).split('pp.walk.sc/badge/bike/score/')[1][:2].replace('.','')
                scores_bike.append(bs)
            else:
                bs = 0
                scores_bike.append(bs)
            if 'pp.walk.sc/badge/transit/score' in str(code_soup):
                ts = str(code_soup).split('pp.walk.sc/badge/transit/score/')[1][:2].replace('.','')
                scores_transit.append(ts)
            else:
                ts = 0
                scores_transit.append(ts)
        except:
            ws = 0
            scores_walk.append(ws)
            bs = 0
            scores_bike.append(bs)
            ts = 0
            scores_transit.append(ts)

    ####### END #######

    scores_walk_num = scores_walk[0]
    scores_bike_num = scores_bike[0]
    scores_transit_num = scores_transit[0]

    # Convert to numeric
    bed = int(bed)
    full_bath = int(full_bath)
    half_bath = int(half_bath)
    property_area = float(property_area)
    years_old = int(years_old)
    distance_downtown = float(distance_downtown)
    lot_size = float(lot_size)
    basement = int(basement)
    garage = int(garage)
    walk_score = int(scores_walk_num)
    bike_score = int(scores_bike_num)
    transit_score = int(scores_transit_num)
    house = int(house)
    condo = int(condo)
    townhouse = int(townhouse)

    #print(postal_code)

    # Testing ML Model
    filename = '/data/LogisticRegression.sav'

    joblib_LR_model = joblib.load(filename)
    joblib_LR_model

    test_data = [[bed, full_bath, half_bath, property_area, years_old, distance_downtown, lot_size, basement, garage, walk_score, bike_score, transit_score, house, condo, townhouse]]

    Ypredict = joblib_LR_model.predict(test_data)

    #reconstructed_model = keras.models.load_model("data/my_h5_model.h5")

    #score = reconstructed_model.fit(test_data)

    return render_template("index.html", Ypredict=[Ypredict])

if __name__ == "__main__":
    app.run(debug=True)
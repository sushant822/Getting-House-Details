# Testing ML Model

import joblib

filename = '/data/LogisticRegression.sav'

joblib_LR_model = joblib.load(filename)
joblib_LR_model

bed = 3
full_bath = 2
half_bath = 1
property_area = 1650
years_old = 1
distance_downtown = 30
lot_size = 1800
basement = 1
garage = 0
walk_score = 35
bike_score = 60
transit_score = 50
house = 1
condo = 0
townhouse = 0

test_data = [[bed, full_bath, half_bath, property_area, years_old, distance_downtown, lot_size, basement, garage, walk_score, bike_score, transit_score, house, condo, townhouse]]

Ypredict = joblib_LR_model.predict(test_data)  

#Ypredict
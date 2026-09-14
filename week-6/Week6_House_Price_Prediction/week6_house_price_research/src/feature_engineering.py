
"""Domain-informed real-estate feature engineering."""
import numpy as np, pandas as pd
_COASTAL_REFERENCE_POINTS=[(32.72,-117.17),(33.77,-118.19),(34.42,-119.70),
                           (36.60,-121.90),(37.77,-122.42),(38.58,-123.28),(40.80,-124.16)]
def _distance_to_coast(lat,lon):
    return np.stack([np.sqrt((lat-a)**2+(lon-b)**2) for a,b in _COASTAL_REFERENCE_POINTS]).min(axis=0)
def engineer_features(X):
    X=X.copy()
    X["RoomsPerHousehold"]=X["AveRooms"]/X["AveOccup"].replace(0,np.nan)
    X["BedroomRatio"]=X["AveBedrms"]/X["AveRooms"].replace(0,np.nan)
    X["PopulationPerHousehold"]=X["Population"]/X["AveOccup"].replace(0,np.nan)
    X["DistanceToCoast"]=_distance_to_coast(X["Latitude"].values,X["Longitude"].values)
    X["IncomePerRoom"]=X["MedInc"]/X["AveRooms"].replace(0,np.nan)
    X["RoomsMinusBedrooms"]=X["AveRooms"]-X["AveBedrms"]
    X["IncomePerOccupant"]=X["MedInc"]/X["AveOccup"].replace(0,np.nan)
    X["RoomsSquared"]=X["AveRooms"]**2
    X["IncomeSquared"]=X["MedInc"]**2
    X["AgeSquared"]=X["HouseAge"]**2
    X["LatitudeLongitudeInteraction"]=X["Latitude"]*X["Longitude"]
    X["IncomeRoomsInteraction"]=X["MedInc"]*X["AveRooms"]
    return X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True))

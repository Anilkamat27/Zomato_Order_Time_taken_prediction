import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from datetime import datetime
import pandas as pd
from src.utils import distcalculate

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Delivery_person_Age: float,
                 Delivery_person_Ratings: float,
                 Restaurant_latitude: float,
                 Restaurant_longitude: float,
                 Delivery_location_latitude: float,
                 Delivery_location_longitude: float,
                 Order_Date: str,
                 Time_Orderd:str,
                 Time_Order_picked:str,
                 Weather_conditions: str,
                 Road_traffic_density: str,
                 Vehicle_condition: int,  # Adding Vehicle_condition column (assuming it's a categorical variable)  # Adding Type_of_order column (assuming it's a categorical variable)
                 Type_of_vehicle: str,
                 multiple_deliveries: float,  # Adding multiple_deliveries column (assuming it's a categorical variable)
                 Festival: str,
                 City: str):  # Adding Picked_Min column (assuming it's an integer)):  # Adding Time_taken column
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Restaurant_latitude = Restaurant_latitude
        self.Restaurant_longitude = Restaurant_longitude
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Order_Date = Order_Date
        self.Time_Orderd = Time_Orderd
        self.Time_Order_picked= Time_Order_picked
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City
        self.Order_prepare_time = None
        self.Order_Day = None
        self.Order_Month = None
        self.Order_Year = None
        self.Order_Hour = None
        self.Order_Min = None
        self.Picked_Hour = None
        self.Picked_Min = None

    def calculate_attributes(self):
        # Convert the time strings to datetime objects
        order_time = datetime.strptime(self.Time_Orderd, '%H:%M')
        picked_time = datetime.strptime(self.Time_Order_picked, '%H:%M')

        # Calculate the Order_prepare_time in minutes
        self.Order_prepare_time = (picked_time - order_time).total_seconds() // 60

        # Extract day, month, and year from the Order_Date
        order_date_obj = datetime.strptime(self.Order_Date, '%d-%m-%Y')
        self.Order_Day = order_date_obj.day
        self.Order_Month = order_date_obj.month
        self.Order_Year = order_date_obj.year

        # Extract hour and minute from the order_time and picked_time
        self.Order_Hour = order_time.hour
        self.Order_Min = order_time.minute
        self.Picked_Hour = picked_time.hour
        self.Picked_Min = picked_time.minute
    def get_data_as_dataframe(self):
        try:
            distance=distcalculate(self.Restaurant_latitude, self.Restaurant_longitude, self.Delivery_location_latitude, self.Delivery_location_longitude)
            self.calculate_attributes()

            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age], 
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Restaurant_latitude':[self.Restaurant_latitude],
                'Restaurant_longitude':[self.Restaurant_longitude],
                'Delivery_location_latitude':[self.Delivery_location_latitude],
                'Delivery_location_longitude':[self.Delivery_location_longitude],
                'Vehicle_condition':[self.Vehicle_condition],
                'multiple_deliveries':[self.multiple_deliveries], 
                'Distance':[distance],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density], 
                'Type_of_vehicle':[self.Type_of_vehicle], 
                'Festival':[self.Festival], 
                'City':[self.City],
                'Order_prepare_time':[self.Order_prepare_time],
                'Order_Day':[self.Order_Day],
                'Order_Month':[self.Order_Month],
                'Order_Year':[self.Order_Year],
                'Order_Hour':[self.Order_Hour],
                'Order_Min':[self.Order_Min],
                'Picked_Hour':[self.Picked_Hour],
                'Picked_Min':[self.Picked_Min]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
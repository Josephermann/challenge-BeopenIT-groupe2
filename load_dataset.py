import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime, date
import logging
logging.basicConfig(level=logging.INFO)

def load_dataset(path1,path2, sep, test_size=0.2, random_state=123):
	data_report = pd.read_csv(path1, sep=sep)
	data_meteo = pd.read_csv(path2, sep=sep)
	data_report['DATE_int']=data_report['YEAR'].astype(str) + data_report['MONTH'].astype(str)+ data_report['DAY_OF_MONTH'].astype(str)
	data_report['DATE'] = pd.to_datetime(data_report['DATE_int'], format='%Y%m%d')
	logging.info("Deleted corrolated cols and cols with unique values...")
	cols_to_del = ['YEAR', 'QUARTER', 'MONTH','OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME','ORIGIN_STATE_ABR',
	'ORIGIN_STATE_NM','DEST','DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 'DEP_TIME',
	'DEP_DELAY','DEP_DELAY_NEW', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT','CRS_ARR_TIME','ARR_DELAY',
	'ARR_DELAY_NEW', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'DIVERTED', 'FLIGHTS','DISTANCE','DISTANCE_GROUP',
	'NAS_DELAY', 'Unnamed: 49','DATE_int', 'CANCELLED']
	logging.info("Handle Na's values...")
	cols_with_most_Na = ["LATE_AIRCRAFT_DELAY", "SECURITY_DELAY","WEATHER_DELAY","CARRIER_DELAY"]
	data_report.drop(cols_to_del, axis=1, inplace=True)
	data_report.drop(cols_with_most_Na, axis=1, inplace=True)
	data_report.dropna(how='any', subset=['ARR_DEL15'],inplace=True)
	logging.info("Change cols types...")
	data_report.DEST_STATE_NM = pd.Categorical(data_report.DAY_OF_MONTH)
	data_report.DEST_STATE_NM.cat.codes
	data_report.DAY_OF_WEEK = pd.Categorical(data_report.DAY_OF_WEEK)
	data_report.OP_CARRIER_FL_NUM = pd.Categorical(data_report.OP_CARRIER_FL_NUM)
	data_report.ORIGIN_AIRPORT_ID= pd.Categorical(data_report.ORIGIN_AIRPORT_ID)
	data_report.ORIGIN_WAC= pd.Categorical(data_report.ORIGIN_WAC)
	data_report.DEST_AIRPORT_ID= pd.Categorical(data_report.DEST_AIRPORT_ID)
	data_report.DEP_DEL15 = pd.Categorical(data_report.DEP_DEL15)
	data_report.ARR_DEL15 = pd.Categorical(data_report.ARR_DEL15)
	logging.info("Selection of cols to final data to pass to model...")
	quantisup_2 = ['DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID',
	       'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_STATE_NM', 'DEST_WAC', 'TAXI_IN',
	       'ARR_TIME', 'ARR_DEL15', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME',
	       'AIR_TIME', 'AWND', 'PRCP', 'SNOW', 'TMAX', 'TMIN', 'DEP_DEL15']
	logging.info("Merging of meteo data frame and flight data frame...")
	meteo_cols_delete = ['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'TAVG']
	data_meteo.drop(meteo_cols_delete, axis=1, inplace=True)
	data_meteo['DATE'] = data_meteo['DATE'].astype('datetime64[ns]')
	merge_data =pd.merge(data_report,data_meteo, how='left', on='DATE')
	data = merge_data.drop('DATE',axis=1)
	x = data.drop(['ARR_DEL15'], axis=1)
	y = data['ARR_DEL15']
	logging.info("Log dataset parameters on mlflow server...")
	mlflow.log_param("dataset1_path", path1)
	mlflow.log_param("dataset2_path", path2)
	mlflow.log_param("dataset_shape", data.shape)
	mlflow.log_param("test_size", test_size)
	mlflow.log_param("random_state", random_state)
	return train_test_split(x, y, test_size=test_size, random_state=random_state)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset(
        './data/488753924_T_ONTIME_REPORTING.csv', './data/2309682.csv',','
    )
    print(x_train.head())
    print(y_train.head())

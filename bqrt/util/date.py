import pandas as pd
import datetime

def weeks_of_month_table(year:int,num_of_week:int,num_of_weekday:int) -> pd.DataFrame:
	"""Generate a pd.DataFrame which 'date' column contains all nth week, nth weekday of certain year

	Parameters
	----------
	year : int
		year to lookup
	num_of_week : int
		nth week of a month
	num_of_weekday : int
		1: Monday, 2: Tuesday and so on

	Returns
	-------
	pd.DataFrame
		a DataFrame contains all information
	"""
	raw_df = pd.DataFrame({'date':pd.date_range(datetime.date(year,1,1),datetime.date(year,12,31))})
	raw_df['month'] = raw_df['date'].dt.month
	raw_df['week'] = raw_df['date'].dt.isocalendar().week
	raw_df['weekday'] = raw_df['date'].dt.isocalendar().day
	df = raw_df.groupby(['month','weekday'],as_index=False).nth(num_of_week-1)
	df = df[df['weekday'] == num_of_weekday].reset_index(drop=True)
	return df

def weeks_of_month(date:datetime.datetime,num_of_week:int,num_of_weekday:int) -> bool:
	"""Return bool result of certain date is nth week of that month, nth weekday of that week

	Parameters
	----------
	date : datetime.datetime
		date to check
	num_of_week : int
		nth week of a month
	num_of_weekday : int
		1: Monday, 2: Tuesday and so on

	Returns
	-------
	bool
		True or False of result
	"""
	datelist = weeks_of_month_table(date.year,num_of_week,num_of_weekday)['date'].to_list()
	if date in datelist:
		return True
	else:
		return False








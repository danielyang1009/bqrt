import pandas as pd
import datetime

def weeks_of_month_table(year:int,num_of_week:int,num_of_weekday:int) -> pd.DataFrame:
	
	raw_df = pd.DataFrame({'date':pd.date_range(datetime.date(year,1,1),datetime.date(year,12,31))})
	raw_df['month'] = raw_df['date'].dt.month
	raw_df['week'] = raw_df['date'].dt.isocalendar().week
	raw_df['weekday'] = raw_df['date'].dt.isocalendar().day

	df = raw_df.groupby(['month','weekday'],as_index=False).nth(num_of_week-1)
	df = df[df['weekday'] == num_of_weekday].reset_index(drop=True)

	return df






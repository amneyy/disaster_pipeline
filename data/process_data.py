import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
	"""The load_data function read data from 2 files and merge them

	Parameters:
	messages_filepath (string): Filepath of the messages csv
	categories_filepath (string): Filepath of the categories csv
	
	Returns:
	dataframe: returns the merged dataframe

   """
	
	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)
	
	df = pd.merge(messages, categories, how='inner', on='id')
	
	return df
  
def clean_data(df):
	
	"""The clean function splits categories into separate category column, convert categories values into 0 or 1 and remove duplicates

	Parameters:
	df (dataframe)

	
	Returns:
	dataframe: returns the cleaned dataframe

   """
	
	cat_df = df['categories'].str.split(';', expand=True)
	
	row = df['categories'][0].split(';')
	category_colnames = [w[:-2] for w in row]
	
	#changing cat_df column names 
	cat_df.columns = category_colnames
	
	for column in cat_df:
		cat_df[column] = cat_df[column].apply(lambda x: x[-1])
		cat_df[column] = cat_df[column].apply(lambda x: pd.to_numeric(x))
	
	#dropping the categories column from df and concat the cat_df dataframe
	df = df.drop('categories', axis=1)
	df = pd.concat([df, cat_df], axis=1)
	
	#removing duplicates
	df = df[df.duplicated()==False]
	#removing rows containing value different than 0 or 1
	df = df[df['related'].isin([0,1])]
	
	return df

def save_data(df, database_filename):
	"""The save_data function saves the dataframe into a database called database_filename

	Parameters:
	df (dataframe) : dataframe to save
	database_filename (db): database to save the dataframe

	
	Returns:
	Saves the dataframe df to the db database_filename

   """
	engine = create_engine('sqlite:///{}'.format(database_filename))
	df.to_sql('my_db', engine, if_exists='replace', index=False)


def main():
	"""The main function will load, clean and save the data into a databse

	Parameters:
	No parameters

	Returns: Load, clean and save data
   """

	if len(sys.argv) == 4:

		messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
			  .format(messages_filepath, categories_filepath))
		df = load_data(messages_filepath, categories_filepath)

		print('Cleaning data...')
		df = clean_data(df)
		
		print('Saving data...\n    DATABASE: {}'.format(database_filepath))
		save_data(df, database_filepath)
		
		print('Cleaned data saved to database!')
	
	else:
		print('Please provide the filepaths of the messages and categories '\
			  'datasets as the first and second argument respectively, as '\
			  'well as the filepath of the database to save the cleaned data '\
			  'to as the third argument. \n\nExample: python process_data.py '\
			  'disaster_messages.csv disaster_categories.csv '\
			  'DisasterResponse.db')


if __name__ == '__main__':
	main()
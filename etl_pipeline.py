def clean_upload(df1, df2)

 ''''
 the function blah blah
    '''
#import
    import sqlite3
    import pandas as pd
    from sqlalchemy import create_engine

# load and merge datasets
    df1 = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')
    df = messages.merge(categories)

# change category column names to type of message
    categories = df.categories.str.split(';', expand=True)
    categories.head()
    row = categories.iloc[0]
    category_colnames = row.str.slice(0, - 2)
    categories.columns = category_colnames

#convert category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

#drop the original categorise column and concatenate dataframes
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis = 1)

#remove duplicated data
    df = df.drop_duplicates()

#save dataset to sqlite database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('disaster_data', engine, index=False)

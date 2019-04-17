import pandas as pd
import country_code_dict
import us_state_abbrev

################################################################################

def GetPermData(years):
    """Reads in an excel file of Permenant Labour Certification data, cleans
    headers for consistency and puts into a dataframe format.
    If multiple files are read in then dataframes created are concatnated into one.
    """

    location = '/Users/laurengilson/Desktop/perm_data/PERM_FY{}.xlsx'
    dfs = []

    for year in years:

        # Read in excel for particular year
        read_data = pd.read_excel(location.format(year))

        # clean each column for consistency, lower case and underscores
        read_data.columns = list(map(lambda header : header.lower(), read_data.columns))
        read_data.columns = read_data.columns.str.replace(' ','_')

        # add column of fiscal year to df - use later for merging economic data
        read_data['fiscal_year'] = str(year)
        dfs.append(read_data)

    return pd.concat(dfs, axis=0, ignore_index=True)

################################################################################

def Predictor(outcome):
    """Classes case outcome into 1 for denied, 0 for certified"""
    if outcome == 'denied':
        return 1
    else:
        return 0

################################################################################

def CountryCode(key):
    """Finds 3 letter country code identifier from full country input
    country_code_dict is a dictionary where keys are the full country name and
    values are the 3 letter code

    Example: country_code_dict['Brazil'] = 'BRA'
    """
    return country_code_dict[key]

################################################################################

def ColValueCheck(col_1, col_2):

    '''Creates a new column from values of two other columns
    dependent on whether columns have the same value.
    1 = Yes 0 = No

    Inputs: 2 columns from dataframe

    Assumptions: Assumes if there is a NaN value, it would be the same
    as the other column value'''

    col_1 = col_1.fillna('unknown')
    col_2 = col_2.fillna('unknown')

    new_col = []

    for idx, value in enumerate(col_1):
        if col_1[idx] == 'unknown' or col_2[idx] == 'unknown': # if unknown assume they're the same
            new_col.append(1)
        elif col_1[idx] == col_2[idx]:
            new_col.append(1)
        else:
            new_col.append(0)

    return new_col

################################################################################

def WageFunction(col1, col2):
    """Changes all wage requests into salary format. Takes in two columns of wage and
    unit of pay then changes wage accordingly"""


    # Clean numeric column to return just floats and zeros
    col1 = col1.fillna(0)
    col1 = list(map(str, col1)) # In order remove non numeric characters
    col1 = [x.replace(',', '') for x in col1]
    col1 = [x.replace('#', '0') for x in col1]
    col1 = list(map(float, col1))

    # Clean unit of pay column and reduce to one letter for consistency
    col2 = col2.fillna(0)
    col2 = list(map(str, col2))
    col2 = list(map(lambda x:x.lower(), col2))
    col2 = [i[0] for i in col2] # Reduce to one letter to get unit of pay
    col2 = list(map(str, col2))

    # Create tuples showing what the wage is and the unit
    tups = list(zip(col1, col2))

    total_salary = []
    av_us_salary = 53039 # Assume average US salary for NaN or 0 values


    for idx, value in enumerate(tups):
        if tups[idx][1] == 'h': # Hourly - 2080 worked hours per year
            total_salary.append((tups[idx][0]*2080))
        elif tups[idx][1] == 'w': # Weekly
            total_salary.append((tups[idx][0]*52))
        elif tups[idx][1] == 'b': # Bi-Weekly
            total_salary.append((tups[idx][0]*26))
        elif tups[idx][1] == 'm': # Monthly
            total_salary.append((tups[idx][0]*12))
        elif tups[idx][1] == 'y': # Yearly
            total_salary.append(tups[idx][0])
        else:
            total_salary.append(av_us_salary) # Assume nulls have average salary

    rounded_list = [ '%.2f' % elem for elem in total_salary] # round each to just 2dp
    rounded_list = list(map(float, rounded_list))
    rounded_list = [av_us_salary if x == 0 else x for x in rounded_list] # Assume zeros have av salary

    return rounded_list

################################################################################

def CategoriseYN(outcome):
    ''' Classes case outcome into 1 for yes, 0 for no'''
    if outcome == 'y':
        return 1
    else:
        return 0

################################################################################

def StringColumns(column, new_null='n'):
    """Cleans columns of strings, makes lowercase and assumes null values as to input.
    Default new_null is n (no) if not defined.
    Returns cleaned column"""

    column = list(map(str, column))
    column = list(map(lambda x:x.lower(), column))
    column = [x.replace('nan', new_null) for x in column] # Assume nans as n

    return column

################################################################################

def StateAbbreviation(state_col):

    '''Abbreviates any full length state names into their corresponding
    two letter code'''

    state_col = state_col.fillna(str('none'))
    state_col = list(map(str, state_col))
    state_col = list(map(lambda state:state.lower(), state_col))

    abbrev_list = []

    for item in state_col:

        if item in us_state_abbrev:
            abbrev_list.append(us_state_abbrev[item].upper())
        else:
            abbrev_list.append(item.upper())

    return abbrev_list

################################################################################

def CleanEconomic(data):
    """Reads in csv file of economic data, removes unnecessary columns. Formats each column dependent
    on data name and returns a transformed dataframe of cleaned columns"""

    # Read data
    economic_data = pd.read_csv(f'/Users/laurengilson/Desktop/perm_data/{data}.csv')

    # remove last 5 rows - no data in these
    economic_data = economic_data[:-5]

    # drop unnecessary columns - just want the country code, year and economic factor
    economic_data.drop(['Series Name', 'Series Code', 'Country Name'], axis=1, inplace=True)

    # Turn to strings and replace '..' with '0'
    for column in list(economic_data.columns):
        economic_data[column] = list(map(str, economic_data[column]))
        economic_data[column] = [x.replace('..', '0') for x in economic_data[column]]

    if data == 'percentage_of_immigrants':

        economic_data['2007 [YR2007]'] = economic_data['2005 [YR2005]']
        economic_data.drop(['2005 [YR2005]', '2006 [YR2006]'], axis=1, inplace=True)

    elif data == 'net_migration':

        for idx, column in enumerate(list(economic_data.columns)[2:]):
            economic_data[column] = economic_data[column].astype(float)
            economic_data[column] = economic_data[column].replace(to_replace=0, value=economic_data.iloc[-1][column])
            economic_data[column] = economic_data[column].replace(0, np.nan) # replace 0 with NaNs
            economic_data.loc[economic_data[column].isnull(), column] = economic_data.iloc[:,economic_data.columns.get_loc(column)-1] # rest of nans with value from prev column
            economic_data[column] = economic_data[column].astype(float)

        economic_data.drop(['2002 [YR2002]', '2003 [YR2003]', '2004 [YR2004]', '2005 [YR2005]', '2006 [YR2006]'], axis=1, inplace=True)

    else:
        None

    # dict of new columns to replace current column headers - these will match the fiscal yeard in PERM data
    years = list(range(2008, 2020))
    col_list = list(economic_data.columns[1:])
    new_cols = dict(zip(col_list, years))

    # Change column head to fiscal years
    economic_data.rename(columns=new_cols, inplace=True)


    if data == 'GDP_data':

        # Change 2019 - to have 3.1% growth on 2018.
        economic_data['2019'] = economic_data['2018'].astype(float) * 1.031

        # Put data in billions & round to 2dp
        for column in list(economic_data.columns)[1:]:
            economic_data[column] = economic_data[column].astype(float)/1000000000 # Put in terms of billions
            economic_data[column] = economic_data[column].apply(lambda x: round(x, 2))

        # Transform df
        economic_data = pd.melt(economic_data, id_vars=['Country Code'], var_name='year', value_name='gdp')

        return economic_data

    elif data == 'employment_to_pop_ratio':

        for column in list(economic_data.columns)[1:]:
            economic_data[column] = economic_data[column].astype(float)
            economic_data[column] = economic_data[column].replace(to_replace=0, value=economic_data[column].mean())
            economic_data[column] = economic_data[column].apply(lambda x: round(x, 2))

        # Transform df
        economic_data = pd.melt(economic_data, id_vars=['Country Code'], var_name='year', value_name='employment')

        return economic_data

    elif data == 'gov_expenditure_education' or data == 'percentage_of_immigrants':

        for idx, column in enumerate(list(economic_data.columns)[1:]):
            economic_data[column] = economic_data[column].astype(float)
            economic_data[column] = economic_data[column].replace(to_replace=0, value=economic_data.iloc[-1][column]) # Replace zero with world average for that year
            economic_data[column] = economic_data[column].replace(0, np.nan) # replace 0 with NaNs
            economic_data.loc[economic_data[column].isnull(), column] = economic_data.iloc[:,economic_data.columns.get_loc(column)-1] # rest of nans with value from prev column
            economic_data[column] = economic_data[column].apply(lambda x: round(x, 2))

        # Transform df
        if data == 'gov_expenditure_education':
            economic_data = pd.melt(economic_data, id_vars=['Country Code'], var_name='year', value_name='gov_expen')
            return economic_data
        else:
            economic_data = pd.melt(economic_data, id_vars=['Country Code'], var_name='year', value_name='percentage_of_immigrants')
            return economic_data

    else:
         # Transform df
        economic_data = pd.melt(economic_data, id_vars=['Country Code'], var_name='year', value_name='net_migration')
        return economic_data

################################################################################

def PoliticalParty(date):
    """Returns which political party was in office dependent on the date
    Returns 1 for Republican party and 0 for Democratic"""

    democratic_start = 200902
    democratic_end = 201701
    if democratic_start <= date <= democratic_end:
        return 0
    else:
        return 1

################################################################################

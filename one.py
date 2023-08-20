#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nunoo Emmanuel Felix Landlord
"""

import pandas as pd

# import dataset
file_path = 'glassdoor_jobs.csv'  
df = pd.read_csv(file_path)

# DATA CLEANING

# salary parsing 
# cleaning salary estimate cloumn into a new dataframe

# remove -1 values i  salary estimate
df = df[df['Salary Estimate'] != '-1']

# remove glassdor est. text by using lambda func 
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])

# remove K's and $ signs
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

# create new column for per hourly and employer provided [in the main df]
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

# replace  per hourly and employer provided salary with the new column above
min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

# create new columns for min,max,avg salary [in the main df]
df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2


# company name text only
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

# state only
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])

# count number of jobs by state
df.job_state.value_counts()

# check if HQ is in the same state...if TRUE 1 if FALSE 0
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

# age of company
df['age'] = df.Founded.apply(lambda x: x if x <1 else 2023 - x)


#njob description parsing (python, etc.)
df['Job Description'][0]

#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
 
#r studio 
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df.R_yn.value_counts()

#spark 
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark.value_counts()

#aws 
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws.value_counts()

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel.value_counts()

# show column names
df.columns

# drop the ['Unnamed: 0'] column
df_out = df.drop(['Unnamed: 0'], axis =1)

#save new csv
df_out.to_csv('salary_data_cleaned.csv',index = False)

pd.read_csv('salary_data_cleaned.csv')

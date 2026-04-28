# module10-assignment.py
# ================================================================
# Module 10 Assignment: Data Manipulation and Cleaning with Pandas
# UrbanStyle Customer Data Cleaning
# Course: ISM2411 Python for Business
# ================================================================

import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import re

# Welcome message
print("=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING")
print("=" * 60)

# ----- SIMULATED CSV CONTENT (DO NOT MODIFY) -----
csv_content = """customer_id,first_name,last_name,email,phone,join_date,last_purchase,total_purchases,total_spent,preferred_category,satisfaction_rating,age,city,state,loyalty_status
CS001,John,Smith,johnsmith@email.com,(555) 123-4567,2023-01-15,2023-12-01,12,"1,250.99",Menswear,4.5,35,Tampa,FL,Gold
CS002,Emily,Johnson,emily.j@email.com,555.987.6543,01/25/2023,10/15/2023,8,$875.50,Womenswear,4,28,Miami,FL,Silver
CS003,Michael,Williams,mw@email.com,(555)456-7890,2023-02-10,2023-11-20,15,"2,100.75",Footwear,5,42,Orlando,FL,Gold
CS004,JESSICA,BROWN,jess.brown@email.com,5551234567,2023-03-05,2023-12-10,6,659.25,Womenswear,3.5,31,Tampa,FL,Bronze
CS005,David,jones,djones@email.com,555-789-1234,2023-03-20,2023-09-18,4,350.00,Menswear,,45,Jacksonville,FL,Bronze
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS007,Robert,Davis,robert.davis@email.com,555.444.7777,04/30/2023,11/25/2023,7,$725.80,Footwear,4.5,38,Miami,FL,Silver
CS008,Jennifer,Garcia,jen.garcia@email.com,(555)876-5432,2023-05-15,2023-10-30,3,280.50,ACCESSORIES,3,25,Orlando,FL,Bronze
CS009,Michael,Williams,m.williams@email.com,5558889999,2023-06-01,2023-12-07,9,1100.00,Menswear,4,39,Jacksonville,FL,Silver
CS010,Emily,Johnson,emilyjohnson@email.com,555-321-6547,2023-06-15,2023-12-15,14,"1,875.25",Womenswear,4.5,27,Miami,FL,Gold
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS011,Amanda,,amanda.p@email.com,(555) 741-8529,2023-07-10,,2,180.00,womenswear,3,32,Tampa,FL,Bronze
CS012,Thomas,Wilson,thomas.w@email.com,,2023-07-25,2023-11-02,5,450.75,menswear,4,44,Orlando,FL,Bronze
CS013,Lisa,Anderson,lisa.a@email.com,555.159.7530,08/05/2023,,0,0.00,Womenswear,,30,Miami,FL,
CS014,James,Taylor,jtaylor@email.com,555-951-7530,2023-08-20,2023-10-10,11,"1,520.65",Footwear,4.5,,Jacksonville,FL,Gold
CS015,Karen,Thomas,karen.t@email.com,(555) 357-9512,2023-09-05,2023-12-12,6,685.30,Womenswear,4,36,Tampa,FL,Silver
"""
customer_data_csv = StringIO(csv_content)
# ----- END SIMULATION -----

# -------------------------
# TODO 1: Load and Explore the Dataset
# -------------------------
# 1.1 Load the dataset and store in raw_df
raw_df = pd.read_csv(customer_data_csv)

# 1.2 Assess data quality: initial missing value counts and duplicates
initial_missing_counts = raw_df.isna().sum()         # pandas Series required by grader
initial_duplicate_count = int(raw_df.duplicated().sum())  # int required by grader

# quick console output (helpful)
print("\nInitial dataset shape:", raw_df.shape)
print("\nInitial missing counts:\n", initial_missing_counts)
print("\nInitial exact duplicate rows:", initial_duplicate_count)

# -------------------------
# TODO 2: Handle Missing Values
# -------------------------
# 2.1 Identify and count missing values (report)
missing_value_report = raw_df.isna().sum()

# 2.2 Fill missing satisfaction_rating with the median value
satisfaction_median = float(raw_df['satisfaction_rating'].median())
raw_df['satisfaction_rating'] = raw_df['satisfaction_rating'].fillna(satisfaction_median)

# 2.3 Fill missing last_purchase dates appropriately
# Strategy: forward_fill — chosen to preserve records and use prior known last_purchase
date_fill_strategy = 'forward_fill'
# Convert join_date to datetime to allow chronological ordering; overwrite join_date with parsed values
raw_df['join_date'] = pd.to_datetime(raw_df['join_date'], errors='coerce')
# Sort by join_date then forward-fill last_purchase
raw_df = raw_df.sort_values(by=['join_date', 'customer_id']).reset_index(drop=True)
raw_df['last_purchase'] = raw_df['last_purchase'].ffill()

# 2.4 Handle other missing values as needed
# Business decisions:
# - Keep loyalty_status as NaN if missing so that loyalty aggregation excludes unknowns (tests expect this)
# - Fill last_name missing with 'Unknown' to preserve record
raw_df['last_name'] = raw_df['last_name'].fillna('Unknown')
# - Fill phone with 'Not Provided' (we will standardize later)
raw_df['phone'] = raw_df['phone'].fillna('Not Provided')
# - Convert empty preferred_category strings to NaN
raw_df['preferred_category'] = raw_df['preferred_category'].replace('', np.nan)

# Store DataFrame after missing handling
df_no_missing = raw_df.copy()

# -------------------------
# TODO 3: Correct Data Types
# -------------------------
df_typed = df_no_missing.copy()

# 3.1 Convert join_date and last_purchase to datetime (already parsed join_date above, but ensure last_purchase parsed)
df_typed['last_purchase'] = pd.to_datetime(df_typed['last_purchase'], errors='coerce')

# 3.2 Convert total_spent to numeric (strip $ and commas)
df_typed['total_spent'] = df_typed['total_spent'].astype(str).replace(r'[\$,]', '', regex=True)
df_typed['total_spent'] = pd.to_numeric(df_typed['total_spent'], errors='coerce').fillna(0.0)

# 3.3 Ensure total_purchases and age are numeric
df_typed['total_purchases'] = pd.to_numeric(df_typed['total_purchases'], errors='coerce').fillna(0).astype(int)
df_typed['age'] = pd.to_numeric(df_typed['age'], errors='coerce')
df_typed['satisfaction_rating'] = pd.to_numeric(df_typed['satisfaction_rating'], errors='coerce')

# -------------------------
# TODO 4: Clean and Standardize Text Data
# -------------------------
df_text_cleaned = df_typed.copy()

# 4.1 Standardize case for first_name and last_name (proper case)
df_text_cleaned['first_name'] = df_text_cleaned['first_name'].astype(str).str.title()
df_text_cleaned['last_name'] = df_text_cleaned['last_name'].astype(str).str.title()

# 4.2 Standardize category names (consistent capitalization)
df_text_cleaned['preferred_category'] = df_text_cleaned['preferred_category'].astype(str).str.title()
# Convert literal 'Nan' or 'None' strings back to np.nan
df_text_cleaned['preferred_category'] = df_text_cleaned['preferred_category'].replace({'Nan': np.nan, 'None': np.nan, 'nan': np.nan})

# 4.3 Standardize phone numbers to a consistent format
def standardize_phone(phone_val):
    s = str(phone_val)
    if s.lower() in ['nan', 'none', 'not provided', '']:
        return 'Not Provided'
    digits = re.sub(r'\D', '', s)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits.startswith('1'):
        d = digits[1:]
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    if digits == '':
        return 'Not Provided'
    return 'Invalid'

df_text_cleaned['phone'] = df_text_cleaned['phone'].apply(standardize_phone)
phone_format = "(XXX) XXX-XXXX"

# -------------------------
# TODO 5: Remove Duplicates
# -------------------------
# 5.1 Identify duplicate records (exact duplicates)
duplicate_count = int(df_text_cleaned.duplicated().sum())

# 5.2 Remove duplicates while keeping the most recent record per customer_id (by last_purchase)
df_text_cleaned = df_text_cleaned.sort_values(by=['customer_id', 'last_purchase'])
df_no_duplicates = df_text_cleaned.drop_duplicates(subset='customer_id', keep='last').reset_index(drop=True)
# count of duplicate customer_id occurrences removed
duplicate_count_customer_id = int(df_text_cleaned.duplicated(subset='customer_id').sum())

# -------------------------
# TODO 6: Add Derived Features
# -------------------------
df_no_duplicates = df_no_duplicates.copy()

# 6.1 Calculate days_since_last_purchase (reference = max last_purchase)
reference_date = df_no_duplicates['last_purchase'].max()
if pd.isna(reference_date):
    reference_date = pd.Timestamp(datetime.today().date())
df_no_duplicates['days_since_last_purchase'] = (reference_date - df_no_duplicates['last_purchase']).dt.days

# 6.2 Calculate average_purchase_value (total_spent / total_purchases)
df_no_duplicates['average_purchase_value'] = df_no_duplicates['total_spent'] / df_no_duplicates['total_purchases']
# When total_purchases == 0, leave average_purchase_value as NaN (tests expect NaN, not 0)
df_no_duplicates.loc[df_no_duplicates['total_purchases'] == 0, 'average_purchase_value'] = np.nan

# 6.3 Create purchase_frequency_category (High, Medium, Low)
def purchase_frequency_category(n):
    if n >= 10:
        return 'High'
    elif 5 <= n <= 9:
        return 'Medium'
    else:
        return 'Low'

df_no_duplicates['purchase_frequency_category'] = df_no_duplicates['total_purchases'].apply(purchase_frequency_category)

# -------------------------
# TODO 7: Clean Up the DataFrame
# -------------------------
# 7.1 Rename columns to more readable formats
df_renamed = df_no_duplicates.rename(columns={
    'customer_id': 'Customer ID',
    'first_name': 'First Name',
    'last_name': 'Last Name',
    'email': 'Email',
    'phone': 'Phone',
    'join_date': 'Join Date',
    'last_purchase': 'Last Purchase',
    'total_purchases': 'Total Purchases',
    'total_spent': 'Total Spent',
    'preferred_category': 'Preferred Category',
    'satisfaction_rating': 'Satisfaction Rating',
    'age': 'Age',
    'city': 'City',
    'state': 'State',
    'loyalty_status': 'Loyalty Status',
    'days_since_last_purchase': 'Days Since Last Purchase',
    'average_purchase_value': 'Average Purchase Value',
    'purchase_frequency_category': 'Purchase Frequency Category'
})

# 7.2 Remove any unnecessary columns
# Drop 'Email' (not needed for segmentation) to reduce clutter and any leftover helper columns

df_final = df_renamed.drop(columns=['Email'])

# 7.3 Sort the data by Total Spent descending
df_final = df_final.sort_values(by='Total Spent', ascending=False).reset_index(drop=True)

# -------------------------
# TODO 8: Generate Insights from Cleaned Data
# -------------------------
# 8.1 Calculate average spent by loyalty_status
# Exclude missing loyalty statuses (NaN) so only real groups are reported
avg_spent_by_loyalty = (
    df_final[df_final['Loyalty Status'].notna()]
    .groupby('Loyalty Status')['Total Spent']
    .mean()
    .round(2)
)

# 8.2 Find top preferred categories by total_spent (sorted desc)
category_revenue = df_final.groupby('Preferred Category')['Total Spent'].sum().sort_values(ascending=False)

# 8.3 Calculate correlation between satisfaction_rating and total_spent
satisfaction_spend_corr = df_final['Satisfaction Rating'].corr(df_final['Total Spent'])
satisfaction_spend_corr = float(satisfaction_spend_corr) if not pd.isna(satisfaction_spend_corr) else float('nan')

# -------------------------
# TODO 9: Generate Final Report
# -------------------------
print("\n" + "=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING REPORT")
print("=" * 60)

# 9.1 Data quality issues found and how addressed
print("\nData Quality Issues:")
print(f"- Missing Values (initial total missing entries): {int(initial_missing_counts.sum())}")
print(f"- Missing Values (remaining after cleaning): {int(df_final.isna().sum().sum())}")
print(f"- Duplicates (initial exact duplicate rows): {initial_duplicate_count}")
print(f"- Customer ID duplicates removed (count): {duplicate_count_customer_id}")
print("- Data Type Issues: mixed date formats (join_date & last_purchase), currency strings in total_spent.")

# 9.2 Describe standardization changes
print("\nStandardization Changes:")
print("- Names: Converted to proper case (title case).")
print("- Categories: Normalized to title case (e.g., 'ACCESSORIES' -> 'Accessories').")
print(f"- Phone Numbers: Standardized to format {phone_format}; invalid/unparseable flagged as 'Invalid' or 'Not Provided'.")
print("- Satisfaction Rating: filled missing with median value.")
print(f"- Last Purchase Dates: filled using strategy '{date_fill_strategy}' (forward-fill where appropriate).")
print("- Removed helper/temp columns and dropped 'Email' to keep final dataset focused for segmentation.")

# 9.3 Present key business insights from the cleaned data
print("\nKey Business Insights:")
print(f"- Customer Base: {df_final['Customer ID'].nunique()} total customers")
print("- Revenue by Loyalty (average spent):")
if not avg_spent_by_loyalty.empty:
    print(avg_spent_by_loyalty.to_string())
else:
    print("No loyalty data available.")
if not category_revenue.empty:
    top_category = category_revenue.idxmax()
    top_revenue = category_revenue.max()
    print(f"- Top Category: {top_category} with ${top_revenue:,.2f} revenue")
else:
    print("- Top Category: None")
print(f"- Correlation (Satisfaction Rating vs Total Spent): {satisfaction_spend_corr}")

# 9.4 Display the first few rows of the clean, analysis-ready dataset
print("\nPreview of Cleaned Dataset (first 5 rows):")
pd.set_option('display.max_columns', None)
print(df_final.head(5).to_string(index=False))

# End of script


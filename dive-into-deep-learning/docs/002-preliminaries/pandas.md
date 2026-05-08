# Data analysis with Pandas

[Documentation](https://pandas.pydata.org/docs/)

## Installation

```bash
python3 -m pip install pandas
```

## Formating data

### Define formatting for pandas

Float formatting

```python
# Example for money
pd.options.display.float_format = '{:,.2f}'.format 
```

Look at the formatting string in the other tutorial

Or with a series

```python
data['USD_Production_Budget'] = data['USD_Production_Budget'].str.replace('$', '')
data['USD_Production_Budget'] = data['USD_Production_Budget'].str.replace(',', '')
data['USD_Production_Budget'] = data['USD_Production_Budget'].astype(float)

print(f"Avarage production budget: {data['USD_Production_Budget'].mean():,.2f}")

```

```text
Avarage production budget: 31,113,737.58
```

### Define formatting for single DataFrame

1. Converting String to DateTime (Parsing)

```python
# Pandas usually detects standard ISO 8601 format (YYYY-MM-DD) 
df['date'] = pd.to_datetime(df['date'])

type(df.date[1])  # pandas._libs.tslibs.timestamps.Timestamp

# Or custom format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
```

| Code | Meaning          | Example |
|------|------------------|---------|
| `%Y` | 4-digit year     | 2024    |
| `%y` | 2-digit year     | 24      |
| `%m` | Month (01–12)    | 03      |
| `%d` | Day (01–31)      | 15      |
| `%H` | Hour (00–23)     | 14      |
| `%I` | Hour (01–12)     | 02      |
| `%M` | Minute           | 30      |
| `%S` | Second           | 45      |
| `%f` | Microseconds     | 123456  |
| `%b` | Short month name | Jan     |
| `%B` | Full month name  | January |
| `%a` | Short weekday    | Mon     |
| `%A` | Full weekday     | Monday  |
| `%p` | AM/PM            | PM      |

2. Change displaying

## Read CSV

Sample data `weather_data.csv`

```python
day, temp, condition
Monday, 12, Sunny
Tuesday, 14, Rain
Wednesday, 15, Rain
Thursday, 14, Cloudy
Friday, 21, Sunny
Saturday, 22, Sunny
Sunday, 24, Sunny
```

```python
import pandas as pd

data = pd.read_csv("weather_data.csv")

print(data)
```

Output is nicely formatted

```python
         day
temp
condition
0
Monday
12
Sunny
1
Tuesday
14
Rain
2
Wednesday
15
Rain
3
Thursday
14
Cloudy
4
Friday
21
Sunny
5
Saturday
22
Sunny
6
Sunday
24
Sunny
```

```python
print(data["temp"])
```

Output

```python
0
12
1
14
2
15
3
14
4
21
5
22
6
24
Name: temp, dtype: int64
```

## DataFrame and Series: working with rows and columns

### `DataFrame`

What pandas read is a [DataFrame](https://pandas.pydata.org/docs/reference/frame.html) object

> A `DataFrame` is Two-dimensional, size-mutable, potentially heterogeneous tabular data.

```python
print(type(data))  # <class 'pandas.DataFrame'>
```

#### DataFrame functions

| Method / Attribute    | Description                               | Example                                                                                |
|-----------------------|-------------------------------------------|----------------------------------------------------------------------------------------|
| pd.read_csv()         | Load CSV file into a DataFrame            | df = pd.read_csv("file.csv")                                                           |
| df.head(n)            | First n rows (default 5)                  | df.head()                                                                              |
| df.tail(n)            | Last n rows                               | df.tail(10)                                                                            |
| df.shape              | Number of rows and columns                | df.shape                                                                               |
| df.count()            | Number entries per column                 | df.count()                                                                             |
| df.columns            | Column names                              | df.columns                                                                             |
| df.dtypes             | Data types of columns                     | df.dtypes                                                                              |
| df.info()             | Summary of DataFrame (types, nulls)       | df.info()                                                                              |
| df.describe()         | Statistical summary of numeric columns    | df.describe()                                                                          |
| df["col"]             | Select a single column                    | df["age"]                                                                              |
| df[["col1","col2"]]   | Select multiple columns                   | df[["age","name"]]                                                                     |
| df.loc[]              | Label-based row/column selection          | df.loc[0:5, ["age"]]                                                                   |
| df.iloc[]             | Position-based row/column selection       | df.iloc[0:5, 0:2]                                                                      |
| df.at[]               | Access single value by label              | df.at[0, "age"]                                                                        |
| df.iat[]              | Access single value by position           | df.iat[0, 1]                                                                           |
| df.query()            | Filter rows using expression              | df.query("age > 30")                                                                   |
| df[df["col"] > value] | Boolean filtering                         | df_btc_price[df_btc_price.isna().any(axis=1)]                                          |
| df.value_counts()     | Count unique values in a column           | df["city"].value_counts()                                                              |
| df.unique()           | Get unique values                         | df["city"].unique()                                                                    |
| df.nunique()          | Count unique values                       | df["city"].nunique()                                                                   |
| df.isnull()           | Detect missing values                     | df.isnull()                                                                            |
| df.isna()             | Detect non numeric values                 | df.isna()                                                                              |
| df.dropna()           | New df with dropped non numeric values    | clean_df = df.isna()                                                                   |
| df.notnull()          | Detect non-missing values                 | df.notnull()                                                                           |
| df.sample(n)          | Random sample of rows                     | df.sample(5)                                                                           |
| df.sort_values()      | Sort by column                            | df_apps_clean.sort_values(by=['col1', 'col2'], ascending=[True, False], inplace=False) |
| df.groupby()          | Group data for aggregation                | df.groupby("city").mean()                                                              |
| df.drop               | Delete (drop) column                      | df.drop(columns=["col4"], inplace=True)                                                |
| df.duplicate()        | Creates a filter mask for duplicated rows | print(f"Are there duplicates?: {len(df[df.duplicated()]) > 0}")                        |

### Pivoting DataFrame

📥 Sample Data

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "city": ["Amsterdam", "Rotterdam", "Amsterdam", "Rotterdam"],
    "sales": [100, 150, 200, 250]
}

df = pd.DataFrame(data)
print(df)
```

Output

```
         date       city  sales
0  2024-01-01  Amsterdam    100
1  2024-01-01  Rotterdam    150
2  2024-01-02  Amsterdam    200
3  2024-01-02  Rotterdam    250
```

🔁 Pivot the Data

```python
pivot_df = df.pivot(index="date", columns="city", values="sales")
print(pivot_df)
```

```
city        Amsterdam  Rotterdam
date                           
2024-01-01        100        150
2024-01-02        200        250
```

Explanation
index → rows (date)
columns → new columns (city)
values → cell values (sales)

Bonus: Using pivot_table (handles duplicates)

```python
pivot_table_df = df.pivot_table(
    index="date",
    columns="city",
    values="sales",
    aggfunc="sum"
)
```

#### Handling `NaN`

For missing values, the pivot table will have `NaN`, we can replace it with

```python
df_pivot = df.pivot(index="Date", columns="TagName", values="NumTags")
print(df_pivot.head())
```

```
agName     assembly      c      c#    c++  delphi  go    java  javascript  \
Date                                                                         
2008-07-01       NaN    NaN     3.0    NaN     NaN NaN     NaN         NaN   
2008-08-01       8.0   82.0   503.0  164.0    13.0 NaN   220.0       160.0   
2008-09-01      28.0  320.0  1637.0  749.0   104.0 NaN  1121.0       629.0   
2008-10-01      16.0  302.0  1982.0  804.0   112.0 NaN  1142.0       720.0   
2008-11-01      16.0  257.0  1728.0  733.0   139.0 NaN   951.0       581.0   

```

```python
df_pivot.fillna(0, inplace=True)
print(df_pivot.head())
```

```
TagName     assembly      c      c#    c++  delphi   go    java  javascript  \
Date                                                                          
2008-07-01       0.0    0.0     3.0    0.0     0.0  0.0     0.0         0.0   
2008-08-01       8.0   82.0   503.0  164.0    13.0  0.0   220.0       160.0   
2008-09-01      28.0  320.0  1637.0  749.0   104.0  0.0  1121.0       629.0   
2008-10-01      16.0  302.0  1982.0  804.0   112.0  0.0  1142.0       720.0   
2008-11-01      16.0  257.0  1728.0  733.0   139.0  0.0   951.0       581.0   
```

### Grouping and Aggregating Data

We can group columns

```python
df.groupby('ColName').count()
```

Every row column will have the same value with the count

This gives the mean mid career salary per group

```python
clean_df.groupby('Group')['Mid-Career Median Salary'].mean()
```

It sums the values for each column on the grouped data frame

```python
cdf.groupby('TagName').sum()
```

Then we can also add filtering or slicing on the grouped dataframe as usual

```python
df_sets.groupby("year").count().query("year == 1955 | year == 2019")

# df  ->  DataFrameGroupBy -> df 

# sliced out the last two years
df_sets_per_full_year = df_sets.groupby("year").count()[:-2:]
```

### Grouping by multiple columns

Counts the number of Nobel prices won by category and sex

```python
# as_index false makes the first column not an index of the dataframe
cat_men_women = df_data.groupby(['category', 'sex'],
                                as_index=False).agg({'prize': pd.Series.count})
print(cat_men_women)
```

```text
  category     sex  prize
0    Chemistry  Female      7
1    Chemistry    Male    179
2    Economics  Female      2
3    Economics    Male     84
4   Literature  Female     16
5   Literature    Male    101
6     Medicine  Female     12
7     Medicine    Male    210
8        Peace  Female     17
9        Peace    Male     90
10     Physics  Female      4
11     Physics    Male    212
```

### The `.agg()` function

The `.agg()` (short for aggregate) function in pandas is used to apply one or more aggregation operations (like `sum`,
`mean`, `min`, `max`, etc.) across a `DataFrame` or `Series`.

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

df.agg('sum') 
```

```
A     6
B    15
```

Multiple aggragations

```python
df.agg(['sum', 'mean'])
```

```
        A     B
sum   6.0  15.0
mean  2.0   5.0
```

Custom functions

```python
df.agg(lambda x: x.max() - x.min())
```

### Combined with `groupby`

```python
df = pd.DataFrame({
    'Category': ['A', 'A', 'B'],
    'Value': [10, 20, 30]
})

df.groupby('Category').agg('sum')

```

```
Category
A    30
B    30
```

Multiple functions

```python
df.groupby('Category').agg({
    'Value': ['sum', 'mean']
})
```

Number of unique values per specific column

The function takes a `dict` as input with the column name and the aggregating function

```python
# the number of different (unique) themes per year 
df_sets.groupby("year").agg({"theme_id": pd.Series.nunique})
```

### Merge dataframes

Given the df

```python
df_themes = pd.read_csv("data/themes.csv")
theme_count_series = df_sets["theme_id"].value_counts()[::5]
df_theme_count = pd.DataFrame({"id": theme_count_series.index, "set_count": theme_count_series.values})
print(df_theme_count.head())
```

```
    id  set_count
0  158        753
1  505        328
2  443        197
3  453        142
4   52        115
```

and

```python
    id
name
parent_id
0
1
Technic
NaN
1
2
Arctic
Technic
1.0
2
3
Competition
1.0
3
4
Expert
Builder
1.0
4
5
Model
1.0
```

> It is important that they share the same merge column name (`id`)

```
merged_df = pd.merge(df_theme_count, df_themes, on='id')
print(merged_df.head())
```

```text
    id  set_count           name  parent_id
0  158        753      Star Wars        NaN
1  505        328      Basic Set      504.0
2  443        197  Service Packs        NaN
3  453        142        Technic      443.0
4   52        115           City       50.0
```

### Split and stack groups

Suppose we have a group by a field where a single row can have multiple field values.
For instance, we have a app dataframe where an app in a row can have more Genres (; separated)
If we run a group by genres the generet index value will contain multiple genres

We need to split the genres and then stack the group by

```python
df_apps_clean['Genres'].value_counts()
```

It contains `;` separated values

```text
Genres
Tools                                  732
Entertainment                          498
                                      ... 
Health & Fitness;Action & Adventure      1
Casual;Music & Video                     1
Lifestyle;Pretend Play                   1
Name: count, Length: 115, dtype: int64
```

we have 115 different values.

We can `split` and then `stack` the values

```python
# Split the strings on the semi-colon and then .stack them.
stacked_series = df_apps_clean.Genres.str.split(';', expand=True).stack()
print(f'We now have a single column with shape: {stack.shape}')
num_genres = stacked_series.value_counts()
print(f'Number of genres: {len(num_genres)}')
```

### Resampling

Convenience method for frequency conversion and resampling of time series. The object must have a datetime-like index (
DatetimeIndex, PeriodIndex, or TimedeltaIndex), or the caller must pass the label of a datetime-like series/index to the
on/level keyword parameter.

```text
print(df_btc_price.head())
```

```text
  DATE       CLOSE      VOLUME
0 2014-09-17  457.334015  21056800.0
1 2014-09-18  424.440002  34483200.0
2 2014-09-19  394.795990  37919700.0
3 2014-09-20  408.903992  36863600.0
4 2014-09-21  398.821014  26580100.0
```

Apply the resampling

```python
df_btc_price_montly = df_btc_price.resample('ME', on='DATE').last()
df_btc_price_montly.head()
```

#### Aggregating function

`last()` takes the last row in the df, so, if not sorted can be misleading.

Better sort by date before with `df_btc_price = df_btc_price.sort_values('DATE')`

Other aggregating functions are:

```python
.resample('M', on='DATE').mean()  # average price per month
.resample('M', on='DATE').max()  # max price per month
.resample('M', on='DATE').ohlc()  # OHLC (great for trading data)
```

Resample based on the last row of the `M` (month)

| Frequency | Meaning                            |
|-----------|------------------------------------|
| `'ME'`    | Month **end** (e.g., 2024-01-31)   |
| `'MS'`    | Month **start** (e.g., 2024-01-01) |

## `Series`

A `DataFrame` composes [Series](https://pandas.pydata.org/docs/reference/series.html)

> A `Series` One-dimensional ndarray with axis labels (including time series).
> Similar to the columns in the `DataFrame` table

```python
print(type(data["temp"]))  # <class 'pandas.Series'>
print(type(data.temp))  # <class 'pandas.Series'>
```

## Extracting columns as `Series`

```python
data = pd.read_csv("weather_data.csv")

# Extract column data (excluding the head)
day_series = data["day"]
temp_series = data["temp"]
condition_series = data.condition

print(condition_series)
```

Output
``
0     Sunny
1      Rain
2      Rain
3    Cloudy
4     Sunny
5     Sunny
6     Sunny
``

###

### Series functions

```python
import pandas as pd

temp_series = data["temp"]  # or data.temp

temperatures_list = temp_series.to_list()
average_temp = temp_series.mean()

print(temperatures_list)  # [12, 14, 15, 14, 21, 22, 24]
print(average_temp)  # 17.428571428571427
print(temp_series.max())  # 24
```

| Method / Attribute       | Description                       | Example                |
|--------------------------|-----------------------------------|------------------------|
| df["col"]                | Select column as Series           | s = df["age"]          |
| s.head(n)                | First n values                    | s.head()               |
| s.tail(n)                | Last n values                     | s.tail(10)             |
| s.shape                  | Length of the Series              | s.shape                |
| s.index                  | Index labels                      | s.index                |
| s.values                 | Underlying array                  | s.values               |
| s.dtype                  | Data type                         | s.dtype                |
| s.describe()             | Summary statistics                | s.describe()           |
| s.mean()                 | Mean value                        | s.mean()               |
| s.median()               | Median value                      | s.median()             |
| s.min() / s.max()        | Minimum / Maximum                 | s.min(), s.max()       |
| s.idxmin() / s.idxmax()  | Index of min / max value          | s.idxmax()             |
| s.sum()                  | Sum of values                     | s.sum()                |
| s.std()                  | Standard deviation                | s.std()                |
| s.count()                | Count non-null values             | s.count()              |
| s.unique()               | Unique values                     | s.unique()             |
| s.nunique()              | Number of unique values           | s.nunique()            |
| s.value_counts()         | Frequency of unique values        | s.value_counts()       |
| s.isnull()               | Detect missing values             | s.isnull()             |
| s.notnull()              | Detect non-missing values         | s.notnull()            |
| s.fillna(value)          | Replace missing values            | s.fillna(0)            |
| s.dropna()               | Remove missing values             | s.dropna()             |
| s.astype(type)           | Convert data type                 | s.astype(int)          |
| s.sort_values()          | Sort values                       | s.sort_values()        |
| s.sort_index()           | Sort by index                     | s.sort_index()         |
| s.apply(func)            | Apply function element-wise       | s.apply(lambda x: x*2) |
| s.map(func/dict)         | Map values using function or dict | s.map({"A":1,"B":2})   |
| s.replace()              | Replace specific values           | s.replace("A","X")     |
| s.between(a, b)          | Filter values within range        | s[s.between(10, 20)]   |
| s.clip(lower, upper)     | Limit values to bounds            | s.clip(0, 100)         |
| s.iloc[]                 | Position-based access             | s.iloc[0]              |
| s.loc[]                  | Label-based access                | s.loc[s.idmax()]       |
| s.cumsum()               | Cumulative sum                    | s.cumsum()             |
| s.rolling(window).mean() | Rolling average                   | s.rolling(3).mean()    |

### Series manipulation functions

| Method / Attribute                          | Description         | Example                                    |
|---------------------------------------------|---------------------|--------------------------------------------|
| df['col1'].subtract(df['col2'])             | Subtract (or `-`)   | diff_col = df['col1'].subtract(df['col2']) |
| clean_df.insert(<#_col>, <label>, <series>) | Insert series in DF | df.insert(1, 'Spread', diff_col)           |

### Series String Methods (`s.str`)

```python
df_apps_top_review_50[df_apps_top_review_50['Type'].str.strip().str.contains('Free', case=False, na=False)]
```

| Method                   | Description                    | Example                     |
|--------------------------|--------------------------------|-----------------------------|
| s.str.lower()            | Convert to lowercase           | s.str.lower()               |
| s.str.upper()            | Convert to uppercase           | s.str.upper()               |
| s.str.title()            | Title case                     | s.str.title()               |
| s.str.strip()            | Remove leading/trailing spaces | s.str.strip()               |
| s.str.lstrip()           | Remove leading spaces          | s.str.lstrip()              |
| s.str.rstrip()           | Remove trailing spaces         | s.str.rstrip()              |
| s.str.len()              | Length of each string          | s.str.len()                 |
| s.str.contains(pat)      | Check if pattern exists        | s.str.contains("abc")       |
| s.str.startswith(pat)    | Check prefix                   | s.str.startswith("A")       |
| s.str.endswith(pat)      | Check suffix                   | s.str.endswith("Z")         |
| s.str.replace(a, b)      | Replace substring              | s.str.replace("old","new")  |
| s.str.split(sep)         | Split strings into lists       | s.str.split(",")            |
| s.str.get(i)             | Get element from split/list    | s.str.split(",").str.get(0) |
| s.str.join(sep)          | Join list elements into string | s.str.join("-")             |
| s.str.extract(regex)     | Extract regex group            | s.str.extract(r"(\d+)")     |
| s.str.findall(regex)     | Find all regex matches         | s.str.findall(r"\d+")       |
| s.str.match(regex)       | Match regex from start         | s.str.match(r"^A")          |
| s.str.pad(width)         | Pad strings to width           | s.str.pad(10)               |
| s.str.slice(start, stop) | Slice substrings               | s.str.slice(0, 3)           |
| s.str.cat(sep)           | Concatenate strings            | s.str.cat(sep=",")          |
| s.str.isnumeric()        | Check if numeric string        | s.str.isnumeric()           |
| s.str.isalpha()          | Check if alphabetic            | s.str.isalpha()             |

### Series 📅 Datetime Methods (`s.dt`)

| Method               | Description                 | Example                             |
|----------------------|-----------------------------|-------------------------------------|
| pd.to_datetime(s)    | Convert Series to datetime  | s = pd.to_datetime(s)               |
| s.dt.date            | Extract date                | s.dt.date                           |
| s.dt.time            | Extract time                | s.dt.time                           |
| s.dt.year            | Extract year                | s.dt.year                           |
| s.dt.month           | Extract month               | s.dt.month                          |
| s.dt.day             | Extract day                 | s.dt.day                            |
| s.dt.hour            | Extract hour                | s.dt.hour                           |
| s.dt.minute          | Extract minute              | s.dt.minute                         |
| s.dt.second          | Extract second              | s.dt.second                         |
| s.dt.dayofweek       | Day of week (0=Mon)         | s.dt.dayofweek                      |
| s.dt.day_name()      | Name of day                 | s.dt.day_name()                     |
| s.dt.month_name()    | Name of month               | s.dt.month_name()                   |
| s.dt.quarter         | Quarter of year             | s.dt.quarter                        |
| s.dt.is_month_start  | Check if first day of month | s.dt.is_month_start                 |
| s.dt.is_month_end    | Check if last day of month  | s.dt.is_month_end                   |
| s.dt.tz_localize(tz) | Set timezone                | s.dt.tz_localize("UTC")             |
| s.dt.tz_convert(tz)  | Convert timezone            | s.dt.tz_convert("Europe/Amsterdam") |
| s.dt.strftime(fmt)   | Format datetime to string   | s.dt.strftime("%Y-%m-%d")           |
| s.dt.round(freq)     | Round datetime              | s.dt.round("H")                     |
| s.dt.floor(freq)     | Floor datetime              | s.dt.floor("D")                     |
| s.dt.ceil(freq)      | Ceil datetime               | s.dt.ceil("D")                      |

## Extracting rows as `DataFrame`

```python
# Get data in row
monday_data = data[data.day == "Monday"]

print(monday_data)  # still a DataFrame
```

Output
``
Name: condition, dtype: str
      day  temp condition
0  Monday    12     Sunny
``

## Create dataframe from series

We can use a dict

```python
# counts the rows with the same theme_id and slices until the last 5 rows
theme_count_series = df_sets["theme_id"].value_counts()[::5]
type(theme_count_series)  # series

df_theme_count = pd.DataFrame({"id": theme_count_series.index, "set_count": theme_count_series.values})
print(df_theme_count.head())
```

```
        id  set_count
0       158        753
1       505        328
2       443        197
3       453        142
4        52        115
```

### Merge two dataframes

To `.merge()` two DataFrame along a particular column, we need to provide our two DataFrames and then the column name on
which to merge.
This is why we set `on='id'`. Both our `set_theme_count` and our `themes` DataFrames have a column with this name.

```python
df_themes = pd.read_csv("data/themes.csv")
# from previous chapter 
theme_count_series

merged_df = pd.merge(df_theme_count, df_themes, on='id')
print(merged_df.head())
```

```
    id  set_count           name  parent_id
0  158        753      Star Wars        NaN
1  505        328      Basic Set      504.0
2  443        197  Service Packs        NaN
3  453        142        Technic      443.0
4   52        115           City       50.0
```

### Conversion to dict

```python
import pandas as pd

data = pd.read_csv("weather_data.csv")

data_dict = data.to_dict()

print(data_dict)
```

The ouptut is a dictionary with keys equal to the column names and values the series corresponding to that column

``{'day': {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}, 'temp': {0: 12, 1: 14, 2: 15, 3: 14, 4: 21, 5: 22, 6: 24}, 'condition': {0: 'Sunny', 1: 'Rain', 2: 'Rain', 3: 'Cloudy', 4: 'Sunny', 5: 'Sunny', 6: 'Sunny'}}``

## Create `DataFrame` from dict

```python
import pandas as pd

data_dict = {"students": ["Amy", "Wine", "House"], "scores": [90, 80, 70]}
df = pd.DataFrame(data_dict)
df.to_csv("data.csv")

print(df)
```

output

``
  students  scores
0      Amy      90
1     Wine      80
2    House      70
``
The `data.csv` will contain

``
,students,scores
0,Amy,90
1,Wine,80
2,House,70
``

### Loopint through `DataFrame` with

```python
import pandas as pd

student_dict = {"students": ["Amy", "Wine", "House"], "scores": [90, 80, 70]}

student_pd = pd.DataFrame(student_dict)

for (key, value) in student_pd.items():
    print(value)(0, 90)(1, 80)(2, 70)
```

The output is

```
0      Amy
1     Wine
2    House
Name: students, dtype: str
0    90
1    80
2    70
Name: scores, dtype: int64

```

### using `iterrows`

```python
for (index, row) in student_pd.iterrows():
    print(f"Index:{index}")
    print(f"Student:{row.students}")  # row is a Series
```

```python
Index: 0
Row: students
Amy
scores
90
Name: 0, dtype: object
Index: 1
Row: students
Wine
scores
80
Name: 1, dtype: object
Index: 2
Row: students
House
scores
70
Name: 2, dtype: object
```

## Clean data example

## Import

Read csv in panda

```python
import pandas as pd

# Show numeric output in decimal format e.g., 2.15
pd.options.display.float_format = '{:,.2f}'.format
df_apps = pd.read_csv('apps.csv')
df_apps.head()

```

|   | App                                          | Category        | Rating | Reviews | Size_MBs | Installs | Type | Price  | Content_Rating | Genres          | Last_Updated   | Android_Ver        |
|---|----------------------------------------------|-----------------|--------|---------|----------|----------|------|--------|----------------|-----------------|----------------|--------------------|
| 0 | Ak Parti Yardım Toplama                      | SOCIAL          | NaN    | 0       | 8.70     | 0        | Paid | $13.99 | Teen           | Social          | July 28, 2017  | 4.1 and up         |
| 1 | Ain Arabic Kids Alif Ba ta                   | FAMILY          | NaN    | 0       | 33.00    | 0        | Paid | $2.99  | Everyone       | Education       | April 15, 2016 | 3.0 and up         |
| 2 | Popsicle Launcher for Android P 9.0 launcher | PERSONALIZATION | NaN    | 0       | 5.50     | 0        | Paid | $1.49  | Everyone       | Personalization | July 11, 2018  | 4.2 and up         |
| 3 | Command &amp; Conquer: Rivals                | FAMILY          | NaN    | 0       | 19.00    | 0        | NaN  | 0      | Everyone 10+   | Strategy        | June 28, 2018  | Varies with device |
| 4 | CX Network                                   | BUSINESS        | NaN    | 0       | 10.00    | 0        | Free | 0      | Everyone       | Business        | August 6, 2018 | 4.1 and up         |

### Drop Unused Columns

**Challenge**: Remove the columns called `Last_Updated` and `Android_Version` from the DataFrame. We will not use these
columns.

```python
df_apps.drop(columns=["Last_Updated", "Android_Ver"], inplace=True)
```

### Find and Remove NaN values in Ratings

**Challenge**: How may rows have a NaN value (not-a-number) in the Ratings column? Create DataFrame called
`df_apps_clean` that does not include these rows.

```python
df_apps.Rating.isna().value_counts()
```

    Rating
    False    9367
    True     1474
    Name: count, dtype: int64

```python
df_apps = df_apps[df_apps['Rating'].isna() == False]
df_apps.Rating.isna().value_counts()
```

    Rating
    False    9367
    Name: count, dtype: int64

## Remove duplicate rows

```python
df_apps.drop_duplicates(inplace=True)
```

```python
df_apps.duplicated().sum()  # expected 0
```

    np.int64(0)




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib import font_manager as fm
from sklearn.preprocessing import StandardScaler

# Load Roboto fonts
roboto_bold = fm.FontProperties(fname=r'C:\cs132\Roboto-Bold.ttf')
roboto_regular = fm.FontProperties(fname=r'C:\cs132\Roboto-Regular.ttf')
roboto_condensed = fm.FontProperties(fname=r'C:\cs132\RobotoCondensed-Regular.ttf')

# Your existing data preparation code
df_agri = pd.read_csv(r"C:/cs132/agriculture.csv")
df_precip = pd.read_csv(r"C:/cs132/precip.csv")

# Colors
color_bg = "#1B181C"
color_text = "#FFFFFF"
color_points = "#648fff"
color_line = "#dc267f"
color_cpi_points = "#ffb000"
color_cpi_line = "#fe6100"

# precip.csv data
df_precip_years = [int(str(year[:4])) for year in df_precip['Year']]  # Removes '07' from each year and converts to string
precips = [x for x in df_precip['Precipitation (mm)']]
precip_avg = sum(precips) / len(precips)
temps = [x for x in df_precip['Temperature']]
temp_avg = sum(temps) / len(temps)

# agriculture.csv data
code = 'AG.PRD.CROP.XD'  # Indicator code for crop production index (cpi)
df_agri_years = (df_agri[df_agri['Indicator Code'] == code]['Year'].tolist())
df_agri_years.reverse()
cpi = df_agri[df_agri['Indicator Code'] == code]['Value'].tolist()
cpi.reverse()
cpi_avg = sum(cpi) / len(cpi)

dict_precif = pd.DataFrame({
    'Year': df_precip_years,
    'Precipitation (mm)': precips,
    'Temperature': temps
})

dict_agri = pd.DataFrame({
    'Year': df_agri_years,
    'Value': cpi
})

all_years = list(range(1961, 2014))

precip_dict = dict(zip(df_precip_years, (precips)))
temp_dict = dict(zip(df_precip_years, temps))
cpi_dict = dict(zip(df_agri_years, cpi))

filled_precips = [precip_dict.get(year, precip_avg) / 100 for year in all_years]
filled_temps = [temp_dict.get(year, temp_avg) for year in all_years]
filled_cpi = [cpi_dict.get(year, cpi_avg) for year in all_years]

# Create new DataFrame with updated values
df_filled = pd.DataFrame({
    'Year': all_years,
    'Precipitation (mm)': filled_precips,
    'Temperature': filled_temps,
    'Crop Production Index': filled_cpi
})


# Function to plot Crop Production Index vs Temperature
def cpi_temp():
    X = df_filled[['Temperature']]  # Independent variable
    y = df_filled['Crop Production Index']  # Dependent variable

    # Fit linear regression model
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Plot the linear regression line
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(color_bg)
    ax.set_facecolor(color_bg)

    ax.scatter(X, y, color=color_points, label='Actual data', alpha=0.8)
    ax.plot(X, y_pred, color=color_line, label='Linear Regression')

    # Add labels and title
    ax.set_xlabel('Temperature (°C)', fontproperties=roboto_regular, color=color_text)
    ax.set_ylabel('Crop Production Index', fontproperties=roboto_regular, color=color_text)
    ax.set_title('Linear Regression for Crop Production Index vs Temperature', fontproperties=roboto_bold, color=color_text)

    # Add legend
    ax.legend(loc='upper center', prop=roboto_regular, facecolor=color_bg, edgecolor='none', labelcolor=color_text)

    # Customize ticks
    ax.tick_params(
        axis='x',
        which='both',
        color=color_text,
        labelcolor=color_text,
        bottom=False,
        top=False,
        labelbottom=True)
    ax.tick_params(
        axis='y',
        which='both',
        color=color_text,
        labelcolor=color_text,
        left=False,
        right=False,
        labelleft=True)
    ax.spines['bottom'].set_color(color_text)
    ax.spines['left'].set_color(color_text)

    # Remove top and right axes
    sns.despine(ax=ax, top=True, right=True)
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = color_bg
    plt.rcParams['figure.facecolor'] = color_bg
    plt.rcParams['axes.edgecolor'] = color_text
    plt.rcParams['grid.color'] = color_text
    plt.rcParams['text.color'] = color_text
    plt.rcParams['xtick.color'] = color_text
    plt.rcParams['ytick.color'] = color_text

    # Show plot
    plt.savefig('rq1cpitemp.png', dpi=300)
    plt.show()


# Function to plot Crop Production Index vs Precipitation
def cpi_precip():
    X = df_filled[['Precipitation (mm)']]  # Independent variable
    y = df_filled['Crop Production Index']  # Dependent variable

    # Fit linear regression model
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Plot the linear regression line
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='blue', label='Actual data', alpha=0.6)
    plt.plot(X, y_pred, color='red', label='Linear Regression')

    # Add labels and title
    plt.xlabel('Precipitation (mm/year)', fontproperties=roboto_regular)
    plt.ylabel('Crop Production Index', fontproperties=roboto_regular)
    plt.title('Linear Regression for Crop Production Index vs Precipitation (mm/year)', fontproperties=roboto_bold)

    # Add legend
    plt.legend(prop=roboto_regular)

    # Remove top and right axes
    sns.despine()

    # Customize ticks
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=True)
    plt.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=True)

    # Show plot
    plt.savefig('rq1cpiprecip.png', dpi=300)
    plt.show()


# Call the function to visualize the plot
#cpi_precip()
cpi_temp()


def cpi_year_temp():
    X = df_filled[['Year']]  # Independent variable (Years)
    y1 = df_filled['Temperature']  # First dependent variable (Temperature)
    y2 = df_filled['Crop Production Index']  # Second dependent variable (Crop Production Index)

    # Fit linear regression model for Temperature
    model_temp = LinearRegression().fit(X, y1)
    y1_pred = model_temp.predict(X)

    # Fit linear regression model for Crop Production Index
    model_cpi = LinearRegression().fit(X, y2)
    y2_pred = model_cpi.predict(X)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(color_bg)
    ax1.set_facecolor(color_bg)

    # Plot Temperature
    ax1.scatter(X, y1, color=color_points, label='Temperature Data', alpha=0.8)
    ax1.plot(X, y1_pred, color=color_line, label='Temperature Linear Regression')
    ax1.set_xlabel('Year', fontproperties=roboto_regular, color=color_text)
    ax1.set_ylabel('Temperature (°C)', fontproperties=roboto_regular, color=color_text)
    ax1.tick_params(axis='y', colors=color_text)
    ax1.tick_params(axis='x', colors=color_text)
    ax1.legend(loc='upper left', prop=roboto_regular, facecolor=color_bg, edgecolor='none', labelcolor=color_text)

    # Add second y-axis for Crop Production Index
    ax2 = ax1.twinx()
    ax2.scatter(X, y2, color=color_cpi_points, label='Crop Production Index Data', alpha=0.8)
    ax2.plot(X, y2_pred, color=color_cpi_line, label='Crop Production Index Linear Regression')
    ax2.set_ylabel('Crop Production Index', fontproperties=roboto_regular, color=color_text)
    ax2.tick_params(axis='y', colors=color_text)
    ax2.legend(loc='upper right', prop=roboto_regular, facecolor=color_bg, edgecolor='none', labelcolor=color_text)

    # Add title
    ax1.set_title('Linear Regression for Crop Production Index vs Time with Temperature', fontproperties=roboto_bold, color=color_text)

    # Remove top and right spines
    sns.despine(ax=ax1, top=True, right=False)
    sns.despine(ax=ax2, top=True, right=False)

    # Show plot
    plt.show()

#cpi_year_temp()


# Standardize the data
scaler = StandardScaler()
df_filled[['Precipitation (mm)', 'Temperature', 'Crop Production Index']] = scaler.fit_transform(df_filled[['Precipitation (mm)', 'Temperature', 'Crop Production Index']])


def trends_over_time():
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(color_bg)
    ax1.set_facecolor(color_bg)

    # Plot Temperature and Precipitation trends
    ax1.plot(df_filled['Year'], df_filled['Temperature'], color=color_line, label='Temperature (°C)')
    ax1.set_xlabel('Year', fontproperties=roboto_regular, color=color_text)
    ax1.set_ylabel('Standardized Values \nfor Crop Production Index and Temperature (°C)', fontproperties=roboto_regular, color=color_text)
    ax1.tick_params(axis='y', colors=color_text)
    ax1.tick_params(axis='x', colors=color_text)
    #ax1.legend(loc='upper left', prop=roboto_regular, facecolor=color_bg, edgecolor='none', labelcolor=color_text)

    #ax2 = ax1.twinx()
    #ax1.plot(df_filled['Year'], df_filled['Precipitation (mm)'], color="#fe6100", label='Precipitation (cm)')
    #ax2.set_ylabel('Crop Production Index\n(Precipitation (m)', fontproperties=roboto_regular, color=color_text)
    #ax2.tick_params(axis='y', colors=color_text)
    #ax1.legend(loc='upper right', prop=roboto_regular, facecolor=color_bg, edgecolor='none', labelcolor=color_text)

    # Plot Crop Production Index trend
    ax1.plot(df_filled['Year'], df_filled['Crop Production Index'], color=color_cpi_points, label='Crop Production Index', linestyle='--')
    ax1.legend(loc='upper right', prop=roboto_regular, facecolor=color_bg, edgecolor='none', labelcolor=color_text)

    # Title
    ax1.set_title('Trends of Precipitation, and Crop Production Index Over Time', fontproperties=roboto_bold, color=color_text)

    sns.despine(ax=ax1, top=True, right=True)
    #sns.despine(ax=ax2, top=True, right=False)
    plt.legend(loc='upper center', prop=roboto_regular, facecolor=color_bg, edgecolor='none', labelcolor=color_text) 
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = color_bg
    plt.rcParams['figure.facecolor'] = color_bg
    plt.rcParams['axes.edgecolor'] = color_text
    plt.rcParams['grid.color'] = color_text
    plt.rcParams['text.color'] = color_text
    plt.rcParams['xtick.color'] = color_text
    plt.rcParams['ytick.color'] = color_text

    plt.savefig('nutshell.pdf', dpi=300)
    plt.show()
    
    
    
trends_over_time()
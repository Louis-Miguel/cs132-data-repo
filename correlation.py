import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
colors_grad = sns.color_palette('flare_r',  12)
colors_heat1 = sns.color_palette('flare_r', as_cmap=True)
colors_heat2 = sns.diverging_palette(315, 261, s=74, l=50, center='dark', as_cmap=True)

# IBM Color Blind Safe Palette
colors_heatmap = sns.color_palette("colorblind")

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
filled_cpi = [cpi_dict.get(year, cpi_avg) / 10 for year in all_years]

# Create new DataFrame with updated values
df_filled = pd.DataFrame({
    'Year': all_years,
    'Temperature': filled_temps,
    'Crop Production Index': filled_cpi
})

# Standardize the data
#scaler = StandardScaler()
#df_filled[['Temperature', 'Crop Production Index']] = scaler.fit_transform(df_filled[['Temperature', 'Crop Production Index']])




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

# Assuming df_filled is already created and standardized as per your previous code

# Bin the data into categories
num_bins = 3  # You can adjust the number of bins as needed
df_filled['Temperature_binned'] = pd.qcut(df_filled['Temperature'], num_bins, labels=False)
df_filled['CPI_binned'] = pd.qcut(df_filled['Crop Production Index'], num_bins, labels=False)

# Create a contingency table
contingency_table = pd.crosstab(df_filled['Temperature_binned'], df_filled['CPI_binned'])

# Perform the chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print the results
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print("Expected frequencies:")
print(expected)

# Interpret the p-value
alpha = 0.05  # Significance level
if p < alpha:
    print("There is a significant association between Temperature and Crop Production Index.")
else:
    print("There is no significant association between Temperature and Crop Production Index.")

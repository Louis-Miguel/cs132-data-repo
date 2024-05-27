import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager as fm

# Load Roboto fonts
roboto_bold = fm.FontProperties(fname=r'C:\cs132\Roboto-Bold.ttf')
roboto_regular = fm.FontProperties(fname=r'C:\cs132\Roboto-Regular.ttf')
roboto_condensed = fm.FontProperties(fname=r'C:\cs132\RobotoCondensed-Regular.ttf')

# Load data
df_agri = pd.read_csv(r"C:/cs132/agriculture.csv")
df_ghg = pd.read_csv(r"C:/cs132/ghg.csv")

# ghg.csv data
df_ghg_years = np.array(df_ghg['Year'])
ghg = np.array(df_ghg['MtCO2e'])

# agriculture.csv data
code = 'NV.AGR.TOTL.ZS' # Indicator code for agriculture (% in GDP)
df_gdp_years = np.array(range(1990, 2021))
gdp = df_agri[(df_agri['Indicator Code'] == code) & (df_agri['Year'].isin(range(1990, 2021)))]
gdp = np.array(gdp['Value'])[::-1]

# Normalize the datasets
normalized_gdp = (gdp - gdp.min()) / (gdp.max() - gdp.min())
normalized_ghg = (ghg - ghg.min()) / (ghg.max() - ghg.min())

# Colors
color_bg = "#1B181C"
color_text = "#FFFFFF"

# Set the style and background color
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = color_bg
plt.rcParams['figure.facecolor'] = color_bg
plt.rcParams['axes.edgecolor'] = color_text
plt.rcParams['grid.color'] = color_text
plt.rcParams['text.color'] = color_text
plt.rcParams['xtick.color'] = color_text
plt.rcParams['ytick.color'] = color_text

# Plotting the normalized data
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_gdp_years, normalized_gdp, label='Normalized Agriculture (% of GDP)', color='#785ef0')
ax.plot(df_ghg_years, normalized_ghg, label='Normalized CO2 emissions (MtCO2e)', color='#DC267F')

# Set font properties for titles, labels, captions, and annotations
ax.set_title('Agriculture (% of GDP) and Metric Ton CO2 Emissions from 1990-2020', fontproperties=roboto_bold)
ax.set_xlabel('Year', fontproperties=roboto_regular)
ax.set_ylabel('Normalized Values', fontproperties=roboto_regular)
ax.legend(loc = 'upper center', prop=roboto_regular, edgecolor='none')

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=12)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(roboto_regular)

sns.despine()
plt.savefig('rq2.png', dpi=300)
plt.show()

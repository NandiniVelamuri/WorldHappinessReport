# **Predicting Happiness Scores: A Machine Learning Approach**
**Exploring the Relationship Between Economic and Social Factors and Happiness**
====================================================================================
This notebook explores the use of machine learning techniques to predict happiness scores based on various factors such as GDP per capita, healthy life expectancy, social support, and more. We will train and evaluate different models, including decision trees and random forests, to identify the most effective approach. Through this analysis, we aim to gain insights into the key drivers of happiness and develop a robust predictive model.

# 1. Importing Libraries
### Libraries Used:
1. **pandas**: Data manipulation and analysis
2. **seaborn**: Data visualization
3. **numpy**: Numerical operations
4. **matplotlib**: Plotting and visualization
5. **scikit-learn**: Machine learning models and utilities


```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

# 2. Loading the Dataset
The World Happiness Report 2024 dataset is loaded into a pandas DataFrame from a CSV file named `final_data.csv`.


```python
df = pd.read_csv('final_data.csv')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Rank</th>
      <th>Ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
      <th>Age Below 30 Score</th>
      <th>Age Above 60 Score</th>
      <th>Age 30-44 Score</th>
      <th>Age 45-59 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Finland</td>
      <td>1</td>
      <td>7.741</td>
      <td>7.815</td>
      <td>7.667</td>
      <td>1.844</td>
      <td>1.572</td>
      <td>0.695</td>
      <td>0.859</td>
      <td>0.142</td>
      <td>0.546</td>
      <td>2.082</td>
      <td>7.300</td>
      <td>7.912</td>
      <td>7.883800</td>
      <td>7.868200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2</td>
      <td>7.583</td>
      <td>7.665</td>
      <td>7.500</td>
      <td>1.908</td>
      <td>1.520</td>
      <td>0.699</td>
      <td>0.823</td>
      <td>0.204</td>
      <td>0.548</td>
      <td>1.881</td>
      <td>7.329</td>
      <td>7.916</td>
      <td>7.540260</td>
      <td>7.546740</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>3</td>
      <td>7.525</td>
      <td>7.618</td>
      <td>7.433</td>
      <td>1.881</td>
      <td>1.617</td>
      <td>0.718</td>
      <td>0.819</td>
      <td>0.258</td>
      <td>0.182</td>
      <td>2.050</td>
      <td>7.598</td>
      <td>7.585</td>
      <td>7.456012</td>
      <td>7.460988</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sweden</td>
      <td>4</td>
      <td>7.344</td>
      <td>7.422</td>
      <td>7.267</td>
      <td>1.878</td>
      <td>1.501</td>
      <td>0.724</td>
      <td>0.838</td>
      <td>0.221</td>
      <td>0.524</td>
      <td>1.658</td>
      <td>7.026</td>
      <td>7.588</td>
      <td>7.372880</td>
      <td>7.389120</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Israel</td>
      <td>5</td>
      <td>7.341</td>
      <td>7.405</td>
      <td>7.277</td>
      <td>1.803</td>
      <td>1.513</td>
      <td>0.740</td>
      <td>0.641</td>
      <td>0.153</td>
      <td>0.193</td>
      <td>2.298</td>
      <td>7.667</td>
      <td>6.854</td>
      <td>7.423066</td>
      <td>7.419934</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Congo (Kinshasa)</td>
      <td>136</td>
      <td>3.295</td>
      <td>3.462</td>
      <td>3.128</td>
      <td>0.534</td>
      <td>0.665</td>
      <td>0.262</td>
      <td>0.473</td>
      <td>0.189</td>
      <td>0.072</td>
      <td>1.102</td>
      <td>3.441</td>
      <td>2.703</td>
      <td>3.516969</td>
      <td>3.519031</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Sierra Leone</td>
      <td>137</td>
      <td>3.245</td>
      <td>3.366</td>
      <td>3.124</td>
      <td>0.654</td>
      <td>0.566</td>
      <td>0.253</td>
      <td>0.469</td>
      <td>0.181</td>
      <td>0.053</td>
      <td>1.068</td>
      <td>3.225</td>
      <td>3.471</td>
      <td>3.137864</td>
      <td>3.146136</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Lesotho</td>
      <td>138</td>
      <td>3.186</td>
      <td>3.469</td>
      <td>2.904</td>
      <td>0.771</td>
      <td>0.851</td>
      <td>0.000</td>
      <td>0.523</td>
      <td>0.082</td>
      <td>0.085</td>
      <td>0.875</td>
      <td>3.700</td>
      <td>2.808</td>
      <td>3.114573</td>
      <td>3.121427</td>
    </tr>
    <tr>
      <th>138</th>
      <td>Lebanon</td>
      <td>139</td>
      <td>2.707</td>
      <td>2.797</td>
      <td>2.616</td>
      <td>1.377</td>
      <td>0.577</td>
      <td>0.556</td>
      <td>0.173</td>
      <td>0.068</td>
      <td>0.029</td>
      <td>-0.073</td>
      <td>2.997</td>
      <td>2.490</td>
      <td>2.673950</td>
      <td>2.667050</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Afghanistan</td>
      <td>140</td>
      <td>1.721</td>
      <td>1.775</td>
      <td>1.667</td>
      <td>0.628</td>
      <td>0.000</td>
      <td>0.242</td>
      <td>0.000</td>
      <td>0.091</td>
      <td>0.088</td>
      <td>0.672</td>
      <td>1.827</td>
      <td>1.456</td>
      <td>1.805547</td>
      <td>1.795453</td>
    </tr>
  </tbody>
</table>
<p>140 rows × 16 columns</p>
</div>



# 3. Checking for Missing Values
This cell checks for missing values in the dataset by counting the number of null values in each column.


```python
df.isnull().sum()
```




    Country                                       0
    Rank                                          0
    Ladder score                                  0
    upperwhisker                                  0
    lowerwhisker                                  0
    Explained by: Log GDP per capita              3
    Explained by: Social support                  3
    Explained by: Healthy life expectancy         3
    Explained by: Freedom to make life choices    3
    Explained by: Generosity                      3
    Explained by: Perceptions of corruption       3
    Dystopia + residual                           3
    Age Below 30 Score                            0
    Age Above 60 Score                            0
    Age 30-44 Score                               0
    Age 45-59 Score                               0
    dtype: int64



# 4. Visualizing Missing Values
A heatmap is used to visualize the missing values in the dataset, providing a clearer representation of the null values.


```python
sns.heatmap(df.isnull())
```




    <Axes: >




    
![png](output_9_1.png)
    


# 5. Handling Missing Values
The `dropna()` function is used to remove rows with missing values from the dataset, resulting in a cleaned dataset.


```python
df = df.dropna()
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Rank</th>
      <th>Ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
      <th>Age Below 30 Score</th>
      <th>Age Above 60 Score</th>
      <th>Age 30-44 Score</th>
      <th>Age 45-59 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Finland</td>
      <td>1</td>
      <td>7.741</td>
      <td>7.815</td>
      <td>7.667</td>
      <td>1.844</td>
      <td>1.572</td>
      <td>0.695</td>
      <td>0.859</td>
      <td>0.142</td>
      <td>0.546</td>
      <td>2.082</td>
      <td>7.300</td>
      <td>7.912</td>
      <td>7.883800</td>
      <td>7.868200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2</td>
      <td>7.583</td>
      <td>7.665</td>
      <td>7.500</td>
      <td>1.908</td>
      <td>1.520</td>
      <td>0.699</td>
      <td>0.823</td>
      <td>0.204</td>
      <td>0.548</td>
      <td>1.881</td>
      <td>7.329</td>
      <td>7.916</td>
      <td>7.540260</td>
      <td>7.546740</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>3</td>
      <td>7.525</td>
      <td>7.618</td>
      <td>7.433</td>
      <td>1.881</td>
      <td>1.617</td>
      <td>0.718</td>
      <td>0.819</td>
      <td>0.258</td>
      <td>0.182</td>
      <td>2.050</td>
      <td>7.598</td>
      <td>7.585</td>
      <td>7.456012</td>
      <td>7.460988</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sweden</td>
      <td>4</td>
      <td>7.344</td>
      <td>7.422</td>
      <td>7.267</td>
      <td>1.878</td>
      <td>1.501</td>
      <td>0.724</td>
      <td>0.838</td>
      <td>0.221</td>
      <td>0.524</td>
      <td>1.658</td>
      <td>7.026</td>
      <td>7.588</td>
      <td>7.372880</td>
      <td>7.389120</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Israel</td>
      <td>5</td>
      <td>7.341</td>
      <td>7.405</td>
      <td>7.277</td>
      <td>1.803</td>
      <td>1.513</td>
      <td>0.740</td>
      <td>0.641</td>
      <td>0.153</td>
      <td>0.193</td>
      <td>2.298</td>
      <td>7.667</td>
      <td>6.854</td>
      <td>7.423066</td>
      <td>7.419934</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Congo (Kinshasa)</td>
      <td>136</td>
      <td>3.295</td>
      <td>3.462</td>
      <td>3.128</td>
      <td>0.534</td>
      <td>0.665</td>
      <td>0.262</td>
      <td>0.473</td>
      <td>0.189</td>
      <td>0.072</td>
      <td>1.102</td>
      <td>3.441</td>
      <td>2.703</td>
      <td>3.516969</td>
      <td>3.519031</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Sierra Leone</td>
      <td>137</td>
      <td>3.245</td>
      <td>3.366</td>
      <td>3.124</td>
      <td>0.654</td>
      <td>0.566</td>
      <td>0.253</td>
      <td>0.469</td>
      <td>0.181</td>
      <td>0.053</td>
      <td>1.068</td>
      <td>3.225</td>
      <td>3.471</td>
      <td>3.137864</td>
      <td>3.146136</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Lesotho</td>
      <td>138</td>
      <td>3.186</td>
      <td>3.469</td>
      <td>2.904</td>
      <td>0.771</td>
      <td>0.851</td>
      <td>0.000</td>
      <td>0.523</td>
      <td>0.082</td>
      <td>0.085</td>
      <td>0.875</td>
      <td>3.700</td>
      <td>2.808</td>
      <td>3.114573</td>
      <td>3.121427</td>
    </tr>
    <tr>
      <th>138</th>
      <td>Lebanon</td>
      <td>139</td>
      <td>2.707</td>
      <td>2.797</td>
      <td>2.616</td>
      <td>1.377</td>
      <td>0.577</td>
      <td>0.556</td>
      <td>0.173</td>
      <td>0.068</td>
      <td>0.029</td>
      <td>-0.073</td>
      <td>2.997</td>
      <td>2.490</td>
      <td>2.673950</td>
      <td>2.667050</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Afghanistan</td>
      <td>140</td>
      <td>1.721</td>
      <td>1.775</td>
      <td>1.667</td>
      <td>0.628</td>
      <td>0.000</td>
      <td>0.242</td>
      <td>0.000</td>
      <td>0.091</td>
      <td>0.088</td>
      <td>0.672</td>
      <td>1.827</td>
      <td>1.456</td>
      <td>1.805547</td>
      <td>1.795453</td>
    </tr>
  </tbody>
</table>
<p>137 rows × 16 columns</p>
</div>



# 6. Summary Statistics
The `describe()` function is used to generate summary statistics for the dataset, providing an overview of the central tendency and dispersion of the data.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
      <th>Age Below 30 Score</th>
      <th>Age Above 60 Score</th>
      <th>Age 30-44 Score</th>
      <th>Age 45-59 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>70.226277</td>
      <td>5.545686</td>
      <td>5.658650</td>
      <td>5.432693</td>
      <td>1.379066</td>
      <td>1.131255</td>
      <td>0.520445</td>
      <td>0.624569</td>
      <td>0.148051</td>
      <td>0.155577</td>
      <td>1.586774</td>
      <td>5.841431</td>
      <td>5.312620</td>
      <td>5.518012</td>
      <td>5.518682</td>
    </tr>
    <tr>
      <th>std</th>
      <td>40.886499</td>
      <td>1.180367</td>
      <td>1.164368</td>
      <td>1.197052</td>
      <td>0.423493</td>
      <td>0.336110</td>
      <td>0.164605</td>
      <td>0.158282</td>
      <td>0.073209</td>
      <td>0.127112</td>
      <td>0.536841</td>
      <td>1.176243</td>
      <td>1.292007</td>
      <td>1.177076</td>
      <td>1.179262</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.721000</td>
      <td>1.775000</td>
      <td>1.667000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.073000</td>
      <td>1.827000</td>
      <td>1.456000</td>
      <td>1.805547</td>
      <td>1.795453</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35.000000</td>
      <td>4.657000</td>
      <td>4.779000</td>
      <td>4.535000</td>
      <td>1.078000</td>
      <td>0.915000</td>
      <td>0.400000</td>
      <td>0.533000</td>
      <td>0.096000</td>
      <td>0.069000</td>
      <td>1.318000</td>
      <td>4.906000</td>
      <td>4.417000</td>
      <td>4.682641</td>
      <td>4.691359</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>70.000000</td>
      <td>5.816000</td>
      <td>5.927000</td>
      <td>5.679000</td>
      <td>1.430000</td>
      <td>1.236000</td>
      <td>0.549000</td>
      <td>0.641000</td>
      <td>0.138000</td>
      <td>0.122000</td>
      <td>1.653000</td>
      <td>6.305000</td>
      <td>5.418000</td>
      <td>5.756969</td>
      <td>5.758031</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>106.000000</td>
      <td>6.442000</td>
      <td>6.522000</td>
      <td>6.339000</td>
      <td>1.752000</td>
      <td>1.390000</td>
      <td>0.648000</td>
      <td>0.735000</td>
      <td>0.194000</td>
      <td>0.196000</td>
      <td>1.884000</td>
      <td>6.732000</td>
      <td>6.164000</td>
      <td>6.424129</td>
      <td>6.425081</td>
    </tr>
    <tr>
      <th>max</th>
      <td>140.000000</td>
      <td>7.741000</td>
      <td>7.815000</td>
      <td>7.667000</td>
      <td>2.141000</td>
      <td>1.617000</td>
      <td>0.857000</td>
      <td>0.863000</td>
      <td>0.401000</td>
      <td>0.575000</td>
      <td>2.998000</td>
      <td>7.759000</td>
      <td>7.916000</td>
      <td>7.883800</td>
      <td>7.868200</td>
    </tr>
  </tbody>
</table>
</div>



# 7. Distribution of Ladder Score
A histogram is used to visualize the distribution of the 'Ladder score' column in the dataset. This plot provides insights into the shape of the distribution, including:
- Central tendency (mean, median)
- Dispersion (range, variance)
- Skewness and kurtosis
- Presence of outliers or multimodality
The `sns.histplot()` function from Seaborn is utilized to create a visually appealing and informative histogram.


```python
sns.histplot(df['Ladder score'])
```

# 8. Global Happiness Choropleth Map
An interactive choropleth map is created to visualize the global happiness rankings. The map displays the happiness rank of each country, providing a geographical perspective on happiness levels worldwide.
- **Locations**: Country names are used to map the data to specific countries.
- **Colorbar**: The colorbar represents the happiness ranking, with colors indicating the relative ranking of each country.
- **Interactivity**: The plotly library enables interactive visualization, allowing users to hover over countries to view specific rankings and country names.
This visualization facilitates comparison of happiness levels across different regions and countries.


```python
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=True)
data = dict(type = 'choropleth', 
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Rank'], 
           text = df['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 
             geo = dict(showframe = False),
             width=800,  
             height=500)
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
</script>
<script type="module">import "https://cdn.plot.ly/plotly-3.0.1.min"</script>




<div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>                <div id="d78a794a-7785-4b23-a52d-cac2caf69e37" class="plotly-graph-div" style="height:500px; width:800px;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("d78a794a-7785-4b23-a52d-cac2caf69e37")) {                    Plotly.newPlot(                        "d78a794a-7785-4b23-a52d-cac2caf69e37",                        [{"colorbar":{"title":{"text":"Happiness"}},"locationmode":"country names","locations":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"text":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"z":{"dtype":"i2","bdata":"AQACAAMABAAFAAYABwAIAAkACgALAAwADQAOAA8AEAARABIAEwAUABUAFgAXABgAGQAaABsAHAAdAB4AHwAgACEAIgAjACQAJQAmACcAKAApACoAKwAsAC0ALgAvADAAMQAyADMANAA1ADYANwA4ADkAOgA7ADwAPgA\u002fAEAAQQBCAEMARABFAEYARwBIAEkASgBLAEwATQBOAE8AUABRAFIAUwBUAFUAVgBYAFkAWgBbAFwAXQBeAF8AYABhAGIAYwBkAGYAZwBoAGkAagBrAGwAbQBuAG8AcABxAHIAcwB0AHUAdgB3AHgAeQB6AHsAfAB9AH4AfwCAAIEAggCDAIQAhQCGAIcAiACJAIoAiwCMAA=="},"type":"choropleth"}],                        {"geo":{"showframe":false},"height":500,"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Global Happiness"},"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('d78a794a-7785-4b23-a52d-cac2caf69e37');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };            </script>        </div>


# 9. GDP per capita Choropleth Map
An interactive choropleth map is created to visualize the GDP per capita across different countries. The map displays the log GDP per capita values, highlighting the economic disparities between nations.
- **Locations**: Country names are used to map the data to specific countries.
- **Colorbar**: The colorbar represents the log GDP per capita values, with colors indicating the relative economic prosperity of each country.
- **Insights**: This visualization helps identify patterns and correlations between GDP per capita and happiness rankings, providing valuable insights into the relationship between economic factors and well-being.


```python
data = dict(type = 'choropleth', 
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Explained by: Log GDP per capita'], 
           text = df['Country'],
           colorbar = {'title':'GDP per capita'})
layout = dict(title = 'GDP per capita',
             geo = dict(showframe = False),
             width=800,  
             height=500)
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
```


<div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>                <div id="0a119170-30f6-44e3-8624-ad6df403c5f5" class="plotly-graph-div" style="height:500px; width:800px;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("0a119170-30f6-44e3-8624-ad6df403c5f5")) {                    Plotly.newPlot(                        "0a119170-30f6-44e3-8624-ad6df403c5f5",                        [{"colorbar":{"title":{"text":"GDP per capita"}},"locationmode":"country names","locations":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"text":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"z":{"dtype":"f8","bdata":"Gy\u002fdJAaB\u002fT+6SQwCK4f+P39qvHSTGP4\u002fPzVeukkM\u002fj8MAiuHFtn8P9Ei2\u002fl+av4\u002fCKwcWmQ7\u002fz9U46WbxCABQIXrUbgehf8\u002fRIts5\u002fup\u002fT\u002f2KFyPwvX8P2Dl0CLb+fg\u002fhetRuB6F\u002fT8pXI\u002fC9Sj+P3E9CtejcP0\u002fF9nO91Pj\u002fT\u002fVeOkmMQgBQLpJDAIrh\u002fw\u002fqMZLN4lB\u002fD\u002f0\u002fdR46Sb9P\u002fp+arx0k\u002fw\u002f7nw\u002fNV66\u002fz+gGi\u002fdJAb\u002fP1YOLbKd7\u002f0\u002fvHSTGARW+D\u002fwp8ZLN4n5P0oMAiuHFv0\u002fRrbz\u002fdR4\u002fT9t5\u002fup8dL1P4ts5\u002fup8QBARrbz\u002fdR4\u002fT\u002fJdr6fGi\u002f7Pz0K16NwPfQ\u002f1XjpJjEI\u002fD8CK4cW2c77P6jGSzeJQfw\u002fz\u002fdT46Wb+D9CYOXQItv5PwisHFpkO\u002fs\u002fCKwcWmQ7\u002fT\u002fNzMzMzMz8Pylcj8L1KPQ\u002fWmQ730+N8T\u002fhehSuR+H2P7Kd76fGS\u002fs\u002fMzMzMzMz+z8xCKwcWmTzP8uhRbbz\u002ffg\u002fwcqhRbbz+T9OYhBYObT8P\u002fp+arx0k\u002fw\u002fCtejcD0K\u002fT+DwMqhRbbzP7Kd76fGS\u002fU\u002fWmQ730+N+z+R7Xw\u002fNV72P1g5tMh2vvc\u002fvHSTGARW+j\u002fByqFFtvP3P9v5fmq8dPE\u002fXI\u002fC9Shc+z+LbOf7qfH6P3E9CtejcPc\u002f0SLb+X5q+D97FK5H4Xr0P1YOLbKd7\u002fU\u002fEoPAyqFF+D8fhetRuB75Pylcj8L1KPY\u002fEoPAyqFF+j9GtvP91HjzPwrXo3A9CvU\u002fd76fGi\u002fd8D+JQWDl0CL5P9nO91PjpfU\u002fy6FFtvP99j8AAAAAAAAAAC2yne+nxvU\u002fqvHSTWIQ+j+0yHa+nxr3P9NNYhBYOfY\u002fmpmZmZmZ9z\u002fJdr6fGi\u002f1PyUGgZVDi\u002f4\u002fNV66SQwC9z8lBoGVQ4vsP+xRuB6F6+E\u002fRrbz\u002fdR49z+WQ4ts5\u002fvzP+F6FK5H4e4\u002fhxbZzvdT8z+mm8QgsHL2P0jhehSuR\u002fE\u002fZDvfT42X6j9Ei2zn+6ntP\u002fYoXI\u002fC9fY\u002fIbByaJHt9j956SYxCKzwP5MYBFYOLe4\u002fmpmZmZmZ9T+oxks3iUH0P5zEILByaPM\u002ftMh2vp8a8T+8dJMYBFbiP\u002f7UeOkmMeg\u002fc2iR7Xw\u002f8T8AAAAAAADoP7Kd76fGS+M\u002fZDvfT42X8D9MN4lBYOX0P3Noke18P+0\u002fTmIQWDm06D+yne+nxkvvP5MYBFYOLfA\u002fCKwcWmQ78T8CK4cW2c7jP4GVQ4ts5+c\u002ff2q8dJMY5D+oxks3iUHoP\u002f7UeOkmMfQ\u002fDi2yne+n8j\u002fsUbgehev1Py2yne+nxvU\u002fwcqhRbbz8T\u002fy0k1iEFjpPz0K16NwPeo\u002feekmMQis7D\u002f4U+Olm8TsPxSuR+F6FPQ\u002fWDm0yHa+4z8fhetRuB73P1YOLbKd7+c\u002fSgwCK4cW4T8hsHJoke3kP3npJjEIrOg\u002f1XjpJjEI9j9\u002farx0kxjkPw=="},"type":"choropleth"}],                        {"geo":{"showframe":false},"height":500,"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"GDP per capita"},"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('0a119170-30f6-44e3-8624-ad6df403c5f5');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };            </script>        </div>


# 10. Social Support Choropleth Map
An interactive choropleth map is created to visualize the social support levels across different countries. The map displays the social support values, highlighting the varying degrees of social cohesion and support networks worldwide.
- **Locations**: Country names are used to map the data to specific countries.
- **Colorbar**: The colorbar represents the social support values, with colors indicating the relative strength of social support systems in each country.
- **Insights**: This visualization helps identify patterns and correlations between social support and happiness rankings, providing valuable insights into the impact of social relationships on well-being.


```python
data = dict(type = 'choropleth', 
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Explained by: Social support'], 
           text = df['Country'],
           colorbar = {'title':'Social support'})
layout = dict(title = 'Social support', 
             geo = dict(showframe = False),
             width=800,  
             height=500)
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
```


<div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>                <div id="4f2e5485-b3a6-4250-8669-dd98f2f55464" class="plotly-graph-div" style="height:500px; width:800px;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("4f2e5485-b3a6-4250-8669-dd98f2f55464")) {                    Plotly.newPlot(                        "4f2e5485-b3a6-4250-8669-dd98f2f55464",                        [{"colorbar":{"title":{"text":"Social support"}},"locationmode":"country names","locations":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"text":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"z":{"dtype":"f8","bdata":"9P3UeOkm+T9SuB6F61H4P6wcWmQ73\u002fk\u002farx0kxgE+D9oke18PzX4PzEIrBxaZPc\u002fEoPAyqFF+D+uR+F6FK71P83MzMzMzPY\u002fx0s3iUFg9z8730+Nl274PyuHFtnO9\u002fU\u002fbef7qfHS9T\u002fHSzeJQWD1P\u002fLSTWIQWPc\u002fCtejcD0K9z89CtejcD32P5MYBFYOLfg\u002f3SQGgZVD9z+e76fGSzf1P9V46SYxCPg\u002fObTIdr6f8j8Sg8DKoUX2Pz0K16NwPfY\u002fQmDl0CLb8z9MN4lBYOX2P8UgsHJokfU\u002fLbKd76fG9T8730+Nl270Py2yne+nxvU\u002f8KfGSzeJ9T8tsp3vp8bzP0jhehSuR\u002fE\u002fO99PjZdu+D956SYxCKz2P\u002fCnxks3ifc\u002fqMZLN4lB9j+BlUOLbOf1PxKDwMqhRfY\u002ftMh2vp8a9z9zaJHtfD\u002f1P05iEFg5tPI\u002faJHtfD819D\u002fn+6nx0k30P6RwPQrXo\u002fg\u002fVOOlm8Qg+D\u002fn+6nx0k32P39qvHSTGPY\u002fHVpkO99P9z9GtvP91HjzP0SLbOf7qfU\u002fDAIrhxbZ8j+8dJMYBFbyPxKDwMqhRfQ\u002fppvEILBy+D+6SQwCK4f2P1pkO99PjfU\u002ffT81XrpJ8j9t5\u002fup8dLzP4\u002fC9Shcj\u002fA\u002fH4XrUbge9z\u002fRItv5fmr0P0oMAiuHFvU\u002fmpmZmZmZ8T\u002fJdr6fGi\u002f1P+F6FK5H4fI\u002fJzEIrBxa9D\u002fufD81Xrr1PzvfT42XbvQ\u002fBFYOLbKd9T93vp8aL93yP9Ei2\u002fl+avI\u002fbxKDwMqh9z9KDAIrhxb1P5MYBFYOLfg\u002fQmDl0CLb8z+JQWDl0CL1P4ts5\u002fup8fI\u002fGy\u002fdJAaB9z8QWDm0yHbyP4GVQ4ts5\u002fU\u002fO99PjZdu9D91kxgEVg7zP4ts5\u002fup8fI\u002fxSCwcmiR7T+BlUOLbOfjP6jGSzeJQew\u002frkfhehSu7z+sHFpkO9\u002fvP65H4XoUru8\u002f30+Nl24S6z\u002fP91PjpZvwP+XQItv5fuI\u002fgZVDi2zn4z\u002fVeOkmMQjoP5MYBFYOLfI\u002f1XjpJjEI7D8zMzMzMzPxPzEIrBxaZOs\u002fCtejcD0K9T8xCKwcWmTzP76fGi\u002fdJN4\u002fMzMzMzMz4z9Ei2zn+6nlP+xRuB6F6+U\u002fj8L1KFyP5j8X2c73U+PlP8P1KFyPwuk\u002fpHA9Ctej7D+PwvUoXI\u002fuP\u002fyp8dJNYsA\u002f0SLb+X5q8j8EVg4tsp3vP+f7qfHSTfA\u002fgZVDi2zn5z\u002fwp8ZLN4nlP2q8dJMYBOY\u002fvHSTGARW6j+Nl24Sg8DiP9v5fmq8dO8\u002fTDeJQWDl5D+sHFpkO9\u002fvP3e+nxov3fI\u002frBxaZDvfzz9I4XoUrkftP2Q730+Nl+Y\u002fy6FFtvP91D8X2c73U+PpP5qZmZmZme0\u002fPQrXo3A92j81XrpJDALvPzMzMzMzM+s\u002fSOF6FK5H5T\u002fpJjEIrBziPwisHFpkO+s\u002fEFg5tMh24j8AAAAAAAAAAA=="},"type":"choropleth"}],                        {"geo":{"showframe":false},"height":500,"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Social support"},"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('4f2e5485-b3a6-4250-8669-dd98f2f55464');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };            </script>        </div>


# 11. Freedom to Make Life Choices Choropleth Map
An interactive choropleth map is created to visualize the freedom to make life choices across different countries. The map displays the freedom values, highlighting the varying degrees of autonomy and liberty worldwide.
- **Locations**: Country names are used to map the data to specific countries.
- **Colorbar**: The colorbar represents the freedom values, with colors indicating the relative level of freedom to make life choices in each country.
- **Insights**: This visualization helps identify patterns and correlations between freedom and happiness rankings, providing valuable insights into the impact of autonomy on well-being.


```python
data = dict(type = 'choropleth', 
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Explained by: Freedom to make life choices'], 
           text = df['Country'],
           colorbar = {'title':'Freedom to make life choices'})
layout = dict(title = 'Freedom to make life choices', 
             geo = dict(showframe = False),
             width=800,  
             height=500)
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
```


<div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>                <div id="c089fdc2-4d80-479a-99fc-e5792baaf5c5" class="plotly-graph-div" style="height:500px; width:800px;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("c089fdc2-4d80-479a-99fc-e5792baaf5c5")) {                    Plotly.newPlot(                        "c089fdc2-4d80-479a-99fc-e5792baaf5c5",                        [{"colorbar":{"title":{"text":"Freedom to make life choices"}},"locationmode":"country names","locations":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"text":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"z":{"dtype":"f8","bdata":"sHJoke186z+8dJMYBFbqP2iR7Xw\u002fNeo\u002fN4lBYOXQ6j9QjZduEoPkPzMzMzMzM+c\u002fuB6F61G46j9vEoPAyqHpP30\u002fNV66Seg\u002f\u002ftR46SYx6D+sHFpkO9\u002fnPxsv3SQGgek\u002fEFg5tMh26j\u002fl0CLb+X7mP1yPwvUoXOc\u002fhxbZzvdT5z+oxks3iUHoP8l2vp8aL+k\u002fdZMYBFYO4T83iUFg5dDmP3Noke18P+k\u002fFK5H4XoU6j+Nl24Sg8DiP2ZmZmZmZuY\u002ftMh2vp8a5z\u002fNzMzMzMzoP83MzMzMzOQ\u002fyXa+nxov6T\u002fZzvdT46XnPy2yne+nxuc\u002f+FPjpZvE5D+LbOf7qfHmP+kmMQisHOo\u002fw\u002fUoXI\u002fC6T8zMzMzMzPjPwIrhxbZzuM\u002fnu+nxks35T+iRbbz\u002fdTkPwrXo3A9Cuc\u002f3SQGgZVD5z\u002fRItv5fmrgP4XrUbgehec\u002fx0s3iUFg6T\u002fsUbgehevlP+kmMQisHOI\u002fHVpkO99P5T+4HoXrUbjqP5huEoPAyuU\u002f2\u002fl+arx05z8hsHJoke3gP9NNYhBYOeQ\u002fw\u002fUoXI\u002fC4T8730+Nl27qP2Dl0CLb+eo\u002fZDvfT42X4j+e76fGSzfpP\u002f7UeOkmMeg\u002fukkMAiuH6j+6SQwCK4fmPwrXo3A9Cuc\u002farx0kxgE3j\u002fFILByaJHVP6wcWmQ73+M\u002fi2zn+6nx4j9OYhBYObTkP65H4XoUruM\u002fXI\u002fC9Shc5z9QjZduEoPkPz0K16NwPeY\u002fbxKDwMqh4T81XrpJDALnPzEIrBxaZOM\u002f46WbxCCw6j\u002fTTWIQWDnkP9V46SYxCOA\u002fz\u002fdT46Wb5D\u002f6fmq8dJPgPyGwcmiR7eg\u002f16NwPQrX4z\u002fNzMzMzMzkP8l2vp8aL+E\u002fj8L1KFyP4j8EVg4tsp3PPwrXo3A9Ct8\u002fFK5H4XoU5j8j2\u002fl+arzgP7Kd76fGS+c\u002fw\u002fUoXI\u002fC5T8zMzMzMzPbP0w3iUFg5eQ\u002fRrbz\u002fdR46T9QjZduEoPgPxsv3SQGgeE\u002feekmMQis4D8GgZVDi2zjPwaBlUOLbNc\u002fx0s3iUFg5T\u002fpJjEIrBziP3npJjEIrOA\u002f\u002ftR46SYx5D8UrkfhehTeP\u002f7UeOkmMeQ\u002f8tJNYhBY4T+uR+F6FK7jP7bz\u002fdR46d4\u002fwcqhRbbz1T\u002fHSzeJQWDdP+f7qfHSTdo\u002fz\u002fdT46Wb4D+oxks3iUHQP76fGi\u002fdJOI\u002fYhBYObTI4j\u002fNzMzMzMzcPwRWDi2ynes\u002fVg4tsp3v4z9GtvP91HjhP42XbhKDwOI\u002fAAAAAAAA0D\u002fLoUW28\u002f3cP2Dl0CLb+eI\u002fJQaBlUOL6D9cj8L1KFzfPw4tsp3vp+I\u002fzczMzMzM6D\u002fTTWIQWDncP+Olm8QgsOY\u002farx0kxgExj\u002fdJAaBlUPnP5MYBFYOLdI\u002fEoPAyqFF4j++nxov3STiP166SQwCK98\u002fEoPAyqFF3j9qvHSTGATePyPb+X5qvOA\u002fvp8aL90kxj8AAAAAAAAAAA=="},"type":"choropleth"}],                        {"geo":{"showframe":false},"height":500,"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Freedom to make life choices"},"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('c089fdc2-4d80-479a-99fc-e5792baaf5c5');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };            </script>        </div>


# 12. Generosity Choropleth Map
An interactive choropleth map is created to visualize the generosity levels across different countries. The map displays the generosity values, highlighting the varying degrees of charitable behavior and prosocial actions worldwide.
- **Locations**: Country names are used to map the data to specific countries.
- **Colorbar**: The colorbar represents the generosity values, with colors indicating the relative level of generosity in each country.
- **Insights**: This visualization helps identify patterns and correlations between generosity and happiness rankings, providing valuable insights into the impact of prosocial behavior on well-being.


```python
data = dict(type = 'choropleth', 
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Explained by: Generosity'], 
           text = df['Country'],
           colorbar = {'title':'Generosity'})
layout = dict(title = 'Generosity', 
             geo = dict(showframe = False),
             width=800,  
             height=500)
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
```


<div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>                <div id="df865446-811f-4341-8c34-45c47a78d2d3" class="plotly-graph-div" style="height:500px; width:800px;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("df865446-811f-4341-8c34-45c47a78d2d3")) {                    Plotly.newPlot(                        "df865446-811f-4341-8c34-45c47a78d2d3",                        [{"colorbar":{"title":{"text":"Generosity"}},"locationmode":"country names","locations":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"text":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"z":{"dtype":"f8","bdata":"kxgEVg4twj\u002fpJjEIrBzKP1CNl24Sg9A\u002ffT81XrpJzD8v3SQGgZXDPwRWDi2ync8\u002feekmMQiszD\u002fjpZvEILDCP76fGi\u002fdJMY\u002fzczMzMzMzD8hsHJoke3MP4GVQ4ts57s\u002fmpmZmZmZyT8xCKwcWmTLP3E9CtejcM0\u002fw\u002fUoXI\u002fCxT89CtejcD3KPw4tsp3vp8Y\u002fukkMAiuHpj9KDAIrhxbRP39qvHSTGMQ\u002fjZduEoPAyj8lBoGVQ4vMPxKDwMqhRcY\u002farx0kxgEtj+JQWDl0CK7P3npJjEIrLw\u002fyXa+nxovvT+oxks3iUHQPxsv3SQGgcU\u002fnMQgsHJosT\u002fLoUW28\u002f2kP3Noke18P7U\u002fc2iR7Xw\u002fxT8j2\u002fl+ary0PxBYObTIdr4\u002fmpmZmZmZyT\u002fByqFFtvO9P1TjpZvEILA\u002fAAAAAAAA0D956SYxCKy8P+F6FK5H4bo\u002foBov3SQGwT+kcD0K16PAP\u002fp+arx0k7g\u002fqMZLN4lBwD+q8dJNYhDQPxKDwMqhRbY\u002f30+Nl24Swz9YObTIdr6\u002fP1pkO99PjZc\u002fVOOlm8QgwD\u002fy0k1iEFi5P6rx0k1iELg\u002fsHJoke18vz\u002f4U+Olm8TAP+kmMQisHNI\u002fIbByaJHtzD9MN4lBYOXAP2ZmZmZmZsY\u002f\u002fKnx0k1isD8730+Nl26SP7ByaJHtfM8\u002f0SLb+X5qvD9iEFg5tMi2PyuHFtnO97M\u002farx0kxgEtj+wcmiR7Xy\u002fP4PAyqFFtrM\u002fRIts5\u002fupwT+amZmZmZm5PxKDwMqhRbY\u002fzczMzMzMzD8pXI\u002fC9Si8P7x0kxgEVs4\u002fO99PjZdusj\u002f6fmq8dJPIP\u002fCnxks3idk\u002fc2iR7Xw\u002ftT\u002fpJjEIrByqPyuHFtnO97M\u002fokW28\u002f3UyD+yne+nxku3PzeJQWDl0MI\u002fRIts5\u002fupwT9YObTIdr6\u002fP9NNYhBYOcQ\u002fAAAAAAAAAAA\u002fNV66SQzCP42XbhKDwMo\u002fw\u002fUoXI\u002fCxT8K16NwPQqnP7gehetRuL4\u002f4XoUrkfhyj\u002fb+X5qvHTDPxSuR+F6FM4\u002feekmMQisvD\u002fufD81XrrJP1TjpZvEIMA\u002fPQrXo3A90j9Ei2zn+6mxPxsv3SQGgaU\u002fO99PjZduwj+PwvUoXI\u002fCP76fGi\u002fdJMY\u002foBov3SQGwT8j2\u002fl+arzUP4GVQ4ts58s\u002fPzVeukkM0j\u002f6fmq8dJOYP3npJjEIrLw\u002fYhBYObTIxj9Ei2zn+6nZP8P1KFyPwsU\u002fBoGVQ4tsxz9iEFg5tMjGP7gehetRuL4\u002farx0kxgExj+oxks3iUHAP2iR7Xw\u002fNa4\u002fEoPAyqFFxj+amZmZmZmZPzvfT42XbsI\u002f7FG4HoXrwT9I4XoUrkfRP6abxCCwcsg\u002f\u002fKnx0k1iwD8bL90kBoHFP2iR7Xw\u002fNa4\u002fSOF6FK5HwT956SYxCKyMP\u002fp+arx0k7g\u002f\u002ftR46SYxyD9eukkMAivHP8uhRbbz\u002fbQ\u002fnMQgsHJosT+yne+nxku3Pw=="},"type":"choropleth"}],                        {"geo":{"showframe":false},"height":500,"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Generosity"},"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('df865446-811f-4341-8c34-45c47a78d2d3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };            </script>        </div>


# 13. Perceptions of Corruption Choropleth Map
An interactive choropleth map is created to visualize the perceptions of corruption levels across different countries. The map displays the corruption perception values, highlighting the varying degrees of perceived corruption worldwide.
- **Locations**: Country names are used to map the data to specific countries.
- **Colorbar**: The colorbar represents the corruption perception values, with colors indicating the relative level of perceived corruption in each country.
- **Insights**: This visualization helps identify patterns and correlations between corruption perceptions and happiness rankings, providing valuable insights into the impact of governance and institutional trust on well-being.


```python
data = dict(type = 'choropleth', 
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Explained by: Perceptions of corruption'], 
           text = df['Country'],
           colorbar = {'title':'Perceptions of corruption'})
layout = dict(title = 'Perceptions of corruption', 
             geo = dict(showframe = False),
             width=800,  
             height=500)
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
```


<div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>                <div id="6b79bcfd-d407-40eb-b94b-33b55cc1276b" class="plotly-graph-div" style="height:500px; width:800px;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("6b79bcfd-d407-40eb-b94b-33b55cc1276b")) {                    Plotly.newPlot(                        "6b79bcfd-d407-40eb-b94b-33b55cc1276b",                        [{"colorbar":{"title":{"text":"Perceptions of corruption"}},"locationmode":"country names","locations":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"text":["Finland","Denmark","Iceland","Sweden","Israel","Netherlands","Norway","Luxembourg","Switzerland","Australia","New Zealand","Costa Rica","Kuwait","Austria","Canada","Belgium","Ireland","Czechia","Lithuania","United Kingdom","Slovenia","United Arab Emirates","United States","Germany","Mexico","Uruguay","France","Saudi Arabia","Kosovo","Singapore","Taiwan Province of China","Romania","El Salvador","Estonia","Poland","Spain","Serbia","Chile","Panama","Malta","Italy","Guatemala","Nicaragua","Brazil","Slovakia","Latvia","Uzbekistan","Argentina","Kazakhstan","Cyprus","Japan","South Korea","Philippines","Vietnam","Hungary","Paraguay","Thailand","Malaysia","China","Honduras","Croatia","Greece","Bosnia and Herzegovina","Libya","Jamaica","Peru","Dominican Republic","Mauritius","Moldova","Russia","Bolivia","Ecuador","Kyrgyzstan","Montenegro","Mongolia","Colombia","Venezuela","Indonesia","Bulgaria","Armenia","South Africa","North Macedonia","Algeria","Hong Kong S.A.R. of China","Albania","Congo (Brazzaville)","Mozambique","Georgia","Iraq","Nepal","Laos","Gabon","Ivory Coast","Guinea","Senegal","Iran","Azerbaijan","Nigeria","Cameroon","Ukraine","Namibia","Morocco","Pakistan","Niger","Burkina Faso","Mauritania","Gambia","Chad","Kenya","Tunisia","Benin","Uganda","Myanmar","Cambodia","Ghana","Liberia","Mali","Madagascar","Togo","Jordan","India","Egypt","Sri Lanka","Bangladesh","Ethiopia","Tanzania","Comoros","Zambia","Eswatini","Malawi","Botswana","Zimbabwe","Congo (Kinshasa)","Sierra Leone","Lesotho","Lebanon","Afghanistan"],"z":{"dtype":"f8","bdata":"Rrbz\u002fdR44T\u002fwp8ZLN4nhP7Kd76fGS8c\u002f+FPjpZvE4D9OYhBYObTIPwIrhxbZztc\u002fYOXQItv53j\u002fZzvdT46XbP6wcWmQ7398\u002feekmMQis1D+4HoXrUbjeP7ByaJHtfL8\u002farx0kxgExj+F61G4HoXTP1pkO99Pjdc\u002fgZVDi2zn0z+Nl24Sg8DaP5zEILByaLE\u002fGQRWDi2yvT8QWDm0yHbWP\u002fhT46WbxMA\u002fUI2XbhKD0D9vEoPAyqHFP1pkO99Pjdc\u002fqMZLN4lBwD8pXI\u002fC9SjMP5ZDi2zn+9E\u002fqvHSTWIQyD\u002fjpZvEILCyP2ZmZmZmZuI\u002fQmDl0CLbyT\u002f6fmq8dJN4P\u002f7UeOkmMdA\u002fRIts5\u002fup2T9mZmZmZmbGPw4tsp3vp8Y\u002fQmDl0CLbuT8zMzMzMzOzP2q8dJMYBKY\u002fAAAAAAAAwD+LbOf7qfGyPyuHFtnO97M\u002fqvHSTWIQ0D+TGARWDi3CPxkEVg4tsq0\u002fK4cW2c73sz+F61G4HoXLP3sUrkfherQ\u002fuB6F61G4vj9KDAIrhxapP9V46SYxCMw\u002f001iEFg5xD+cxCCwcmjBP3sUrkfhesQ\u002f9P3UeOkmsT+kcD0K16OwP\u002fp+arx0k5g\u002fEFg5tMh2vj\u002fLoUW28\u002f3EPyPb+X5qvLQ\u002farx0kxgEpj8CK4cW2c63PwAAAAAAAAAA6SYxCKwcyj956SYxCKycPxkEVg4tsp0\u002fSgwCK4cWyT9oke18PzW+P7pJDAIrh6Y\u002fYOXQItv5vj8IrBxaZDuvPyuHFtnO97M\u002fuB6F61G4nj9MN4lBYOXAPylcj8L1KKw\u002faJHtfD81rj9qvHSTGAS2Pylcj8L1KKw\u002f+n5qvHSTeD++nxov3STGP5zEILByaKE\u002fuB6F61G4jj+amZmZmZnJP+58PzVeutk\u002fSgwCK4cWqT9Ei2zn+6nBP0oMAiuHFsk\u002fEoPAyqFFxj\u002f6fmq8dJOoP3E9CtejcL0\u002fx0s3iUFgxT+amZmZmZm5P8uhRbbz\u002fcQ\u002fMQisHFpkuz9Ei2zn+6mxP7ByaJHtfL8\u002fRrbz\u002fdR4yT\u002fb+X5qvHSTP7gehetRuK4\u002fmpmZmZmZmT8IrBxaZDuvP8uhRbbz\u002fbQ\u002fi2zn+6nxsj83iUFg5dDCP7bz\u002fdR46cY\u002f8tJNYhBYyT\u002f6fmq8dJOoPyGwcmiR7bw\u002fRIts5\u002fupsT8730+Nl26SP1TjpZvEINA\u002f2c73U+Olqz8Sg8DKoUXGP5MYBFYOLbI\u002feekmMQisnD8zMzMzMzOzPwrXo3A9Crc\u002fsHJoke18vz8rhxbZzvfDP\u002f7UeOkmMcg\u002fCKwcWmQ7vz\u002f6fmq8dJPQP1g5tMh2vp8\u002fx0s3iUFgxT9CYOXQItu5P6abxCCwctA\u002fexSuR+F6xD+BlUOLbOe7PxkEVg4tsr0\u002fnMQgsHJowT\u002fLoUW28\u002f20P\u002fhT46WbxMA\u002fO99PjZdusj+JQWDl0CKrP8P1KFyPwrU\u002fGQRWDi2ynT+6SQwCK4e2Pw=="},"type":"choropleth"}],                        {"geo":{"showframe":false},"height":500,"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Perceptions of corruption"},"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('6b79bcfd-d407-40eb-b94b-33b55cc1276b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };            </script>        </div>


# 14. Scatter Plots for Happiness Score vs. Contributing Factors
A series of scatter plots are created to visualize the relationships between the happiness score and various contributing factors.
- **First row**:
  - GDP per capita vs. Happiness score
  - Social support vs. Happiness score
  - Healthy life expectancy vs. Happiness score
- **Second row**:
  - Freedom to make life choices vs. Happiness score
  - Generosity vs. Happiness score
  - Perceptions of corruption vs. Happiness score
These plots help identify correlations and patterns between the happiness score and each contributing factor, providing insights into the factors that most strongly influence happiness.


```python
plt.figure(figsize=(24,8))

plt.subplot(1,3,1)
plt.scatter( df['Ladder score'], df['Explained by: Log GDP per capita'])
plt.xlabel('GDP per capita')
plt.ylabel('Happiness score')

plt.subplot(1,3,2)
plt.scatter( df['Ladder score'], df['Explained by: Social support'])
plt.xlabel('Social support')
plt.ylabel('Happiness score')

plt.subplot(1,3,3)
plt.scatter( df['Ladder score'], df['Explained by: Healthy life expectancy'])
plt.xlabel('Healthy life expectancy')
plt.ylabel('Happiness score')

plt.figure(figsize=(24,8))
plt.subplot(1,3,1)
plt.scatter( df['Ladder score'], df['Explained by: Freedom to make life choices'])
plt.xlabel('Freedom to make life choices')
plt.ylabel('Happiness score')

plt.subplot(1,3,2)
plt.scatter( df['Ladder score'], df['Explained by: Generosity'])
plt.xlabel('Generosity')
plt.ylabel('Happiness score')

plt.subplot(1,3,3)
plt.scatter( df['Ladder score'], df['Explained by: Perceptions of corruption'])
plt.xlabel('Perceptions of corruption')
plt.ylabel('Happiness score')
```




    Text(0, 0.5, 'Happiness score')




    
![png](output_30_1.png)
    



    
![png](output_30_2.png)
    


# 15. Pairplot with Regression Lines
A pairplot is created to visualize the relationships between the various factors contributing to happiness. The plot includes regression lines to highlight the correlations between each pair of factors.
- **Diagonal plots**: Histograms of each factor
- **Upper and lower triangle plots**: Scatter plots with regression lines showing the relationships between each pair of factors
This visualization helps identify patterns, correlations, and potential relationships between the factors, providing insights into the complex interactions driving happiness.


```python
correlations = df[['Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy',
                                 'Explained by: Freedom to make life choices', 'Explained by: Generosity','Explained by: Perceptions of corruption',
                                 'Dystopia + residual']]
sns.pairplot(correlations, kind='reg')
```




    <seaborn.axisgrid.PairGrid at 0x7f8fe414de80>




    
![png](output_32_1.png)
    


# 16. Multivariate Scatter Plots
A series of multivariate scatter plots are created to visualize the relationships between various factors contributing to happiness. Each plot includes a third variable as the hue, adding an additional dimension to the visualization.
- **Plot 1**: Ladder score vs GDP per capita, colored by freedom to make life choices
- **Plot 2**: Healthy life expectancy vs social support, colored by ladder score
- **Plot 3**: Social support vs GDP per capita, colored by ladder score
- **Plot 4**: Perceptions of corruption vs freedom to make life choices, colored by GDP per capita
These plots help identify complex relationships and patterns between multiple factors, providing insights into the interactions driving happiness.


```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.scatterplot(x='Ladder score', y='Explained by: Log GDP per capita', hue='Explained by: Freedom to make life choices', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Ladder score vs GDP per capita')

sns.scatterplot(x='Explained by: Healthy life expectancy', y='Explained by: Social support', hue='Ladder score', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Healthy life expectancy vs Social support')

sns.scatterplot(x='Explained by: Social support', y='Explained by: Log GDP per capita', hue='Ladder score', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Social support vs GDP per capita')

sns.scatterplot(x='Explained by: Perceptions of corruption', y='Explained by: Freedom to make life choices', hue='Explained by: Log GDP per capita', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Perceptions of corruption vs Freedom to make life choices')

plt.tight_layout()
plt.show()
```


    
![png](output_34_0.png)
    


# 17. Pairplot with KDE and Ladder Score
A pairplot is created to visualize the relationships between the ladder score and age group scores. The plot includes kernel density estimates (KDE) on the diagonal and scatter plots in the upper and lower triangles.
- **Diagonal plots**: KDE plots showing the distribution of each age group score and ladder score
- **Upper and lower triangle plots**: Scatter plots showing the relationships between each pair of age group scores and ladder score
This visualization helps identify patterns, correlations, and potential relationships between age groups and happiness, providing insights into how happiness varies across different age groups.


```python
sns.pairplot(df[['Age Below 30 Score', 'Age 30-44 Score', 'Age 45-59 Score', 'Age Above 60 Score', 'Ladder score']], 
             diag_kind='kde', palette='coolwarm')
plt.show()
```

    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1513: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1513: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1513: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1513: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1513: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    
    /mnt/f/venv/lib/python3.13/site-packages/seaborn/axisgrid.py:1615: UserWarning:
    
    Ignoring `palette` because no `hue` variable has been assigned.
    



    
![png](output_36_1.png)
    


# 18. Correlation Heatmap
A correlation heatmap is created to visualize the relationships between different age groups' scores and the ladder score. The heatmap displays the correlation coefficients, providing insights into the strength and direction of the relationships.
- **Color scheme**: The 'coolwarm' colormap is used to represent positive and negative correlations.
- **Annotations**: Correlation coefficients are displayed on the heatmap for easy interpretation.
This visualization helps identify patterns and correlations between age groups and happiness, informing further analysis and modeling efforts.


```python
plt.figure(figsize=(10, 6))
corr = df[['Age Below 30 Score', 'Age 30-44 Score', 'Age 45-59 Score', 'Age Above 60 Score', 'Ladder score']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation between Age Groups and Ladder Score', fontsize=16)
plt.show()
```


    
![png](output_38_0.png)
    


# 19. Linear Regression Plot
A linear regression plot is created to visualize the relationship between the ladder score and GDP per capita. The plot displays the data points and the fitted linear regression line.
- **Data points**: The scatter plot represents the relationship between ladder score and GDP per capita.
- **Linear fit**: The black line represents the fitted linear regression model, with the equation displayed in the legend.
This visualization helps identify the strength and direction of the relationship between ladder score and GDP per capita, providing insights into the impact of economic factors on happiness.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data creation (uncomment the following line and use your DataFrame instead)
# df = pd.read_csv('your_dataset.csv')  # Load your dataset

# Example variables
x = df['Ladder score']
y = df['Explained by: Log GDP per capita']

# Perform linear fit
m, c = np.polyfit(x, y, 1)

# Set figure and axes colors
plt.style.use('ggplot')
plt.figure(figsize=(12, 8), facecolor='lightgrey')  # Change figure background color here

plt.scatter(x, y, label="Data points", color='skyblue', s=100, edgecolor='white', alpha=0.7)
plt.plot(x, m * x + c, color='black', linewidth=2.5, label=f"Fit: y = {m:.2f}x + {c:.2f}")

# Set axes background color
plt.gca().set_facecolor('whitesmoke')  # Change axes background color here

# Title and labels
plt.title('Ladder score vs. GDP per capita - Linear Regression', fontsize=18, weight='bold', pad=15)
plt.xlabel('Ladder score', fontsize=14)
plt.ylabel('Explained By: Log GDP per capita', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.8)
plt.legend(frameon=True, fontsize=12, loc='upper left', fancybox=True, shadow=True, borderpad=1)

# Show the plot
plt.show()
```


    
![png](output_40_0.png)
    


# 20. Histogram with KDE
A histogram is created to visualize the distribution of ladder scores. The plot includes a kernel density estimate (KDE) curve, providing insights into the shape of the distribution.
- **Custom colors**: The plot uses custom colors for the histogram, title, and labels.
- **KDE curve**: The KDE curve provides a smoothed representation of the distribution.
This visualization helps understand the distribution of ladder scores, identifying patterns such as skewness or multimodality.


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style and background color
sns.set_style("whitegrid")  # You can use 'darkgrid', 'white', 'dark', 'ticks', etc.
plt.figure(figsize=(10, 6), facecolor='white')  # Change facecolor to your desired background color

# Create the histogram with custom colors
sns.histplot(df['Ladder score'], bins=10, kde=True, color='teal')  # Change 'teal' to your preferred color

# Set the title and labels
plt.title('Distribution of Ladder Scores', color='navy')  # Title color
plt.xlabel('Ladder Score', color='navy')  # X-axis label color
plt.ylabel('Frequency', color='navy')  # Y-axis label color

# Show the plot
plt.show()
```


    
![png](output_42_0.png)
    


# 21. Bubble Plot
A bubble plot is created to visualize the relationship between GDP per capita and ladder score, with bubble size representing healthy life expectancy.
- **Bubble size**: The size of each bubble represents the healthy life expectancy value.
- **Relationship**: The plot shows the relationship between GDP per capita and ladder score.
This visualization helps identify patterns and correlations between economic factors, health outcomes, and happiness.


```python
import matplotlib.pyplot as plt

# Create the scatter plot
plt.figure(figsize=(10, 6), facecolor='white')  # Change figure background color
plt.scatter(
    df['Explained by: Log GDP per capita'], 
    df['Ladder score'], 
    s=df['Explained by: Healthy life expectancy'] * 100, 
    alpha=0.5, 
    color='green'  # Change marker color
)

# Set axis background color
plt.gca().set_facecolor('white')  # Change axis background color

# Add titles and labels
plt.title('Ladder Score vs. GDP per Capita (Bubble Size: Healthy Life Expectancy)', fontsize=14)
plt.xlabel('Explained By: Log GDP per Capita', fontsize=12)
plt.ylabel('Ladder Score', fontsize=12)

# Show the plot
plt.show()
```


    
![png](output_44_0.png)
    


# 22. Bubble Plot: Healthy Life Expectancy vs GDP per Capita
A bubble plot is created to visualize the relationship between healthy life expectancy and GDP per capita, with bubble size representing generosity.
- **Bubble size**: The size of each bubble represents the generosity value.
- **Relationship**: The plot shows the relationship between healthy life expectancy and GDP per capita.
This visualization helps identify patterns and correlations between health outcomes, economic factors, and generosity.


```python
import matplotlib.pyplot as plt

# Create a scatter plot with custom point colors and background color
plt.figure(figsize=(10, 6))

# Set the background color
plt.gca().set_facecolor('white')  # Change to your desired background color

# Create scatter plot with custom colors for points
scatter = plt.scatter(df['Explained by: Log GDP per capita'], 
                      df['Explained by: Healthy life expectancy'],
                      s=df['Explained by: Generosity']*200, alpha=0.7, color='teal')  # Change 'orange' to your desired point color

# Add title and labels
plt.title('Healthy Life Expectancy vs. GDP per Capita (Bubble Size: Generosity)', fontsize=14)
plt.xlabel('Explained by: Log GDP per Capita', fontsize=12)
plt.ylabel('Explained by: Healthy Life Expectancy', fontsize=12)

# Show the plot
plt.show()
```


    
![png](output_46_0.png)
    


# 23. Bubble Plot: Social Support vs Freedom to Make Life Choices
A bubble plot is created to visualize the relationship between social support and freedom to make life choices, with bubble size representing healthy life expectancy.
- **Bubble size**: The size of each bubble represents the healthy life expectancy value.
- **Relationship**: The plot shows the relationship between social support and freedom to make life choices.
This visualization helps identify patterns and correlations between social factors, personal freedoms, and health outcomes.


```python
import matplotlib.pyplot as plt

# Create a scatter plot with custom point color and background color
plt.figure(figsize=(10, 6))

# Change point color to blue and background color to light gray
plt.scatter(df['Explained by: Social support'], df['Explained by: Freedom to make life choices'], 
            s=df['Explained by: Healthy life expectancy'] * 100, alpha=0.5, color='teal')

# Change the background color
plt.gca().set_facecolor('white')

plt.title('Social Support vs. Freedom to Make Life Choices (Bubble Size: Healthy Life Expectancy)')
plt.xlabel('Explained by: Social Support')
plt.ylabel('Explained by: Freedom to Make Life Choices')
plt.grid(True, color='white')  # Optional: Change grid color for better visibility
plt.show()
```


    
![png](output_48_0.png)
    


# 24. Linear Regression Model
A linear regression model is created to predict the ladder score based on various factors, including GDP per capita, healthy life expectancy, social support, freedom to make life choices, generosity, perceptions of corruption, and age group scores.
- **Features**: The model uses 10 features, including economic, health, social, and age-related variables.
- **Target variable**: The ladder score is the target variable being predicted.
- **Model performance**: The model is evaluated using mean squared error (MSE) and R² score.


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare your DataFrame (df) as needed
# Selecting independent variables (features) and dependent variable (target)
X = df[['Explained by: Log GDP per capita', 'Explained by: Healthy life expectancy', 'Explained by: Social support', 
         'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption',
         'Age Below 30 Score', 'Age 30-44 Score', 'Age 45-59 Score', 'Age Above 60 Score']]
y = df['Ladder score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the coefficients and model performance metrics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

    Coefficients: [ 0.00706822 -0.00299334  0.01144377  0.0142918   0.00192976 -0.01314055
      0.24788695  0.44365612  0.05721071  0.24899345]
    Intercept: -0.017371175749815393
    Mean Squared Error: 0.00012559600595341672
    R² Score: 0.9999057835189213


# 25. Scatter Plots with Regression Lines
A series of scatter plots are created to visualize the relationships between each feature and the ladder score, with regression lines added to illustrate the predicted relationships.
- **Customizable plot styles**: The plots use custom colors for the scatter points, regression lines, and background.
- **Regression lines**: The regression lines are based on the predictions from the linear regression model.
This visualization helps identify the relationships between each feature and the ladder score, providing insights into the drivers of happiness.


```python
import matplotlib.pyplot as plt
import numpy as np

# Function to plot scatter plot with regression line
def plot_regression(X, y, feature_name, scatter_color='blue', line_color='red', bg_color='lightgrey'):
    plt.figure(figsize=(10, 6))
    
    # Set background color
    plt.gca().set_facecolor(bg_color)
    
    # Scatter plot
    plt.scatter(X[feature_name], y, color=scatter_color, alpha=0.5, label='Data Points')
    
    # Create the regression line
    X_fit = np.linspace(X[feature_name].min(), X[feature_name].max(), 100).reshape(-1, 1)
    X_fit_full = pd.DataFrame(
    np.zeros((100, X.shape[1])),
    columns=X.columns
    )
    X_fit_full[feature_name] = X_fit.flatten()
    y_fit = model.predict(X_fit_full)
    
    # Regression line
    plt.plot(X_fit, y_fit, color=line_color, label='Regression Line')
    
    plt.title(f'{feature_name} vs. Ladder Score with Regression Line')
    plt.xlabel(feature_name)
    plt.ylabel('Ladder Score')
    plt.legend()
    plt.grid()
    plt.show()

# Plot regression for each feature with custom colors
for feature in X.columns:
    plot_regression(X, y, feature, scatter_color='blue', line_color='black', bg_color='white')
```


    
![png](output_52_0.png)
    



    
![png](output_52_1.png)
    



    
![png](output_52_2.png)
    



    
![png](output_52_3.png)
    



    
![png](output_52_4.png)
    



    
![png](output_52_5.png)
    



    
![png](output_52_6.png)
    



    
![png](output_52_7.png)
    



    
![png](output_52_8.png)
    



    
![png](output_52_9.png)
    


# 26. Histogram of Residuals
A histogram is created to visualize the distribution of residuals from the linear regression model.
- **Residuals**: The differences between actual and predicted values.
- **Distribution**: The histogram shows the shape of the residual distribution.
This visualization helps evaluate the model's performance and identify potential issues, such as non-normality or outliers.


```python
# Predict y using the trained model
y_pred = model.predict(X)

# Calculate residuals
residuals = y - y_pred

# Now plot the histogram
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.gcf().set_facecolor('white')  # Background color
plt.hist(residuals, bins=30, color='pink', alpha=0.7)
plt.title('Histogram of Residuals', fontsize=14)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(color='white', linestyle='--', linewidth=0.5)
plt.show()
```


    
![png](output_54_0.png)
    


# 27. Coefficients of Features
A horizontal bar chart is created to visualize the coefficients of the features in the linear regression model.
- **Coefficient values**: The chart shows the magnitude and direction of each feature's impact on the target variable.
- **Feature importance**: The chart helps identify the most important features in the model.
This visualization provides insights into the relationships between the features and the target variable.


```python
# Plotting coefficients
import matplotlib.pyplot as plt

# Sample coefficients for demonstration; replace with actual model coefficients
# model.coef_ = [...]  # Uncomment this line and assign actual model coefficients

# Create a horizontal bar chart with customized colors
plt.figure(figsize=(10, 6))
plt.barh(X.columns, model.coef_, color='#1f77b4', alpha=0.7)  # Change bar color to a different shade
plt.title('Coefficients of Features', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Features', fontsize=14)

# Change background color
plt.gca().set_facecolor('#f0f0f0')  # Light grey background color

plt.grid(color='white', linestyle='--', linewidth=0.5)  # Optional: customize grid color and style
plt.show()
```


    
![png](output_56_0.png)
    


# 28. Random Forest Regression
A random forest regression model is created to predict the ladder score based on various factors, including GDP per capita, healthy life expectancy, social support, freedom to make life choices, generosity, perceptions of corruption, and age group scores.
- **Feature scaling**: The features are scaled using StandardScaler to improve model performance.
- **Model evaluation**: The model is evaluated using mean squared error (MSE) and R² score.


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('final_data.csv')  # Uncomment and set your file path

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (or handle them as needed)
df.dropna(inplace=True)

# Selecting independent variables (features) and dependent variable (target)
X = df[['Explained by: Log GDP per capita', 'Explained by: Healthy life expectancy', 'Explained by: Social support', 
         'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption',
         'Age Below 30 Score','Age 30-44 Score', 'Age 45-59 Score', 'Age Above 60 Score']]
y = df['Ladder score']
# Scale f Scoreeatures (opti Scoreonal butAbove  reScorended for tree-based models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

    Country                                       0
    Rank                                          0
    Ladder score                                  0
    upperwhisker                                  0
    lowerwhisker                                  0
    Explained by: Log GDP per capita              3
    Explained by: Social support                  3
    Explained by: Healthy life expectancy         3
    Explained by: Freedom to make life choices    3
    Explained by: Generosity                      3
    Explained by: Perceptions of corruption       3
    Dystopia + residual                           3
    Age Below 30 Score                            0
    Age Above 60 Score                            0
    Age 30-44 Score                               0
    Age 45-59 Score                               0
    dtype: int64


# 29. Training the Random Forest Model
A random forest regressor model is trained on the training data to predict the ladder score.
- **Model parameters**: The model uses 100 estimators and a random state of 42 for reproducibility.
- **Training data**: The model is trained on the scaled training data.
This step is crucial in developing a predictive model that can accurately forecast ladder scores based on various factors.


```python
# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(random_state=42)</pre></div> </div></div></div></div>



# 30. Making Predictions
The trained random forest regressor model is used to make predictions on the testing data.
- **Predictions**: The model generates predicted values for the ladder score based on the input features.
- **Output**: The predicted values are printed to the console.
This step is essential in evaluating the model's performance and identifying areas for improvement.


```python
# Make predictions
y_pred = rf_model.predict(X_test)

# Print the predictions
print("Predictions:", y_pred)
```

    Predictions: [4.41669 4.42147 6.86695 6.68636 3.80465 5.84136 5.99686 4.021   3.20045
     5.17469 5.18664 6.19094 5.75631 5.59506 6.36315 6.23574 6.66314 6.75806
     3.39547 5.79122 3.76406 7.35991 4.34696 6.09035 5.1034  4.97443 6.84267
     6.7283 ]


# 31. Evaluating Model Performance
The performance of the random forest regressor model is evaluated using mean squared error (MSE) and R² score.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **R² Score**: Measures the proportion of variance in the target variable explained by the model.
This step provides insights into the model's accuracy and goodness of fit.


```python
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
```

    Mean Squared Error: 0.014531954803571367
    R² Score: 0.9890987803760658


# 32. Feature Importance Plot
A bar chart is created to visualize the feature importance of the random forest regressor model.
- **Feature importance**: The chart shows the relative importance of each feature in predicting the target variable.
- **Sorted features**: The features are sorted in descending order of importance.
This visualization helps identify the most influential features in the model.


```python
# Plotting feature importance
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances[indices], align="center", color='brown')  # Change bar color here
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices], rotation=90)
plt.xlim([-1, X.shape[1]])

# Change the background color
plt.gca().set_facecolor('white')  # Change to your desired background color

plt.ylabel('Importance', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.show()
```


    
![png](output_66_0.png)
    


# 33. Actual vs. Predicted Values Plot
A scatter plot is created to visualize the relationship between actual and predicted ladder scores.
- **Actual values**: The x-axis represents the actual ladder scores.
- **Predicted values**: The y-axis represents the predicted ladder scores.
- **Perfect prediction line**: A dashed line represents perfect predictions, where actual and predicted values are equal.
This visualization helps evaluate the model's performance and identify patterns or biases.


```python
# Plot actual vs predicted values
import matplotlib.pyplot as plt

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.5)  # Change point color to green
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='orange', linestyle='--')  # Change line color to orange

# Set background color
plt.gca().set_facecolor('white')  # Change background color to light grey

# Set title and labels
plt.title('Actual vs. Predicted Ladder Score', fontsize=16)
plt.xlabel('Actual Ladder Score', fontsize=14)
plt.ylabel('Predicted Ladder Score', fontsize=14)

# Show grid
plt.grid(color='white')  # Change grid color to white for better visibility

# Show the plot
plt.show()
```


    
![png](output_68_0.png)
    


# 34. Classification Problem Setup
A classification problem is set up by converting the continuous 'Ladder score' into a categorical variable 'Ladder Score Category'.
- **Categorical variable**: The 'Ladder Score Category' is created using quantiles with labels 'Low', 'Medium', 'High', and 'Very High'.
- **Features**: The independent variables include factors such as GDP per capita, healthy life expectancy, social support, and age group scores.
- **Target variable**: The target variable is the categorical 'Ladder Score Category'.
This setup allows for classification modeling to predict the ladder score category based on the input features.


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv('final_data.csv')  # Uncomment and set your file path

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (or handle them as needed)
df.dropna(inplace=True)

# Convert Ladder Score to categorical (for example, using quantiles)
bins = [0, 4, 6, 8, 10]  # Define bins (adjust according to your data)
labels = ['Low', 'Medium', 'High', 'Very High']  # Define class labels
df['Ladder Score Category'] = pd.cut(df['Ladder score'], bins=bins, labels=labels)

# Selecting independent variables (features) and dependent variable (target)
X = df[['Explained by: Log GDP per capita', 'Explained by: Healthy life expectancy', 'Explained by: Social support', 
         'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption',
         'Age Below 30 Score','Age 30-44 Score', 'Age 45-59 Score', 'Age Above 60 Score']]
y = df['Ladder score']

# Scale features (optional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

    Country                                       0
    Rank                                          0
    Ladder score                                  0
    upperwhisker                                  0
    lowerwhisker                                  0
    Explained by: Log GDP per capita              3
    Explained by: Social support                  3
    Explained by: Healthy life expectancy         3
    Explained by: Freedom to make life choices    3
    Explained by: Generosity                      3
    Explained by: Perceptions of corruption       3
    Dystopia + residual                           3
    Age Below 30 Score                            0
    Age Above 60 Score                            0
    Age 30-44 Score                               0
    Age 45-59 Score                               0
    dtype: int64


# 35. Binary Classification
A binary classification problem is created by classifying the ladder scores based on the median value.
- **Median threshold**: The median value of the training data is used as a threshold to create binary labels.
- **Binary labels**: The labels are converted to binary (0 or 1) based on whether the score is above or below the median.
This approach allows for training a classifier to predict whether a score is above or below the median.


```python
# For example, classify based on median value
median_score = y_train.median()

# Create binary labels
y_train_class = (y_train > median_score).astype(int)
y_test_class = (y_test > median_score).astype(int)

# Now train classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train_class)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>DecisionTreeClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeClassifier.html">?<span>Documentation for DecisionTreeClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(random_state=42)</pre></div> </div></div></div></div>



# 36. Multi-Class Classification
A multi-class classification problem is created by categorizing the ladder scores into three categories: 'Low', 'Medium', and 'High'.
- **Quantile-based categorization**: The categories are defined based on the 33rd and 66th percentiles of the training data.
- **Decision Tree Classifier**: A decision tree classifier is trained to predict the category of the ladder score.
This approach allows for predicting the category of the ladder score based on the input features.


```python
# Create 3 categories based on y_train values
y_train_class = pd.cut(y_train, bins=[-np.inf, y_train.quantile(0.33), y_train.quantile(0.66), np.inf], 
                       labels=['Low', 'Medium', 'High'])

y_test_class = pd.cut(y_test, bins=[-np.inf, y_test.quantile(0.33), y_test.quantile(0.66), np.inf], 
                      labels=['Low', 'Medium', 'High'])

# Train the classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train_class)

# Predict
y_pred = dt_classifier.predict(X_test)

print("Predictions:", y_pred)

```

    Predictions: ['Low' 'Low' 'High' 'High' 'Low' 'Medium' 'Medium' 'Low' 'Low' 'Low' 'Low'
     'Medium' 'Medium' 'Medium' 'High' 'Medium' 'High' 'High' 'Low' 'Medium'
     'Low' 'High' 'Low' 'Medium' 'Medium' 'Low' 'High' 'High']


# 37. Evaluating Multi-Class Classification Performance
The performance of the decision tree classifier is evaluated using a classification report and a confusion matrix.
- **Classification Report**: The report provides precision, recall, and F1-score for each category.
- **Confusion Matrix**: The matrix shows the number of true positives, false positives, true negatives, and false negatives for each category.
This evaluation provides insights into the classifier's performance and identifies areas for improvement.


```python
# Categorize y_test the same way as training
y_test_class = pd.cut(
    y_test, 
    bins=[-np.inf, y_test.quantile(0.33), y_test.quantile(0.66), np.inf], 
    labels=['Low', 'Medium', 'High']
)

# Now y_test_class and y_pred have the same format (both categorical)

# Print classification report
print("Classification Report:\n", classification_report(y_test_class, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test_class, y_pred)
print("Confusion Matrix:\n", conf_matrix)

```

    Classification Report:
                   precision    recall  f1-score   support
    
            High       1.00      0.80      0.89        10
             Low       0.82      1.00      0.90         9
          Medium       0.78      0.78      0.78         9
    
        accuracy                           0.86        28
       macro avg       0.87      0.86      0.86        28
    weighted avg       0.87      0.86      0.86        28
    
    Confusion Matrix:
     [[8 0 2]
     [0 9 0]
     [0 2 7]]


# 38. Visualizing the Decision Tree
The decision tree classifier is visualized using a tree plot to understand its structure and decision-making process.
- **Feature names**: The feature names are displayed in the plot to understand the decision-making process.
- **Class names**: The class names ('Low', 'Medium', 'High') are displayed in the plot to understand the predicted categories.
This visualization helps understand the decision-making process of the classifier.


```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plotting the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, feature_names=X.columns, class_names=labels, filled=True, rounded=True)
plt.title('Decision Tree Classifier')
plt.show()
```


    
![png](output_78_0.png)
    


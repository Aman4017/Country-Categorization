import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.cluster import KMeans
import geopandas as gpd
from matplotlib.colors import ListedColormap
import scipy.cluster.hierarchy as shc


"""#Data Preprocessing

###Any necessary data cleaning or preprocessing steps, such as removing missing values or scaling data.
"""

data = pd.read_csv('/content/Country-data.csv')
data

data_discription = pd.read_csv('/content/data-dictionary.csv')

# Check for missing values
print(data.isnull().sum())

# Check data types
print(data.dtypes)

# Convert columns to appropriate data types
data['country'] = data['country'].astype('category')

# Normalize numerical columns using z-score normalization
cols_to_normalize = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
data[cols_to_normalize] = (data[cols_to_normalize] - data[cols_to_normalize].mean()) / data[cols_to_normalize].std()

"""#Exploratory Data Analysis:

###Analyzing the data to gain insights and identify patterns or trends.
"""

# Create a pairplot using Seaborn
sns.pairplot(data, hue='country')
plt.show()

# Create a heatmap using Seaborn
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Create a bar plot using Matplotlib
top10_gdp = data.nlargest(20, 'gdpp')
plt.bar(top10_gdp['country'], top10_gdp['gdpp'])
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('GDP per capita')
plt.title('Top 20 Countries by GDP per Capita')
plt.show()

# Create a histogram
plt.hist(data['child_mort'], bins=20)
plt.xlabel('Child Mortality Rate')
plt.ylabel('Frequency')
plt.title('Distribution of Child Mortality Rates')
plt.show()

# Create a stacked bar chart
expenditure_cols = ['health', 'inflation', 'income']
expenditure_by_region = data.groupby('country')[expenditure_cols].mean()
expenditure_by_region.plot(kind='bar', stacked=True, figsize=(25,6))
plt.xlabel('Region')
plt.ylabel('Expenditure (% of GDP)')
plt.title('Distribution of Expenditure by Region')
plt.show()

X = data.drop(['country'], axis = 1)
y = data['country']

columns = ["exports", "imports", "gdpp"]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(columns):

    top_5 = data.nlargest(5, col)
    axs[i].pie(top_5[col], labels=top_5['country'], autopct='%1.1f%%', startangle=90)
    axs[i].set_title(col)


fig.suptitle("Top 5 Countries by Exports, Imports, and GDP per Capita")
plt.show()

columns = ['health', 'inflation', 'child_mort']

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(columns):

    top_5 = data.nlargest(5, col)
    axs[i].pie(top_5[col], labels=top_5['country'], autopct='%1.1f%%', startangle=90)
    axs[i].set_title(col)


fig.suptitle("Top 5 Countries by Health, Inflamation, and Child Mortality")
plt.show()

columns = ['life_expec', 'total_fer']

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

for i, col in enumerate(columns):

    top_5 = data.nlargest(5, col)
    axs[i].pie(top_5[col], labels=top_5['country'], autopct='%1.1f%%', startangle=90)
    axs[i].set_title(col)


fig.suptitle("Top 5 Countries by Life Expectancy and Total_fer ")
plt.show()

# Create a line plot of the GDP per capita over time
plt.plot(data['country'], data['gdpp'], color='purple', linewidth=2, linestyle='--', marker='o', markersize=8, label='GDP per capita')

# Set the title and axis labels
plt.title("GDP per Capita Over Time", fontsize=18, fontweight='bold')
plt.xlabel("country", fontsize=14)
plt.ylabel("GDP ($ per capita)", fontsize=14)

# Add a grid
plt.grid(True, linestyle=':', color='gray', alpha=0.5)

# Add legend
plt.legend(loc='upper left', fontsize=12)

# Add annotations
plt.annotate('Financial Crisis', xy=(2009, 48000), xytext=(2000, 40000),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)

# Customize the figure size and background color
fig = plt.gcf()
fig.set_size_inches(25, 6)
fig.set_facecolor('#F5F5F5')

# Show the plot
plt.show()

# Create a line plot of the GDP per capita over time
plt.plot(data['country'], data['income'], color='purple', linewidth=2, linestyle='--', marker='o', markersize=8, label='GDP per capita')

# Set the title and axis labels
plt.title("Income of ecah country", fontsize=18, fontweight='bold')
plt.xlabel("country", fontsize=14)
plt.ylabel("Income", fontsize=14)

# Add a grid
plt.grid(True, linestyle=':', color='gray', alpha=0.5)

# Add legend
plt.legend(loc='upper left', fontsize=12)

# Add annotations
plt.annotate('Financial Crisis', xy=(2009, 48000), xytext=(2000, 40000),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)

# Customize the figure size and background color
fig = plt.gcf()
fig.set_size_inches(25, 6)
fig.set_facecolor('#F5F5F5')

# Show the plot
plt.show()

"""#Detecting Outliers"""

# Identify outliers using IQR
q1 = data['gdpp'].quantile(0.25)
q3 = data['gdpp'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
outliers = data[(data['gdpp'] > upper_bound) | (data['gdpp'] < lower_bound)]

# Plot the data with outliers colored differently
plt.figure(figsize=(6, 4))
plt.scatter(x=data.index, y=data['gdpp'], s=50, c='blue', alpha=0.5)
plt.scatter(x=outliers.index, y=outliers['gdpp'], s=100, c='red')

# Add dashed lines to indicate the upper and lower bounds of the data
plt.axhline(y=upper_bound, color='gray', linestyle='--')
plt.axhline(y=lower_bound, color='gray', linestyle='--')

# Add axis labels and title
plt.xlabel('Index')
plt.ylabel('gdpp')
plt.title('Outliers in gdpp')

# Show the plot
plt.show()

# Identify outliers using IQR
q1 = data['income'].quantile(0.25)
q3 = data['income'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
outliers = data[(data['income'] > upper_bound) | (data['income'] < lower_bound)]

# Plot the data with outliers colored differently
plt.figure(figsize=(6, 4))
plt.scatter(x=data.index, y=data['income'], s=50, c='blue', alpha=0.5)
plt.scatter(x=outliers.index, y=outliers['income'], s=100, c='red')

# Add dashed lines to indicate the upper and lower bounds of the data
plt.axhline(y=upper_bound, color='gray', linestyle='--')
plt.axhline(y=lower_bound, color='gray', linestyle='--')

# Add axis labels and title
plt.xlabel('Index')
plt.ylabel('income')
plt.title('Outliers in income')

# Show the plot
plt.show()

# Identify outliers using IQR
q1 = data['exports'].quantile(0.25)
q3 = data['exports'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
outliers = data[(data['exports'] > upper_bound) | (data['exports'] < lower_bound)]

# Plot the data with outliers colored differently
plt.figure(figsize=(6, 4))
plt.scatter(x=data.index, y=data['exports'], s=50, c='blue', alpha=0.5)
plt.scatter(x=outliers.index, y=outliers['exports'], s=100, c='red')

# Add dashed lines to indicate the upper and lower bounds of the data
plt.axhline(y=upper_bound, color='gray', linestyle='--')
plt.axhline(y=lower_bound, color='gray', linestyle='--')

# Add axis labels and title
plt.xlabel('Index')
plt.ylabel('exports')
plt.title('Outliers in exports')

# Show the plot
plt.show()

"""#Model Training:

###Building and training machine learning models on the data.
"""

X = data.drop(['country'], axis = 1)
y = data['country']

# Create a PCA object with all components
pca = PCA()

# Fit the PCA model to the data
pca.fit(X)
X_pca = pca.transform(X)
# Plot the scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(pca.n_components_ , 0, -1), pca.explained_variance_ratio_, 'bo-', linewidth=2)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()

# Create a PCA object with 2 components
pca = PCA(n_components=2)

# Fit and transform the data to 2 components
X_pca = pca.fit_transform(X)

# Apply K-means clustering
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter = 500, init = 'random').fit(X_pca)
    inertias.append(kmeans.inertia_)

# Plot the inertia vs. number of clusters graph
plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Inertia vs. Number of Clusters')
plt.show()

#Finding elbow point

knee = KneeLocator(range(1, 11), inertias, curve = 'convex', direction = 'decreasing')
knee.elbow



dend = shc.dendrogram(shc.linkage(X, method='ward'))

# display the dendrogram
plt.title("Dendrogram")
plt.xlabel("Countries")
plt.ylabel("Euclidean distances")
plt.show()

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_pca)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Plot the elbow curve and silhouette score vs. number of clusters graph
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax2.plot(range(2, 11), silhouette_scores, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score')

plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

pca = PCA(n_components = 2)
pca.fit(X)
X_pca = pca.transform(X)

k_pca = KMeans(n_clusters=4, max_iter = 500,  init = 'random', random_state = 42)

k_pca.fit(X_pca)

data['labels'] = k_pca.labels_

data.head()

"""#Model Evaluation:

###Analysing the trained model.
"""

sns.set()

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

plt.subplot(1,2,1)
sns.boxplot(x = 'labels', y = 'child_mort', data  = data, color = '#BA55D3');
plt.title('child_mort vs Class')

plt.subplot(1,2,2)
sns.boxplot(x = 'labels', y = 'gdpp', data  = data, color = '#BA55D3');
plt.title('income vs Class')

plt.tight_layout()
plt.show()

data.insert(0,column = 'Country', value = data['country'])

data

data['Country'] = data['Country'].replace('United States', 'United States of America')

data['labels'].loc[data['labels'] == 0] = 'Self-Sufficient'
data['labels'].loc[data['labels'] == 1] = 'Fully Dependent'
data['labels'].loc[data['labels'] == 2] = 'Moderately Independent'
data['labels'].loc[data['labels'] == 3] = 'Partially Dependent'

"""#Results and Conclusion:

###Summarizing the findings and conclusions drawn from the analysis and modeling.
"""



# Load the world map shapefile using geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge the world map dataframe with the country data dataframe
merged = world.merge(data, left_on='name', right_on='Country')



colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
cmap = ListedColormap(colors)

ax = merged.plot(column='labels', cmap=cmap, figsize=(20,10), edgecolor='gray', legend=True)
# Add a title and adjust the plot settings
ax.set_title('Countries Classified by Our Model', fontdict={'fontsize': 20})
ax.set_axis_off()
plt.axis('equal')
plt.show()


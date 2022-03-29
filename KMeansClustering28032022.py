{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Owner:      Shilton Jonatan, Data & Analytics, Digitas Singapore, shilton.salindeho@digitas.com\n",
    "\n",
    "Solution:   K-Means Clustering\n",
    "\n",
    "Date of publication:  28 March 2022"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## K-Means Clustering\n",
    "\n",
    "Clustering algorithms seek to learn, from the properties of the data, an optimal division or discrete labeling of groups of points.\n",
    "\n",
    "Many clustering algorithms are available in Scikit-Learn and elsewhere, but perhaps the simplest to understand is an algorithm known as k-means clustering, which is implemented in sklearn.cluster.KMeans.\n",
    "\n",
    "Clustering algorithms will be useful to do Segmentation Analysis e.g. finding out specific customer segments/profiles based on data of client's customer with their purchase behavior included."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![title](p45s1.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First, we import the Python dependencies:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import scipy.stats as stats\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.preprocessing import MinMaxScaler\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Then, we load the sample data - in this sample, we'll be using a data of car brands and makes along with the specs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load mtcars sample data set\n",
    "mtcars = pd.read_csv(\"DatasetMtcars25032022.csv\") #reads text data into data frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>mpg</th>\n",
       "      <th>cyl</th>\n",
       "      <th>disp</th>\n",
       "      <th>hp</th>\n",
       "      <th>drat</th>\n",
       "      <th>wt</th>\n",
       "      <th>qsec</th>\n",
       "      <th>vs</th>\n",
       "      <th>am</th>\n",
       "      <th>gear</th>\n",
       "      <th>carb</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Mazda RX4</td>\n",
       "      <td>21.0</td>\n",
       "      <td>6</td>\n",
       "      <td>160.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.90</td>\n",
       "      <td>2.620</td>\n",
       "      <td>16.46</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Mazda RX4 Wag</td>\n",
       "      <td>21.0</td>\n",
       "      <td>6</td>\n",
       "      <td>160.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.90</td>\n",
       "      <td>2.875</td>\n",
       "      <td>17.02</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Datsun 710</td>\n",
       "      <td>22.8</td>\n",
       "      <td>4</td>\n",
       "      <td>108.0</td>\n",
       "      <td>93</td>\n",
       "      <td>3.85</td>\n",
       "      <td>2.320</td>\n",
       "      <td>18.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Hornet 4 Drive</td>\n",
       "      <td>21.4</td>\n",
       "      <td>6</td>\n",
       "      <td>258.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.08</td>\n",
       "      <td>3.215</td>\n",
       "      <td>19.44</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Hornet Sportabout</td>\n",
       "      <td>18.7</td>\n",
       "      <td>8</td>\n",
       "      <td>360.0</td>\n",
       "      <td>175</td>\n",
       "      <td>3.15</td>\n",
       "      <td>3.440</td>\n",
       "      <td>17.02</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Unnamed: 0   mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  \\\n",
       "0          Mazda RX4  21.0    6  160.0  110  3.90  2.620  16.46   0   1     4   \n",
       "1      Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4   \n",
       "2         Datsun 710  22.8    4  108.0   93  3.85  2.320  18.61   1   1     4   \n",
       "3     Hornet 4 Drive  21.4    6  258.0  110  3.08  3.215  19.44   1   0     3   \n",
       "4  Hornet Sportabout  18.7    8  360.0  175  3.15  3.440  17.02   0   0     3   \n",
       "\n",
       "   carb  \n",
       "0     4  \n",
       "1     4  \n",
       "2     1  \n",
       "3     1  \n",
       "4     2  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#See first few lines of data\n",
    "mtcars.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(32, 12)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mtcars.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to conduct a K-Means, we'll have to do some preprocessing to the data first. The two most common steps of preprocessing before K-Means are scaling and one-hot encoding.\n",
    "\n",
    "- One-hot encoding converts qualitative variables (e.g. gender, race, nationality) into quantitative variables by splitting one variable into multiple \"dummy\" variables (values are either 0 or 1).\n",
    "\n",
    "- Scaling converts variables of different ranges/scales into a uniform scale between 0 to 1."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Scaling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First we create a subset of the data that contains only the variable we are interested in scaling. \n",
    "\n",
    "As seen above, all variables are quantitative except the first column thus we will just exclude the first column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.451064</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.221751</td>\n",
       "      <td>0.204947</td>\n",
       "      <td>0.525346</td>\n",
       "      <td>0.283048</td>\n",
       "      <td>0.233333</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.428571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.451064</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.221751</td>\n",
       "      <td>0.204947</td>\n",
       "      <td>0.525346</td>\n",
       "      <td>0.348249</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.428571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.527660</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.092043</td>\n",
       "      <td>0.144876</td>\n",
       "      <td>0.502304</td>\n",
       "      <td>0.206341</td>\n",
       "      <td>0.489286</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.468085</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.466201</td>\n",
       "      <td>0.204947</td>\n",
       "      <td>0.147465</td>\n",
       "      <td>0.435183</td>\n",
       "      <td>0.588095</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.353191</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.720629</td>\n",
       "      <td>0.434629</td>\n",
       "      <td>0.179724</td>\n",
       "      <td>0.492713</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.142857</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         0    1         2         3         4         5         6    7    8   \\\n",
       "0  0.451064  0.5  0.221751  0.204947  0.525346  0.283048  0.233333  0.0  1.0   \n",
       "1  0.451064  0.5  0.221751  0.204947  0.525346  0.348249  0.300000  0.0  1.0   \n",
       "2  0.527660  0.0  0.092043  0.144876  0.502304  0.206341  0.489286  1.0  1.0   \n",
       "3  0.468085  0.5  0.466201  0.204947  0.147465  0.435183  0.588095  1.0  0.0   \n",
       "4  0.353191  1.0  0.720629  0.434629  0.179724  0.492713  0.300000  0.0  0.0   \n",
       "\n",
       "    9         10  \n",
       "0  0.5  0.428571  \n",
       "1  0.5  0.428571  \n",
       "2  0.5  0.000000  \n",
       "3  0.0  0.000000  \n",
       "4  0.0  0.142857  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mms = MinMaxScaler()                                    #Calls the Scaler Object\n",
    "mms.fit(mtcars.iloc[:,1:])                              #Selects the data to be scaled\n",
    "mtcars_scaled=pd.DataFrame(mms.transform(mtcars.iloc[:,1:])) #Transforms the data\n",
    "mtcars_scaled.head()                                    #Check final scaled result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Elbow Method"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "k-Means is an unsupervised machine learning algorithm, but there are ways to check which k number of segments best fits our data. One of them is the ‘Elbow-Method’"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "Sum_of_squared_distances = []\n",
    "K = range(1,15)\n",
    "for k in K:\n",
    "    km = KMeans(n_clusters=k)\n",
    "    km = km.fit(mtcars_scaled)\n",
    "    Sum_of_squared_distances.append(km.inertia_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(K, Sum_of_squared_distances, 'bx-')\n",
    "plt.xlabel('k')\n",
    "plt.ylabel('Sum_of_squared_distances')\n",
    "plt.title('Elbow Method For Optimal k')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The chart above shows us that 4 is the optimal number of segments."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### K-Means Fitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "km=KMeans(n_clusters=4)\n",
    "km.fit(mtcars_scaled)\n",
    "cluster_result = pd.DataFrame(km.predict(mtcars_scaled))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can add the resulting cluster back to the original dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "mtcars_final=mtcars\n",
    "mtcars_final['cluster_result']=cluster_result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>mpg</th>\n",
       "      <th>cyl</th>\n",
       "      <th>disp</th>\n",
       "      <th>hp</th>\n",
       "      <th>drat</th>\n",
       "      <th>wt</th>\n",
       "      <th>qsec</th>\n",
       "      <th>vs</th>\n",
       "      <th>am</th>\n",
       "      <th>gear</th>\n",
       "      <th>carb</th>\n",
       "      <th>cluster_result</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Mazda RX4</td>\n",
       "      <td>21.0</td>\n",
       "      <td>6</td>\n",
       "      <td>160.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.90</td>\n",
       "      <td>2.620</td>\n",
       "      <td>16.46</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Mazda RX4 Wag</td>\n",
       "      <td>21.0</td>\n",
       "      <td>6</td>\n",
       "      <td>160.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.90</td>\n",
       "      <td>2.875</td>\n",
       "      <td>17.02</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Datsun 710</td>\n",
       "      <td>22.8</td>\n",
       "      <td>4</td>\n",
       "      <td>108.0</td>\n",
       "      <td>93</td>\n",
       "      <td>3.85</td>\n",
       "      <td>2.320</td>\n",
       "      <td>18.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Hornet 4 Drive</td>\n",
       "      <td>21.4</td>\n",
       "      <td>6</td>\n",
       "      <td>258.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.08</td>\n",
       "      <td>3.215</td>\n",
       "      <td>19.44</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Hornet Sportabout</td>\n",
       "      <td>18.7</td>\n",
       "      <td>8</td>\n",
       "      <td>360.0</td>\n",
       "      <td>175</td>\n",
       "      <td>3.15</td>\n",
       "      <td>3.440</td>\n",
       "      <td>17.02</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Unnamed: 0   mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  \\\n",
       "0          Mazda RX4  21.0    6  160.0  110  3.90  2.620  16.46   0   1     4   \n",
       "1      Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4   \n",
       "2         Datsun 710  22.8    4  108.0   93  3.85  2.320  18.61   1   1     4   \n",
       "3     Hornet 4 Drive  21.4    6  258.0  110  3.08  3.215  19.44   1   0     3   \n",
       "4  Hornet Sportabout  18.7    8  360.0  175  3.15  3.440  17.02   0   0     3   \n",
       "\n",
       "   carb  cluster_result  \n",
       "0     4               3  \n",
       "1     4               3  \n",
       "2     1               2  \n",
       "3     1               0  \n",
       "4     2               1  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mtcars_final.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis of Cluster Results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can analyze the resulting clusters, e.g. what are the mean values of mpg for each cluster:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>mpg</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>cluster_result</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20.742857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15.050000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28.371429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>19.750000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      mpg\n",
       "cluster_result           \n",
       "0               20.742857\n",
       "1               15.050000\n",
       "2               28.371429\n",
       "3               19.750000"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.pivot_table(mtcars_final, values='mpg', index='cluster_result', aggfunc=np.mean)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

from ucimlrepo import fetch_ucirepo


# fetch dataset
taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572)

# data (as pandas dataframes)
X = taiwanese_bankruptcy_prediction.data.features
y = taiwanese_bankruptcy_prediction.data.targets

# metadata
print(taiwanese_bankruptcy_prediction.metadata)

# variable information
print(taiwanese_bankruptcy_prediction.variables)

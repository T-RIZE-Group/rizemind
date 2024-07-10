import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import json

# Loading  dataset

# Create a DataFrame
df = pd.read_csv('/home/iman/projects/kara/Projects/T-Rize/archive/American_Housing_Data_20231209.csv')


# Define the number of bins
num_bins = 300

# Define bin edges
bin_edges = np.linspace(df['Price'].min(), df['Price'].max(), num_bins + 1)

# Assign samples to bins manually

# Define bin labels
bin_labels = [i for i in range(num_bins)]

# Create a mapping dictionary with interval boundaries
mapping_dict = {}
for i, label in enumerate(bin_labels):
    if i < len(bin_edges) - 1:
        interval = {'min': bin_edges[i], 'max': bin_edges[i + 1]}
    else:
        interval = {'min': bin_edges[i], 'max': np.inf}  # For the last bin, max is infinity
    mapping_dict[label] = interval

labels = []
for i in range(len(df)):
    
    for j in bin_labels:
        if df.iloc[i]['Price'] >= mapping_dict[j]['min']:
            if df.iloc[i]['Price'] <= mapping_dict[j]['max']:
                labels.append(j)
                break

df['price_label'] = np.array(labels)

classes_to_keep = []
unique_class_labels = df['price_label'].unique().tolist()
for i in unique_class_labels:
    class_count = len(df[df['price_label'] == i])
    
    if class_count >= 500:
        classes_to_keep.append(i)

dataframes = []
for i in classes_to_keep:
    data = df[df['price_label'] == i]
    class_count = len(df[df['price_label'] == i])
    print(f'number of samples in class {i}: {class_count}')
    dataframes.append(data)

final_dataset = pd.concat(dataframes, ignore_index=True)
# Save mapping to JSON file
with open('/home/iman/projects/kara/Projects/T-Rize/archive/price_bins_mapping.json', 'w') as json_file:
    json.dump(mapping_dict, json_file, indent=4)

final_dataset.to_csv('/home/iman/projects/kara/Projects/T-Rize/archive/American_housing_data.csv')
print("Mapping saved to 'price_bins_mapping.json'")

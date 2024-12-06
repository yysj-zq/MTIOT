# MTIOT
Identifying HPV subtypes from multiple infection data in an individual
## Examples
Python 3.7.13  
pandas 1.3.5  
biopython 1.78  
sktime 0.12.1  
numpy 1.21.5  
scikit-learn 1.0.2  
joblib 1.2.0  
### Train with default data
```python
from MTIOT import Mtiot

path_predict_data = 'path of data you want to predict'
path_result = 'path you want to store result'

model = Mtiot(n_jobs = -1)
model.result(path_data, path_result)
```
### Train with your own data
```python
from MTIOT import Mtiot

path_train_data = ''
path_train_label = ''
path_predict_data = 'path of data you want to predict'
path_result = 'path you want to store result'

model = Mtiot(path_train_label, path_train_data, n_jobs = -1)
model.result(path_data, path_result)
```
The train/predict data path should point to a folder containing AB1 files. Seem like [./data/NDATA](./data/NDATA)  
The label path should point to a CSV file, where the first column contains each subtype separated by commas, and the second column contains the corresponding ab1 file name. Seem like [./data/label.csv](./data/label.csv)

# MTIOT
Identifying HPV subtypes from multiple infection data in an individual
## Abstract
Persistent infection with high-risk human papillomavirus (hrHPV) is the principal etiological factor in the development of cervical epithelial neoplasia and cervical cancer. Consequently, the execution of HPV typing tests and the confirmation of ongoing infections bear substantial clinical significance.  The World Health Organization endorses HPV-DNA testing as the primary method for cervical cancer screening, yet the current methodologies exhibit several limitations, including low sensitivity, suboptimal accuracy, and an inability to perform comprehensive typing. In response to these challenges, we have devised a novel HPV detection method, designated MTIOT (Multiple subTypes In One Time), predicated on HPV-DNA. This method enables swift and precise HPV typing tests on samples with multiple infections. When applied to simulated samples with multiple infections, MTIOT demonstrates exceptional performance, achieving 98% specificity and sensitivity. MTIOT offers a novel approach for the precise detection of high-risk HPV infections and is poised to play a pivotal role in cervical cancer screening. This study furnishes robust support for future clinical applications and holds promise for enhancing the efficiency and precision of cervical cancer screening. Future research will focus on further clinical validation and methodological optimization to ensure the practical efficacy of this approach.
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
The train/predict data path should point to a folder containing AB1 files. Seem like [./data/NDATA](./Data/NDATA)  
The label path should point to a CSV file, where the first column contains each subtype separated by commas, and the second column contains the corresponding ab1 file name. Seem like [./data/label.csv](./data/label.csv)

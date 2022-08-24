from sklearn import datasets
from tqdm import tqdm

def AcquireImages():
    
    path = "../Inventory/DataSets"
    
    files = datasets.load_files(path)
    
    print(files.target)
    
AcquireImages()
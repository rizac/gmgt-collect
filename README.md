(G)round (M)otion (G)round (T)ruth - collect is a collection of scripts for the 
generation of harmonized ground truth datasets of processed ground motion data and 
metadata which can then feed Deep- and Machine-learning applications.

This package is n ot intended for public use: the source data is not available
publicly for rights and size issues. Please contact the author for questions

# Installation

- Clone this package

- Install a Python virtual environment:
  ```
  python3 -m venv .env 
  ```

- Activate the environment **to be done every time you run Python code**
  ```
  source .env/bin/activate
  ```

- Install required packages (add any package of your choice if needed):
  ```
  pip install --upgrade pip setuptools && pip install pandas h5py pyyaml tables tqdm
  ```
   <!-- pip install --upgrade pip setuptools && pip install obspy pandas pyyaml tqdm -->

# Implementation

1. Copy one of the already implemented scripts `create_<dataset_name>.py` and modify the editable 
   functions (see instructions therein)
   
2. Create a yaml file that will be used as argument, where you setup the source metadata
   and the source time histories, e.g. in a file `<dataset_name>.yml`:
   
   ```yaml
    source_metadata: "/home/dasegen/source-datasets/ngawest2/metadata.csv"
    source_data: "/home/dasegen/source-datasets/ngawest2/Waveforms"
    destination: "/home/dasegen/datasets/ngawest2"
    ```

3. execute `reate_<dataset_name>.py <dataset_name>.yml`
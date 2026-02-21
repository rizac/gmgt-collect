(G)round (M)otion (G)round (T)ruth - collect is a set of scripts for assembling
harmonized ground truth datasets of processed ground motion data and 
metadata which can then feed Deep- and Machine-learning applications.

This package is not intended for public use: the source data is not available
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

- Install required packages (pytest is optional, i.e. in square brackets. Remove
  if you do not plan to test scripts):
  ```
  pip install --upgrade pip setuptools && pip install "pandas<3" h5py pyyaml tables tqdm [pytest]
  ```

# Implementation

1. Copy one of the already implemented scripts `create_<dataset_name>.py` and modify the editable 
   functions (see instructions therein)
   
2. Create a YAML file that will be used as argument, where you setup the source metadata
   and the source time histories, e.g. in a file `<dataset_name>.yml`:
   
   ```yaml
    source_metadata: "/home/dasegen/source-datasets/ngawest2/metadata.csv"
    source_data: "/home/dasegen/source-datasets/ngawest2/Waveforms"
    destination: "/home/dasegen/datasets/ngawest2"
    ```

3. Execute `create_<dataset_name>.py <dataset_name>.yml`
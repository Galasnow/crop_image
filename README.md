# crop_image
Crop images including interest target. It's suitable for generate testset from remote sensing images.
## Usage
### 1. Clone the repository.
    git clone https://github.com/Galasnow/crop_image.git

### 2. Install requirements. Conda env is recommended.
    cd crop_image
    conda create -n <name> python=3.12
    conda activate <name>
    conda install conda-forge::gdal
    pip install -r requirements.txt

### 3. Prepare images.

### 4. Edit `config/config_center.yml` or `config/config_grid.yml` according to your setting.

### 5. Run the script.
    python src/crop_image.py config/config_center.yml
    python src/crop_image.py config/config_grid.yml

## Note
This is an experimental project and it can't guarantee any functionality now.

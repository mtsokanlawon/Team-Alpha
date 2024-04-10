import zipfile

# Path to the zip file
zip_file_path = 'final GS predictor.zip'

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('final GS predictor')

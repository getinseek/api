import gzip
import shutil
import requests
import os

import tqdm

def download_model():
    if check_installed():
        return
    url = "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz"
    print("Downloading Moondream Image Model: \r")
    response = requests.get(url, stream=True)


    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    gz_file = "moondream-0_5b-int8.mf.gz"
    extract_dir = "moondream_model"
    
    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(gz_file, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")
    
    # # write the downloaded gz file
    # with open(gz_file, "wb") as f:
    #     f.write(response.content)

    # make extraction dir if not exists
    os.makedirs(extract_dir, exist_ok=True)
    
    # extract the gz file
    with gzip.open(gz_file, "rb") as f_in:
        with open(os.path.join(extract_dir, "moondream-0_5b-int8.mf"), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"Model extracted to {extract_dir}")

def check_installed():
    

    file_path = "./moondream_model/moondream-0_5b-int8.mf"  # Replace with the actual file path

    return os.path.exists(file_path)
        
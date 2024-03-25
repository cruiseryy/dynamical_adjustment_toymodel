from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import subprocess

# Define the download task as a function
def download_file(url, dest_path):
    subprocess.run(['wget', '-P', dest_path, url])

# Your base URL and destination path
url_pre = "https://downloads.psl.noaa.gov//Datasets/20thC_ReanV3/Dailies/miscMO/prmsl."
dest_path = '~/nas/home/20CR_V3/daily'
suf = ".nc"
years = 1980

# Use ThreadPoolExecutor to download up to 5 files at a time
with ThreadPoolExecutor(max_workers=5) as executor:
    for y in range(42):
        t1 = time()
        
        # Prepare a list to keep track of the futures
        futures = []
        tmpyy = years + y
        url = url_pre + str(tmpyy) + suf
        futures.append(executor.submit(download_file, url, dest_path))

        
        # Wait for the downloads to completed
        for future in as_completed(futures):
            try:
                future.result()  # If the download function raised an exception, it will be re-raised here
            except Exception as exc:
                print(f'Download task generated an exception: {exc}')
        
        t2 = time()
        print(f'Time taken for interval {y} is {t2 - t1}')

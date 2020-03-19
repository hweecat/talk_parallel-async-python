import numpy as np

import requests
import json

import sys
import time
from datetime import datetime, timedelta
from tqdm import trange, tqdm
from time import sleep
from retrying import retry

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
def get_airtemp_data_from_date(date):

    print('{}: running {}'.format(
        threading.current_thread().name,
        date)
    )
    url = \
        "https://api.data.gov.sg/v1/environment/air-temperature?date=" + \
            str(date) # for daily API request
    JSONContent = requests.get(url).json()
    content = json.dumps(JSONContent, sort_keys=True)
    sleep(1)
    print('{}: done with {}'.format(
        threading.current_thread().name,
        date)
    )
    return content

def main():

    date_range = np.array(sorted(
        [datetime.strftime(
            datetime.now() - timedelta(i)
    ,'%Y-%m-%d') for i in trange(100)])) 

    ## list comprehensions, without ThreadPoolExecutor()

    start_cpu_time = time.clock()

    data_np = [get_airtemp_data_from_date(str(date)) for date in tqdm(date_range)]

    end_cpu_time = time.clock()
    total_tpe_time = end_cpu_time - start_cpu_time
    sys.stdout.write('List comprehension completed in {} seconds.\n'.format(total_tpe_time))
    
    sleep(1)
    
    ## Using ThreadPoolExecutor() and executor.submit()

    start_cpu_time = time.clock()

    with ThreadPoolExecutor() as executor:
        future = {executor.submit(get_airtemp_data_from_date, date) : date
            for date in tqdm(date_range)}

    resultarray_np = np.array([x.result() for x in as_completed(future)])

    end_cpu_time = time.clock()
    total_tpe_time = end_cpu_time - start_cpu_time
    sys.stdout.write('ThreadPoolExecutor completed in {} seconds.\n'.format(total_tpe_time))

    for i in range(len(resultarray_np)):
        sys.stdout.write('{}\n'.format(resultarray_np[i]))

if __name__ == '__main__':
    main()
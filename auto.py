import os
import boto3
import model
import secret
import random
import requests
import datetime
import pandas as pd
from timeloop import Timeloop
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

tl = Timeloop()

engine = create_engine(
    URL(
        user='kshitizjain',
        password=secret.pwd,
        account='gna86856.us-east-1',
        warehouse='COMPUTE_WH',
        database='COVID_DATA',
        schema='PUBLIC',
        role='ACCOUNTADMIN'
    )
)

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=secret.aws_access_key,
    aws_secret_access_key=secret.aws_secret_key
)

@tl.job(interval=datetime.timedelta(seconds=1440))
def periodic():

    cur = engine.connect()
    s3.Bucket('snowflake-covid-bucket').objects.all().delete()

    date = pd.read_sql_query(secret.latest_date_query, cur)['test_date'][0]

    while True:

        date += datetime.timedelta(1)

        res = requests.get(url=secret.api_url.format(str(date))).json()['result']

        if len(res['records']) == 0:
            print('SUCCESS: DATA HAS BEEN UPDATED\n')
            print('NEXT DATA UPDATION JOB WILL EXECUTE IN 24 HOURS\n')
            model.run(cur)
            break

        data = pd.json_normalize(res, record_path=['records']).drop(['_id', 'rank'], axis=1)
        new_dtypes = {
            'test_date'          : str,
            'cough'              : bool,
            'fever'              : bool,
            'sore_throat'        : bool,
            'shortness_of_breath': bool,
            'head_ache'          : bool,
            'corona_result'      : str,
            'age_60_and_above'   : str,
            'gender'             : str,
            'test_indication'    : str
        }
        data = data.astype(new_dtypes)

        filename = '{}.csv'.format(str(date) + str(random.random()))
        data.to_csv(filename, index=False)

        try:
            s3.Bucket('snowflake-covid-bucket').upload_file(Filename=filename, Key=filename)
            print('SUCCESS: DATA INSERTION FOR DATE={}'.format(str(date)))
        except:
            print('FAILED: DATA INSERTION FOR DATE={}\n'.format(str(date)))
            print('CLOSED: CONNECTION\n')
            print('NEXT JOB WILL EXECUTE IN 24 HOURS\n')
            cur.close()
            return
        finally:
            os.remove(filename)

if __name__ == "__main__":
    tl.start(block=True)
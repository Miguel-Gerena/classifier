import pandas as pd
import os
from time import sleep
import json

def paraleliz2(data):
    json_dir = "D:/classes/CS224N/HUPD_dataset/data"
    # id_, x = arg

    application_year = str(data["filing_date"].year)
    application_number = data["application_number"]
    filepath = os.path.join(json_dir, application_year, application_number + '.json')
    with open(filepath, 'r') as f:
        patent = json.load(f)

    # Most up-to-date-decision in meta dataframe

    data["comb"] = {
    "patent_number": [application_number],
    "decision": patent["decision"], # decision,
    "title": patent["title"],
    "abstract": patent["abstract"],
    "claims": patent["claims"],
    "description": patent["full_description"],
    "background": patent["background"],
    "summary": patent["summary"],
    "cpc_label": patent["main_cpc_label"],
    'filing_date': patent['filing_date'],
    'patent_issue_date': patent['patent_issue_date'],
    'date_published': patent['date_published'],
    'examiner_id': patent['examiner_id'],
    "ipc_label": patent["main_ipcr_label"],
}

    return data

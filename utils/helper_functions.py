import json
import csv
import yaml
import dill
import os

def load_from_file(fpath, noheader=True):
    ftype = os.path.splitext(fpath)[-1][1:]
    if ftype == 'pkl':
        with open(fpath, 'rb') as rfile:
            out = dill.load(rfile)
    elif ftype == 'txt':
        with open(fpath, 'r') as rfile:
            if 'prompt' in fpath:
                out = "".join(rfile.readlines())
            else:
                out = [line.strip() for line in rfile.readlines()]
    elif ftype == 'json':
        with open(fpath, 'r') as rfile:
            out = json.load(rfile)
    elif ftype == 'csv':
        with open(fpath, 'r') as rfile:
            csvreader = csv.reader(rfile)
            if noheader:
                fileds = next(csvreader)
            out = [row for row in csvreader]
    elif ftype == 'yaml':
        with open(fpath, 'r') as rfile:
            out = yaml.load(rfile, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")
    return out

def save_to_file(data, fpth, mode=None):
    ftype = os.path.splitext(fpth)[-1][1:]
    if ftype == 'pkl':
        with open(fpth, mode if mode else 'wb') as wfile:
            dill.dump(data, wfile)
    elif ftype == 'txt':
        with open(fpth, mode if mode else 'w') as wfile:
            wfile.write(data)
    elif ftype == 'json':
        with open(fpth, mode if mode else 'w') as wfile:
            json.dump(data, wfile, sort_keys=True,  indent=4)
    elif ftype == 'csv':
        with open(fpth, mode if mode else 'w', newline='') as wfile:
            writer = csv.writer(wfile)
            writer.writerows(data)
    elif ftype == 'yaml':
        with open(fpth, 'w') as wfile:
            yaml.dump(data, wfile)
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")
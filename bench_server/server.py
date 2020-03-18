import sqlite3
import os
import json
from git import Repo

# TODO grab these from env
benchmark_db_fn = '../benchmarks.db'
repo_dir = '../nnc_bench'
repo = Repo(repo_dir)


if 'SQLITE_DB' in os.environ:
    benchmark_db_fn = os.environ['SQLITE_DB']
conn = sqlite3.connect(benchmark_db_fn)
c = conn.cursor()

jsons = []
for r in c.execute("SELECT * FROM benchmarks"):
    commit_hash = r[0]
    commit = repo.commit(commit_hash)
    date = commit.committed_date
    data = r[1].split('\n')
    json_data = []
    for datum in data:
        d = json.loads(datum)
        json_data.append(d)
        print(d)#r[1])
    jsons.append((commit_hash, date, json_data))
    #for k in str(r[1]).split('\n'):
        #d = json.loads(k)
        



from flask import Flask, render_template
app = Flask(__name__, static_folder='static/')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def serve_json_data():
    return json.dumps(jsons)

if __name__ == '__main__':
    app.run('0.0.0.0')

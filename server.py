import sqlite3
import os
import json
dir_path = os.path.dirname(os.path.realpath(__file__))

from flask import Flask, render_template
app = Flask(__name__, static_folder='static/', template_folder=dir_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def serve_json_data():
    result_path = os.path.join(dir_path, 'nnc_bench_results')
    result_files = []
    for filename in os.listdir(result_path):
        result_files.append(filename)
    result_files = sorted(result_files)
    jsons = []
    
    for filename in result_files:
        with open(os.path.join(result_path, filename), 'r') as f:
            data = f.read().strip().split('\n')
            parsed_filename = filename.split('_')
            date = int(parsed_filename[1])
            def parse_json(d):
              d = json.loads(d)
              # resnet18_bs_1_eval_cuda_te
              parsed_name = d["name"].split('_')
              d["model"] = '_'.join(parsed_name[:-5])
              d["fuser"] = parsed_name[-1]
              d["device"] = parsed_name[-2]
              d["workload"] = parsed_name[-3]
              d["batch_size"] = int(parsed_name[-4])
              return d
            try:
              bench_data = [parse_json(d) for d in data]
              jsons.append((date, bench_data))
            except:
              pass
    return json.dumps(jsons)

@app.route('/pr/<int:pr>')
def bench_pr(pr):
    pr_dir = os.path.join(dir_path, 'nnc_bench_prs')
    pr_path = os.path.join(pr_dir, '%d.result' % pr)
    if os.path.exists(pr_path):
        #with open(pr_path, 'r') as f:
        #    #bench_data = try_parse(filename, f)
        #    if bench_data:
        #        return json.dumps(bench_data)
        return 'Still benchmarking %d...' % pr 
    else:
        run = os.path.join(dir_path, 'run_benchmarks.sh')
        env = 'PR="pull/%d/head" RESULT_FILE=%s ' % (pr, pr_path)
        os.system('mkdir -p %s' % pr_dir)
        print(env + run)
        os.system(env + run)
    return 'I will benchmark PR %d' % pr

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000)

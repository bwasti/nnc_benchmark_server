# Benchmark Server

This codebase contains an FB internal benchmark server.

Collect benchmarks with `./collect_benchmarks.sh`.
This populates a database file, `benchmarks.db` that can be used
with the python server in `bench_server`.
Run the server in that folder with `python server.py`.

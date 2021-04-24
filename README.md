# 🎶 Generation of Music Based on Artists' Styles using Attention-Based and Long Short-Term Memory Neural Nets

## Development

### Python

The Python version should be 3.6+. Optionally set up
a [venv](https://docs.python.org/3/library/venv.html).

Run the following command to install all required packages:

```shell
pip install --requirement requirements.txt 
```

Make sure to update `requirements.txt` after adding a new package:

```shell
pip freeze > requirements.txt # only if pip is in a project-only environment
```

### Data

Data for running and executing the models is in the `./data` directory.

Execute the following command to download all data:

```shell
python data/download.py # optionally add --datasets [lmd, maestro]
```

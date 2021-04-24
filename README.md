# 🎶 Generation of Music Based on Artists' Styles using Attention-Based and Long Short-Term Memory Neural Nets

## Usage & Development

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

### Getting the Data

Data for running and executing the models is in the `./data` directory.

Execute the following command to download all data:

```shell
python data/download.py # optionally add --datasets [lmd, maestro, maestro-notes-pl]
```

Only preprocessed data from the MAESTRO dataset `maestro-notes-pl` is necessary for training.

### Training the Model

Execute the following command to start training:

```shell
python src/rnn/train.py --seq_len 200 --batch_size 32 --epoch_count=50 --artist "Wolfgang Amadeus Mozart"
```

All arguments have defaults and are optional.

### Generating Melodies

Execute the following to generate melody:

```shell
python3 src/rnn/generate.py --training_id ***** --seq_len 150 --name "sample"
```

Replace `*****` with the training ID of the trained weights and configuration you wish to use. This
is the randomly-generated first five characters of the weight and config filenames under
`weights/rnn`.

All arguments have defaults and are optional.
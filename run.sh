#! /usr/bin/env bash

python ./convert_model.py model_src/ gte-small-q4.gtemodel && \
make clean && make generic &&\
./test_gte --model-path gte-small-q4.gtemodel "The weather is lovely today." "It's so sunny outside!" "He drove to the stadium."

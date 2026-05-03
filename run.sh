#! /usr/bin/env bash

export NIX_ENFORCE_NO_NATIVE=0

if [ ! -d "simde" ]; then
    git clone https://github.com/simd-everywhere/simde.git vendor_simde && cp vendor_simde/simde simde
fi

python ./convert_model.py model_src/ gte-small-q4.gtemodel && \
make clean && make generic &&\
./test_gte --model-path gte-small-q4.gtemodel "The weather is lovely today." "It's so sunny outside!" "He drove to the stadium."

#!/bin/bash

python lm_ltr/create_mappings.py
python lm_ltr/create_trectext.py
python lm_ltr/create_query_params_for_dataset.py
IndriBuildIndex /home/dany/lm_ltr/indri/index_params.xml
IndriRunQuery query_params.xml -count=10 -index=/home/dany/lm_ltr/indri/out -trecFormat=true > query_result

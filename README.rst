Weak Supervision for Document Retrieval
======

This project has transitioned from: using a pretrained neural language model for document retrieval, to investigating different representations of documents and queries, to reproducing the results of `[1]`_, to investigating methods of learning from weak supervision with fewer training examples. Despite the name of this repo, the transfer learning experiments are no longer the focus of this codebase.

.. _[1]: https://arxiv.org/abs/1704.08803

Usage
-----
If the db is not available, create `./rows` or `./preprocessed` by running `main.py` wherever the db is available and loaded. Then transfer via `ssh` etc.

- Create `indri/in` by running `create_trectext.py`
- Create `query_params.xml` by running `create_query_params_for_dataset.py`
- Run `IndriBuildIndex index_params.xml` to build the Indri index of the documents in `in`
- Run `IndriRunQuery query_params.xml -count=10 -index=PATH_TO_PROJECT/lm_ltr/indri/out -trecFormat=true > query_result` to query the Indri index and save results to `query_result`
- Run `read_query_results.py` to process the results and save them to `./indri_results.pkl`
- Run `main.py` with the desired arguments. To load a previously trained model, use `--load_model`. Training and testing results are stored using `rabbit_ml` (https://github.com/dmh43/rabbit-ml).

Requirements
^^^^^^^^^^^^
- See `requirements.txt`

Licence
-------
GNU GPL v3

Authors
-------

`lm_ltr` was written by `Dany Haddad <danyhaddad43@gmail.com>`_.

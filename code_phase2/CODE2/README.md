As the same in phase 1, our key method is to add an initial matrix before running PTHP, which means we only check where the value is 1 in initial matrix. Using initial matrices can improve the accuracy of the results and speed up the PTHP algorithm. In our experiments, the initial matrices are obtained from DAGMA, PC and causal effect.  For higher accuracy, we use the prior matrix as a filter covering our result matrices in the end. Followings are the brief descriptions of the methods including PTHP.

In our folder, `CD_methods.py` is a collection of the methods for causal discovery. `Utils.py` is the collection of some tool functions. `to_get_initial_matrices.py` is the python file to generate the initial matrices. For convenience, we have generated the required initial matrices in the folder `_init_matrices`. And the `main.py` is the main file to run by command  `python ./main.py`. Folders `pthp` and `trustworthyAI` are the needed frameworks. The results will be saved in the folder `./FINAL_RESULTS` as defined in the file `datapath.json`. And `environment.yml` is the environment file.

```
git clone https://github.com/huawei-noah/trustworthyAI.git
```

```python
conda env create -f environment.yml
conda activate hsong_CD
python ./main.py
```

The best iteration are as following:

| **dataset**        | 4    | 5    | 6    |
| ------------------ | ---- | ---- | ---- |
| **best iteration** | 2    | 56   | 6    |
In this folder, `CD_methods.py` is a collection of the methods for causal discovery, and we only took an approach called **PTHP**. Moreover, before PTHP, we add an matrix used for initialization. Through an initial matrix we just check the position whose value is 1 while doing PTHP. These initial matrixes were obtained from some other methods like PC, DAGMA and so on. We took the DAGMA result as the initial matrix for dataset3, and causal effect, a method based on statistics, for the rest of the datasets.

`main.py` is the program to run by `python ./main.py` .

`Utils.py` is the collection of some tool functions.

`environment.yml` is the environment file. The environment can be set by `conda env create -f environment.yml`

folders `pthp` and `trustworthyAI` are the needed frameworks.

Additionally, our results are affected by the number of iterations, We explored the best iterations on our method as following:

| dataset | best iteration |
| ------- | -------------- |
| 1       | 62             |
| 2       | 72             |
| 3       | 48             |
| 4       | 14             |


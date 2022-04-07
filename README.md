
## Overview

This is the replication package for the paper Balke and Lamadon (2022) "Productivity Shocks, Long-Term Contracts and Earnings Dynamics". 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tlamadon/balke-lamadon-2022-aer/binder) 
[<img src="https://img.shields.io/badge/docker-balke--lamadon--2022--aer-blue?logo=appveyor">](https://hub.docker.com/r/tlamadon/balke-lamadon-2022-aer)

There are three main parts to the replication:

  1. Part I: We compute a set of moments and data statistics using confidential administrative data from Sweden. This part uses the R language and relies on minimal dependencies. Computation takes around 3 core-hours.
  2. Part II: Using the set of moments and their standard errors, we run an optimizer to find the parameter values that minimize the distance between the data moments and the moments simulated from the model. This part is written in python. Given the provided starting value, this takes about 600 core-hours.
  3. Part III: Using the set of parameter values, we compute all 7 tables and 8 figures from the empirical sections of the paper and the online appendix. This part is written in python and we provide a conda environment file as well as a fully isolated docker container to replicate all output. The master file can be run in parallel and takes approximately 300 core-hours.

Part I requires access to confidential data and can only be ran on the Institute for Evaluation of Labour Market and Education Policy (IFAU) servers. However, all inputs are provided inside the current public package to reproduce the computationally intensive part of the paper contained in Part II and III.

## Data Availability and Provenance Statements

Due to strict regulations regarding access to and processing of personal data, the Swedish microdata cannot be uploaded to journal servers. However, the IFAU ensures data availability in accordance with requirements by allowing access to researchers who wish to replicate the analyses. The authors will also assist with any reasonable replication attempts for two years following publication.

Researchers wishing to perform replication analyses can apply for access to the data. The researcher will be granted remote (or site) access to the data to the extent necessary to perform replication, provided he/she signs a reservation of secrecy. The reservation states the terms of access, most importantly that the data can only be used for the stated purposes (replication), only be accessed from within the EU/EEA, and not transferred to any third party. The authors will be available for consultation.

Apart from allowing access for replication purposes, any researcher can apply to Statistics Sweden to obtain the same data for research projects, subject to their conditions.

This project uses a quarterly panel data on individual firms and workers initially prepared as part of the project `dnr167/2009` by Benjamin Friedrich, Lisa Laun, Costas Meghir, and Luigi Pistaferri (see [NBER wp](https://www.nber.org/papers/w28527)). This project and ours are linked. The main data source should be the following list of files: `selectedf0educ1.dta`, `selectedf0educ2.dta`, `selectedf0educ3.dta`, `selectedfirms9708.dta`. These data sources are uniquely identified by the project ID `dnr123/2014` of the current project at the IFAU. This reference should be used to apply for access for replication. 

### Details on each Data Source

The confidential data sources are stata files with anonymized individual panel data. 

 - `selectedf0educ1.dta`, `selectedf0educ2.dta`, `selectedf0educ3.dta` include information about males' employment spells. We use the columns: employer anonymized identifier `peorgnrdnr2008025`, worker anonymized identifier `dnr2008025`, log real monthly earnings `logrealmanadslon`, number of months worked in the quarter `monthsworked`, age `age` and year `aret` and quarter `quarter`.
 - `selectedfirms9708.dta` includes information about firms' balance sheets. We use the columns: firm anonymized identifier `peorgnrdnr2008025`, reported value added `valueadded`, reported size `ant_anst`, year `aret` and broad industry `industry`.
 - See the paper web appendix for further details.

Computational requirements
---------------------------

### Software Requirements

This replication package **doesn't require any propriatory software** and all runtime libraries can be installed via [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

- Part I: R (code was last run with version 4.0.2 2020-06-22, taking-off-again on Win64)
  - `data.table` (1.14.2)
  - `foreign` (0.8-80)
  - One can run `conda install -c conda-forge r-base=4.0.2 r-foreign=0.8_80 r-data.table=1.14.2 r-futile.logger` to reproduce the environment used.
  - With data in place, simply run `R/main.r`.

- Part II and III: python (>=3.8)
  - See `requirements.txt` for the exact libraries used, alternatively, you can use the docker image provided at `tlamadon/balke-lamadon-2022-aer:latest` that was built using the conda environment file `environment.yml` and the docker file `Dockerfile`. 
  - To set up the environment on your system, you can run `conda env create -f environment.yml`.
  - To run the master script, run `python scriptflow.py run part3 -n 10` from within the `python` folder. You should replace 10 with the number of cores available to you. Replace `part3` with `part2` to run Part II.

### Memory and Runtime Requirements

The code will run with as little as 1 core and 2Gb of ram. It can however run with multicore, which would speed up the process. In such a case, we recommend having about 1.5Gb of memory per core. The number of cores is passed to `scriptflow.py` using the `-n` parameter. 

#### Details

The R code was last run on the Windows maching at IFAU. Computation took around 2 core-hours. This should run on any recent computer.

The python code was last run on [amazon aws](https://aws.amazon.com/) using an instance [c5a.24xlarge](https://aws.amazon.com/ec2/instance-types/c5/) with the docker image `tlamadon/balke-lamadon-2022-aer:clean`. Computation took around **900 core-hours**. 

Description of programs/code
----------------------------

The code is licensed under an MIT licence. See [licence.md](license.md) for details.

- Code files in `R/` contain the code for Part I. It will load the data, run sample selection, compute the moments and the sample statistics. 
  - The file `R/main.r` will run everything to create the inputs to Part II, which will be stored in the `results` folder.
- Code files in `python/` contain all the routines to generate the tables and figures. 
  - Code files in `python/wagedyn/` are organized as a library and should be used and imported as such. It provides the code that can solve and simulate the model when provided a set of given parameter values. Most classes and functions are documented using python docstrings. The package also includes a small set of tests in `python/wagedyn/test`.
  - `python/scriptflow.py` is the **master script** for Part II and III and should be run inside the conda environment provided in `environment.yml`. To run everything simply call `python scriptflow.py run all -n 10` where 10 should be replaced with the number of cores you would like to use. To run Part III only, call `python scriptflow.py run part3 -n 10`.
  - `results.py` imports `wagedyn` and is called from the master script to generate all intermediate inputs and the final tables and figures, which are saved in the `figures` folder.

Instructions to Replicators
---------------------------

## Part I (computing moments)

This has to be run on the IFAU secure data server.

- Prepare the `R` environment as described earlier with `conda install -c conda-forge r-base=4.0.2 r-foreign=0.8_80 r-data.table=1.14.2 r-futile.logger`. 
- Edit `R/main.r` to adjust the path to the data sources.
- Run `R/main.r` to generate all necessary output from the data that will be stored in the `results` folder. This creates `results/data-stats.csv` and `results/moments-jmp-2022.csv`.

## Part II (finding parameter estimates)

This can run anywhere with either python and conda or docker installed.

This part is computationally extremely costly when we don't have a good starting value. For the purpose of the replication package we start at the preferred values, run 15 iterations and see that the new parameter values are within the s.e. bounds computed in Part III, which uses the original preferred parameters.

You can choose between using the docker image or using a conda environment. We will describe the latter first:

- Prepare the `python` environment as descrived earlier with `conda env create -f environment.yml && conda activate balke-lamadon`.
- Make sure that `balkelamadon.toml`, `results/parameters_at_optimal.json` and `results/moments-jmp-2022.csv` are present.
- Call `python scriptflow.py run part2 -n 10` within the `python` folder, where you should replace 10 with the number of cores available to you.

Alternatively, with docker installed and without downloading the current package, you can simply run:

```shell
docker run -it tlamadon/balke-lamadon-2022-aer:clean python scriptflow.py run part2 -n 10
```

Intermediate figures and parameter updates will be in `python/build/opt/` within the container.

## Part III (creating all figures, tables and counterfactuals)

This can run anywhere with either python and conda or docker installed. You can choose between using the docker image or using a conda environment. We will describe the latter first:

- Prepare the `python` environment as descrived earlier with `conda env create -f environment.yml && conda activate balke-lamadon`.
- Make sure that `balkelamadon.toml`, `results/parameters_at_optimal.json` , `results/data-stats.csv` and `results/moments-jmp-2022.csv` are present.
- Start the master script with `python scriptflow.py run part3 -n 10` within the `python` folder.

Alternatively, with docker installed and without downloading the current package, you can simply run:

```shell
# start the computation of part III using docker
docker run -it tlamadon/balke-lamadon-2022-aer:clean python scriptflow.py run part3 -n 10 
```

This will download the docker image and start the computation in a container. Figures and tables will be generated in the folder `/app/figures` inside the container. You can retrieve this folder at any time (when the container running, or after its done) using a docker copy command as follows:

```shell
# extract the figures folder from the container to your local folder
docker cp `docker ps -alq`:/app/figures ./balke-lamadon-figures
```

The `docker ps -alq` command will automaically get the container id of the last started container. As you long you didn't start another container in the mean time, this will get the right container.


List of tables and programs
---------------------------

Provided `results/parameters_at_optimal.json` , `results/data-stats.csv` and `results/moments-jmp-2022.csv`, the code reproduces:

- [x] All numbers provided in text in the paper
- [x] All tables and figures in the paper

All functions mentioned nest here are in `python/results.py`:

| Figure/Table #             | Code                                  | Output file                           |
| -------------------------- | ------------------------------------- | ------------------------------------- |
| Table 1 (moments and fit)  | table_model_fit()                     | figures/table1-moments.tex            |
| Table 2 (parameters)       | cf_bootstrap_collect()                | figures/table2-parameters.tex         |
| Table 3 (var. dec. level)  | table_variance_decomposition_level()  | figures/table3-vdec-cs.tex            |
| Table 4 (var. dec. growth) | table_variance_decomposition_growth() | figures/table4-vdec-growth.tex        |
| Table 5 (passtrhough)      | passthrough_analysis()                | figures/table5-passthrough.tex        |
| Table 6 (policy)           | cf_policy_collect()                   | figures/table6-policy.tex             |
| Table W1 (sample stats)    | table_stats()                         | figures/tablew1-stats.tex             |
| Figure 3 (ir xshock)       | cf_model_to_life()                    | figures/figure3-ir-xshock.pdf         |
| Figure 4 (ir zshock)       | cf_model_to_life()                    | figures/figure4-ir-zshock.pdf         |
| Figure W1 (line search)    | surrogate_collect()                   | figures/fgiurew1-surrogate.pdf        |
| Figure W2 (obj. slices)    | cf_sensitivity_measure()              | figures/figurew2-slices-objective.pdf |
| Figure W3 (slices moments) | cf_sensitivity_measure()              | figures/figurew3-slices-moments.pdf   |
| Figure W4 (sensitivity)    | cf_sensitivity_measure()              | figures/figurew4-sensitivity.pdf      |
| Figures W5 (ir xshock fb)  | cf_model_to_life()                    | figures/figurew5-ir-xshock-fb.pdf     |
| Figures W6 (ir zshock fb)  | cf_model_to_life()                    | figures/figurew6-ir-zshock-fb.pdf     |


You can directly call any such function by using `python -c "import results as r; r.table_model_fit()"`. Of course, we recommend using the master script `scriptflow.py` instead as it will make sure that the different functions are called in the right order. There are several intermediate flows that `scriptflow.py` can run, for instance `python scriptflow.py run policy -n 10` will generate the policy table.

### Details on intermediate computations

For several of the tables and figure we first generate sets of parameter values at which the model needs to be solved, solve the model for these values in parallel using `main_solve_model_once.py` and finally collect the output. All these steps are handled by the master script `python/scriptflow.py`, see for instance `scriptflow.py:flow_policy()`. 

We describe here the intermediate files created by these procedures and where they can be found:

| Intermediate files                   | Code                        | Notes                                         |
| ------------------------------------ | --------------------------- | --------------------------------------------- |
| results/res_main_model.pkl           | save_model_solution()       | solves the mdoel at the optimal value         |
| results/res_main_fit.csv             | save_model_fit()            | simulate from the model at optimal value      |
| python/build/slices/*                | cf_slices_gen()             | computes the slices of the objective function |
| python/build/bootsrap/*              | cf_bootstrap_gen()          | computes all boostrap replications            |
| python/build/policy/*                | cf_policy_gen_neutral()     | computes grids for budget neutral policies    |
| python/build/surrogate/*             | surrogate_gen()             | create gird for surrogate picture             |
| python/build/optim/*                 | optimize_gen()              | saves intermediate steps of the optimizer     |
| python/build/cf/cf-vdec-level*.json  | cf_simulate_level_var_dec() | simulates for Table 3                         |
| python/build/cf/cf-vdec-growth*.json | cf_vardec_growth_rep_one()  | simulates for Table 4                         |

## Note on Conditional Numerical Reproducibility

This is notorioulsy difficult. All seeds and randomness are controlled for, even when running in parallel. The root seeds for the different tables and figures are set inside `balkelamadon.toml`.

We provide a docker container at [balke-lamadon-2022-aer](https://hub.docker.com/r/tlamadon/balke-lamadon-2022-aer) with all dependencies fixed. Yet we have noticed some rounding differences when using different hardware. To get the exact same numbers please use a similar CPU to the one we used.    

The final numbers were produced using the latest docker image on the following machine:

```txt
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              96
On-line CPU(s) list: 0-95
Thread(s) per core:  2
Core(s) per socket:  48
Socket(s):           1
NUMA node(s):        1
Vendor ID:           AuthenticAMD
CPU family:          23
Model:               49
Model name:          AMD EPYC 7R32
```

As mentioned before we used a [c5a.24xlarge](https://aws.amazon.com/ec2/instance-types/c5/) instance.

When using conda directly and potentially slightly different versions of the libraries, please set MKL_CBWR=COMPATIBLE as described on the intel website on [CNR](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-conditional-numerical-reproducibility-cnr.html) as well as MKL_NUM_THREADS=1 and MKL_DYNAMIC=FALSE as described [here](https://tut-arg.github.io/DCASE2017-baseline-system/reproducibility.html). To be as close as possible to the numbers in the paper, we recommend running the code on an AMD cpu and use the provided docker. 

## Extra notes on docker containers

You might want to recreate the docker container from scratch. We provide the `Dockerfile` to do that. You simply need to run `docker build -t some_name .` inside the root directory. 

In addition to the one-liners listed previously to run part II and III, we have bundled a jupyterlab inside the container. Run the following command to start it:

```shell
docker run --rm -it -p 127.0.0.1:8087:8080 tlamadon/balke-lamadon-2022-aer jupyter-lab --ip 0.0.0.0 --port 8080 --LabApp.token="" --notebook-dir /app
```

You should then be able to access the jupyterlab interface in your browser at [http://127.0.0.1:8087](http://127.0.0.1:8083/lab). You can then see the `figures` folder with all the results. You can also re-run computation by opening a terminal and typing in any scriptlfow command listed before. Note that I used the port 8087 which hopefully you are not using already, feel to change that.

## References

Friedrich, B., L. Laun, C. Meghir, and L. Pistaferri (2019a): “Data for: Earnings dynamics and firm-level shocks,” The Institute for Evaluation of Labour Market and Education Policy, Project ID dnr167/2009.
Friedrich, B., L. Laun, C. Meghir, and L. Pistaferri (2019b): “Earnings dynamics and firm-level shocks,” Discussion paper, National Bureau of Economic Research.
IFAU (2001–2006): “Data for: Project ID dnr123/2014,” The Institute for Evaluation of Labour Market and Education Policy, https://www.ifau.se/en/.

---

## Acknowledgements

Content for this readme was adapted from [AEA readme template](https://aeadataeditor.github.io/posts/2020-12-08-template-readme).

## Links

 - We maintain a github repository at [Github-balke-lamadon-2022-aer](https://github.com/tlamadon/balke-lamadon-2022-aer).
 - We also keep up-to-date containers at [Dockerhub-balke-lamadon-2022-aer](https://hub.docker.com/r/tlamadon/balke-lamadon-2022-aer).


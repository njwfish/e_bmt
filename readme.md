# Code for [A unified framework for bandit multiple testing](https://arxiv.org/pdf/2107.07322.pdf)

To install the pre-requisite packages, first install `conda` and `python`.
Install the requirements with:
```
conda env create -f environment.yml -n <environment name>
pip install -r requirements.txt
```
Then, activate the environment
```
conda activate <environment name>
```

## Documentation

There is partial documentation of the code through `sphinx`. To make the docs, run `make html` inside of the `docs/` folder, and then open `_build/index.html` document to get to the home page of the documentation.

## Reproducing the figures in the paper

There are 3 different simulations we conduct in the paper. The first is a comparison between p-variables with BH and e-variables with e-BH. This is the simulation described in Section 5 of the paper.

To run any of the experiments, run the following lines while in the main repo directory:
```
EXP_DIR=<directory with experiment configs> \
STYLE_FLAGS=<whether there are any style flags> \
REL_METHOD=<method to be compare other methods' relative times to> \
PROCESSES=<number of processors> \
OUTPUT_DIR=<output directory> \
mkdir -p $OUTPUT_DIR && \
scripts/run_configs.sh $EXP_DIR $PROCESSES $OUTPUT_DIR && \
scripts/plot_configs.sh $OUTPUT_DIR $OUTPUT_DIR "${REL_METHOD}" "${STYLE_FLAGS}"
```

`PROCESSES` and `OUTPUT_DIR` are chosen by you, the user running these experiments. The other three variables depend on the experiment you wish to reproduce.

For the simulations in Section 5, comparing BH w/ p-variables to e-BH with e-variables:
```
EXP_DIR=exp_configs/pve \
STYLE_FLAGS="" \
REL_METHOD="PUCB_EBH"
```

For the simulations in Section E.1 (Figure 5), comparing BH w/ *different choices* of p-variables to e-BH with e-variables:
```
EXP_DIR=exp_configs/supp/UCB \
STYLE_FLAGS="--style_map styles/supp.json" \
REL_METHOD="PM-H (E)"
```

For the simulations in Section E.2 (Figure 6), comparing BH w/ e-BH in a graph bandit (arms correspond to nodes) setting:
```
EXP_DIR=exp_configs/supp/graph
STYLE_FLAGS="--log_scale"
REL_METHOD="e-BH"
```



BibTex citation for this code/paper:
```
@inproceedings{xu2021unified,
  title={A unified framework for bandit multiple testing},
  author={Xu, Ziyu and Wang, Ruodu and Ramdas, Aaditya},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

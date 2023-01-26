# Code for the synthetic continuous task in the Continuous GFlowNets paper

To run the code:

```bash
conda create -n contgfn python=3.10
conda activate contfgn

pip install numpy matplotlib scikit-learn scipy torch tqdm

python main.py
```

and enjoy playing with the parameters. For examples, you could run

```bash
python main.py --loss tb --lr 1e-3 --delta 0.1 --PB tied --gamma_scheduler 0.5 --scheduler_milestone 2500 --n_components_s0 2 --n_components 1
```

If you want to get some nice plots, use `wandb`. Don't forget to set `USE_WANDB` to `True` in the header of `main.py`.
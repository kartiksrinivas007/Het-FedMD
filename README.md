## Instructions

Most of the AIJack implementatiosn require boost libraries installed for MPI

```
sudo apt install libboost-all-dev
sudo apt install libopenmpi-dev

```

Activate the environment and then 

```
pip install mpi4py
pip install "pybind11[global]"
pip install aijack
```

Or just clone requirements.txt

```
pip install -r requirements.txt
```

Once the environment is setup, go to file `env/lib/python3.8/site-packages/aijack/collaborative/fedmd/api.py`
and replace this file with the modified api `api_mod.py`


`Num_communication = args.rounds` contains the total number of global communication rounds.


The epochs for different steps(digest revisit etc) in FedMD are in the constructor in `main.py` and the optimizers contain the learning rate.
the function `self.transfer_phase_logging` in the api.py is where the pretraining happens to soem extent. Maybe that should be made better , try training for more epochs on those by changing the epoch value `transfer_epoch_public` in the args and the learning rates for the client because the accuracies may blow up

> The epochs are counted from one onwards not zero


> The code will run slow if the evaluation is called too many times. you can change the frequency of evaluations fo accuracy in `main.py` at line 118
change the modulo, make it 10 or 15.

The CSV files contain the accuracies in the order of clients row-wise, with the dataset id it is evaluated on and epoch number with it



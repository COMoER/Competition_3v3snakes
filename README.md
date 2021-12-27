# Competition_3v3snakes

#### Acknowledgement

This code is based on the [origin_repository](https://github.com/jidiai/Competition_3v3snakes)

---
### Dependency
You need to create competition environment.
>conda create -n snake3v3 python=3.6

>conda activate snake3v3

>pip install -r requirements.txt

---

### How to train qmix

- to train GRU model

>python rl_trainer_qmix/main.py

- to train MLP model

> python rl_trainer_qmix/main.py --step_model

As for the train parameter, please refer to `rl_trainer_qmix/main.py`.

You can locally evaluation your model.

>python evaluation_local.py --my_ai qmix --opponent rl


---
### How to test submission 
You can locally test your submission. At Jidi platform, we evaluate your submission as same as **run_log.py**

Once you run this file, you can locally check battle logs in the folder named "logs".

For example, 
>python run_log.py --my_ai "random" --opponent "rl"








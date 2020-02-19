# GADA

## Acknowledgements
This code was developed based on [dirt-t](https://github.com/RuiShu/dirt-t).

## Dependencies

```python
numpy==1.15.1
scikit_image==0.19.1
scipy==1.1.0
tensorflow_gpu==1.10.1
tensorbayes==0.4.0
```

## Download Data

Run the scripts in `./data/` to download MNIST and SVHN datasets.

## Run code

### Training
```
python main.py --datadir data --run 0 --src mnist --trg svhn
```

Tensorboard logs will be saved to `./log/` by default.

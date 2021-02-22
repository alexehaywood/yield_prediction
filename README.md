# yield_prediction
Prediction of reaction yields using support vector regression models built on structure-based and quantum chemical descriptors

## Dependencies
* Python (3.6)
* GraKel-dev (0.1a5)
* RDKit (2017.09.1)
* Matplotlib (3.3.2)
* xlrd (1.2.0)
* openpyxl (3.0.5)
* Scikit Learn (0.22.1)
* Numpy (1.19.1)
* Scipy (1.5.2)
* Pandas (1.1.1)

# Instructions

### Preprocessing
The data and quantum chemical descriptors in `yield_prediction/data/original` are from the open-source dataset published by Doyle et al.<sup>1</sup> (__[rxnpredict](https://github.com/doylelab/rxnpredict)__).

The Doyle et al. dataset is preprocessed using `yield_prediction/assemble_rxns.py`. The molecules in each reaction and corresponding yield data can be found in `yield_prediction/original/reactions`. Prospective combinatorial reactions are compiled from a list of molecules in each reaction component class.

### Model Development
Models are trained and tested using `run_ml_out-of-sample.py`. Prospective predictions are generated using `run_ml_validation.py`.

### Predictions
Model predictions for the out-of-sample tests and validation reactions were collated using `gather_results.py` and can be found in `yield_prediction/results`.

## References
[1] D. T. Ahneman, J. G. Estrada, S. Lin, S. D. Dreher and A. G. Doyle, *Science*, 2018, **360**, 186–190.

# yield_prediction

Predicting the yield of chemical reactions.

## Requirements
- Python (3.7)
- GraKel (0.1.8)
- OpenPyXL (3.0.5)
- RDKit (2021.03.4)
- scikit-learn (0.23.2)
- xlrd (1.2.0)

## Installation
Install yield_prediction using conda. 
Create a new conda environment with the yield_prediction requirements installed.
```sh
conda env create -f yield_prediction.yml
``` 
Activate the yield_prediction nvironment and verify that it was installed correctly.
```sh
conda activate yield_prediction
conda env list
```

## Usage
### Configuration 
The name of the configuration file name with start with "settings". Use `yield_prediction/settings.ini` as a template. Examples can be found in the `yield_prediction/settings_examples` directory.

#### FileSettings Section
| Parameter | Description |
| ------ | ------ |
| **input_reactions_fpath** = str | Excel file containing the smiles strings of the molecules in the reactions. |
| **output_log_fpath** = str | Output log file (*.log). |
| **output_table_fpaths** = comma-separated list | Directory to save results table to (one per split of data). |
| **output_model_fpaths** (optional) = comma-separated list | Directory to save trained models to (one per split of data). |
| **n_jobs** = int | Number of cores used to train the machine learning models. |

```ini
[FileSettings]
input_reactions_fpath = input/Doyle/reactions/rxns_smi.xlsx
output_log_fpath = output/wl_kernel.log
output_table_fpaths = output/results/wl_kernel/cross_validation, output/results/wl_kernel/activity_ranking, output/results/wl_kernel/loo output/results/wl_kernel/user_defined_test
output_model_fpaths = output/models/wl_kernel/cross_validation, output/models/wl_kernel/activity_ranking, output/models/wl_kernel/loo output/models/wl_kernel/user_defined_test
n_jobs = 1
```
#### DescriptorSettings
| Parameter | Description |
| ------ | ------ |
| **descriptor_type** = str | Type of descriptor. Options: quantum, wl_kernel, tanimoto_kernel, fingerprints, one-hot. |
| **descriptor_settings** = {} | Settings for the descriptors. Options: quantum, {'dir_descriptors': str}; wl_kernel,<sup>a</sup> {'n_iter': int}; fingerprints,<sup>b</sup> {fp_type: str, fps_kw={}}; tanimoto_kernel,<sup>b</sup> {fp_type: str, fps_kw={}}; one-hot, {}.  |
| **descriptor_cols** = comma-separated list | List of column headings in ***input_reactions_fpath*** corresponding to the descriptors. |
| **descriptor_index** = python list | List of column headings corresponding to the names of the reaction components.  |
| **target_col** = str | Name of column heading in ***input_reactions_fpath*** corresponding to the target.

```ini
[DescriptorSettings]
descriptor_type = wl_kernel
descriptor_settings = {'n_iter': 5}
descriptor_cols = additive_smiles, aryl_halide_smiles, base_smiles, ligand_smiles
descriptor_index = ['additive', 'aryl_halide', 'base', 'ligand']
target_col = yield_exp
```
> <sup>a</sup> Additional options can be found in the [GraKel documentation](https://ysig.github.io/GraKeL/0.1a8/generated/grakel.WeisfeilerLehman.html#grakel.WeisfeilerLehman).
<sup>b</sup> fp_type, fps_kw options: 'rdk', {'fpSize': int}; 'morgan', {'radius': int, 'useFeatures': bool, 'nBits': int}; 'maccs', {}.

#### MachineLearningSettings
| Parameter | Description |
| ------ | ------ |
| **models** = comma-separated list | Name of machine learning algorithm. Options: svr-linear_kernel, svr-poly_kernel, svr-RBF_kernel, svr-sigmoid_kernel, linear_regression, lasso, ridge, elastic_net, bayesian_ridge, k-nearest_neighbours, random_forest, gradient_boosting, decision_tree. | 
```ini
[MachineLearningSettings]
models = svr-linear_kernel, svr-poly_kernel, svr-RBF_kernel, svr-sigmoid_kernel
```

#### InternalSplitterSettings
| Parameter | Description |
| ------ | ------ |
| **splitter** = comma-separated list | Name of splitter. Options: cross-validation, activity_ranking, leave-one-component-out, user-defined_mols. |
| **splitter_settings** = python list of dictionaries | Settings for the splitter (one dictionary per splitter). Options: cross-validation, {}; activity_ranking, {'rxn_component': str, 'n_splits': int}; leave-one-component-out, {'rxn_component': str}; user-defined_mols, {'rxn_component': str, 'test_sets_mols': list of lists, 'test_sets_names': list}. |

```ini
[InternalSplitterSettings]
splitter = cross-validation, activity_ranking, leave-one-component-out, user-defined_mols
splitter_settings = [{}, {'rxn_component': 'aryl_halide', 'n_splits': 3}, {'rxn_component': 'base'}, {'rxn_component': 'additive', 'test_sets_mols': [['additive1', 'additive2'], ['additive3', 'additive4']], 'test_sets_names': ['set1', 'set2']}]
```

#### ValidationSettings
| Parameter | Description |
| ------ | ------ |
| **validation_reactions_fpath** = str | Excel file containing the smiles strings of the molecules in the validation reactions. |
| **validation_output_table_fpath** = str | Directory to save results table to. |
| **validation_output_model_fpath** = str | Directory to save trained models to. |
| **pretrained_model_fpath** = | *In development.* | 

```ini
[ValidationSettings]
validation_reactions_fpath = input/validation/reactions/rxns_smi.xlsx
validation_output_table_fpath = output/results/wl_kernel
validation_output_model_fpath = output/models/wl_kernel
pretrained_model_fpath = 
```

### Run on the Command Line

```sh
python main.py -f <settings_fpath>
```

| Parameter | Description |
| ------ | ------ |
| -f <settings_fpath> | Path to configuration (settings.ini) file in your working directory. |

## Datasets
### Doyle *et al*. 
The SMILES string of the molecules in the Doyle *et al.* reactions and corresponding reaction yields can be found in `yield_prediction/input/Doyle/reactions/*_smi.xlsx`. The quantum chemical descriptors of the molecules in the reactions can be found in `yield_prediction/input/Doyle/quantum_descriptors/`. 
### Prospective Reactions
The SMILES string of the molecules in the prospective reactions can be found in `yield_prediction/input/validation/reactions/*_smi.xlsx`. The quantum chemical descriptors of the molecules in the Doyle *et al.* dataset and prospective reactions can be found in `yield_prediction/input/quantum_descriptors_missing_additive/`. 
### Using Your Own Dataset
Enter the names and SMILES strings of the molecules in the dataset into an excel document using the format below. 
| componet1 | componet2 | componet3 | componet1_SMILES | componet2_SMILES | componet3_SMILES | target |
| --- | --- | --- | --- | --- | --- | --- |
| componet1a | componet2a | componet3a | componet1a smiles | componet2a smiles | componet3a smiles | x |
| componet1a | componet2b | componet3b | componet1a smiles | componet2b smiles | componet3b smiles | y | 
| componet1a | componet2b | componet3c | componet1a smiles | componet2b smiles | componet3c smiles | z |
For this example, the **descriptor_cols**, **descriptor_index** and **target_col** in the configuration file would be defined as:
```ini
descriptor_cols = component1_SMILES, component2_SMILES, component1_SMILES
descriptor_index = ['componet1_SMILES', 'componet2_SMILES', 'componet3_SMILES']
target_col = target
```

Quantum chemical descriptors for the molecules in each reaction component should be defined in seperate excel files. If defining training reactions using **input_reactions_fpath** and test reactions **validation_reactions_fpath**, quantum chemical descripors of the molecules in each reaction component for both reactions should be defined in the same seperate excel files. 



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Doyle *et al.*]: <https://github.com/joemccann/dillinger>

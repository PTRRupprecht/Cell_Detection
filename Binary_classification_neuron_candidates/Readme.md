## Classification of cell candidates into cells and non-cells

Here, we provide a simple classifier based on 2D CNN networks to classify an input (`N x 1 x 91 x 31 x 31`) with `N` the number of cell candidates into cells vs. non-cells. For each cell, a local environment from a 3D stack is selected (91 pixels in z, 31 in x and y) and used to infer whether the cell candidate is a neuron (centered in the local environment) or not.

Three networks for different sections (xy, xz, yz) are combined by adding their output. All networks are standard convolutional networks and address the question: Is this cell candidate a cell or not?

<!---![Deep 2D CNN used for classification of cell candidates into cells and non-cells](https://github.com/PTRRupprecht/Cell_Detection/blob/main/Binary_classification_neuron_candidates/DeepNetwork2DCNN.png)--->
<p align="center"><img src="https://github.com/PTRRupprecht/Cell_Detection/blob/main/Binary_classification_neuron_candidates/DeepNetwork2DCNN.png"  width="55%"></p>

The results from an ensemble of 5 models for each network type (xy, xz, yz) are combined to yield a more confident predictions about cell vs. non-cell.

`Train_binary_classification_2DCNN.py` trains all models. Training and inference was performed in in Python 3.7.12 with Torch 1.12.1.post201 with GPU support. However, due to the low complexity of this deep network model, most standard installations of Torch are likely to work as well. Ground truth (1900 manually annotated cell candidates) can be found in the [ground truth data folder](https://github.com/PTRRupprecht/Cell_Detection/tree/main/Binary_classification_neuron_candidates/ground_truth_data). Pretrained models can be found in the [pretrained models folder](https://github.com/PTRRupprecht/Cell_Detection/tree/main/Binary_classification_neuron_candidates/trained_models).

`Apply_binary_classification_2DCNN.py` applies the trained network to new data. The example script applies the trained network to the already existing training data for the sake of simplicity.

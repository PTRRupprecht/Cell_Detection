## Interactive labeling of cell candidates into cells and non-cells

We provide a simple interactive Python tool to label cell candidates as cells or non-cells. This manual annotation is then used to train a [supervised cell-classifier](https://github.com/PTRRupprecht/Cell_Detection/tree/main/Binary_classification_neuron_candidates) based on convolutional deep networks. As described there, the input for classification/manual labeling is a local 3D environment around each cell of `91 x 31 x 31` pixels.

Two orthogonal cross-sections (left: xz, right: xy) are simultaneously viewed to enable a well-informed decision whether the inspected cell candidate is a cell or not:

<!---![Manual classification of cell candidates into cells and non-cells to generate a ground truth](https://github.com/PTRRupprecht/Cell_Detection/blob/main/Label_tool/Label_tool_interface.png)--->
<p align="center"><img src="https://github.com/PTRRupprecht/Cell_Detection/blob/main/Label_tool/Label_tool_interface.png"  width="55%"></p>

The following keyboard shortcuts are used to interact with the user interface, to decide about "cell" vs. "non-cell" and to more closely inspect the data. The mouse scroll wheel can be used to go through the local environment of the cross-section:

`b`: Switch for scrolling between the xz-axis or the xy-axis view; default is xy-view

`g`: Inspected candidate is a cell; continue with next sample

`d`: Inspected candidate is not a cell; continue with next sample

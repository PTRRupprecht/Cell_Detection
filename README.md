## Cell detection in large 3D brain samples

This repository provides Python code for the improvement of the detection of neurons in large brain samples, as described here (add link to paper once it is online).

First, a **cell segmentation**. An initial guess of cell candidates is made using the ClearMap 1.0 ([link](https://github.com/ChristophKirst/ClearMap)) framework based on a supervised pixel-wise Ilastik classifier ([link](https://www.ilastik.org/)). This part of the detection code is not provided within this repository.

Second, a **gradient-based cell merger**. The initial cell candidate guesses are refined based on a peak finder that uses local gradient search to find local maxima. Spurious multiple detections of the same cell are thereby merged together.

Third, a **supervised cell classifier**. The refined cell candidates are visually classified into cells and non-cells. To this end, a supervised convolutional network has been train on manually annotated data.

<!---![Pipeline for reliable cell detection](https://github.com/PTRRupprecht/Cell_Detection/blob/main/Overview_pipeline.png)--->
<p align="center"><img src="https://github.com/PTRRupprecht/Cell_Detection/blob/main/Overview_pipeline.png"  width="35%"></p>

The algorithms and code together with example data, ground truth and pretrained models are available in the subfolders referring to [gradient-based cell refinement](https://github.com/PTRRupprecht/Cell_Detection/tree/main/Gradient_cell_merger) and [classification](https://github.com/PTRRupprecht/Cell_Detection/tree/main/Binary_classification_neuron_candidates). Please note that these subfolders of the Github repository also come with their own Readme files.

In addition, we provide a tool for [manual annotation of cells vs. non-cells](https://github.com/PTRRupprecht/Cell_Detection/tree/main/Label_tool).

Comments and questions are welcome as Github issues or via [email](mailto:ptrrupprecht+celldetection@gmail.com).

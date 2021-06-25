### Multi-scale convolutional neural networks for cloud segmentation

This repository contains code of the paper [Multi-scale convolutional neural networks for cloud segmentation](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11531/115310E/Multi-scale-convolutional-neural-networks-for-cloud-segmentation/10.1117/12.2573810.short)

### Requirements
- Python 3
- Tensorflow 2.0
### Datasets

- [38-Cloud dataset](https://www.kaggle.com/sorour/38cloud-cloud-segmentation-in-satellite-images)
- [SPARCS dataset](https://www.usgs.gov/core-science-systems/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs)

### Preprocessing

Before training the model we have to do some preprocessing for the raw data to fit the model input size. Usually
satellite TIF images are composed of many bands, we chose four spectral bands for
processing. Since in the 38-Cloud dataset, the bands are already extracted from the original images in from of
small batches, we just stack the bands together, resize them to 224x224, and normalise them before they used as
input of the ConvNet model. But the SPARCS dataset requires extracting the bands manually first, then resizing
and normalising. Run the file l8-raw2train.py to extract the bands from the sparcs dataset.

- l8 suffix meand landsat8 (SPARCS dataset)
- 38 suffix means 38-cloud dataset


### Models
To train the model on 38-cloud dataset run 38-train.py it will preprocess the data and traine the model. The data must be in the folder 38-Cloud_training
To train the model on SPARCS dataset run l8-train.py after the preprocessing using l8-raw2train.py.
### Results
- To test the pretrained model "best_model_38.hdf5" on 38-Cloud dataset use the file 38-test.py and 38-test-eval.py
- Segmentation results on 38-Cloud dataset are show bellow:
![Capture](https://user-images.githubusercontent.com/50513215/123413321-8bb53f80-d5aa-11eb-9818-959ce5031e01.PNG)

- To test the pretrained model "best_model_l8.hdf5" on SPARCS dataset use the file l8-test.py
- Segmentation results on SPARCS dataset are show bellow:
![Capture1](https://user-images.githubusercontent.com/50513215/123413334-9079f380-d5aa-11eb-968e-54fd4822752f.PNG)


#### Citation
If you use this code for your research, please cite the following paper:
```
@inproceedings{aouaidjia2020multi,
  title={Multi-scale convolutional neural networks for cloud segmentation},
  author={Aouaidjia, Kamel and Boukerch, Issam},
  booktitle={Remote Sensing of Clouds and the Atmosphere XXV},
  volume={11531},
  pages={115310E},
  year={2020},
  organization={International Society for Optics and Photonics}
}

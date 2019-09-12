# cosine_metric_learning

## Introduction

This repository contains code for training a metric feature representation to be
used with the [deep_sort tracker](https://github.com/nwojke/deep_sort). The
approach is described in

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }

Pre-trained models used in the paper can be found
[here](https://drive.google.com/open?id=13HtkxD6ggcrGJLWaUcqgXl2UO6-p4PK0).
A preprint of the paper is available [here](http://elib.dlr.de/116408/).
The repository comes with code to train a model on the
[Market1501](http://www.liangzheng.org/Project/project_reid.html)
and [MARS](http://www.liangzheng.com.cn/Project/project_mars.html) datasets.


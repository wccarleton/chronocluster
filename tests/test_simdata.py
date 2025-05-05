#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster

import pytest
from chronocluster.data.simdata import generate_random_points, generate_params
from chronocluster.clustering import Point

def test_generate_random_points():
    n_points = 100
    cluster_centers = [(5, 5), (15, 15)]
    cluster_stds = [1.0, 1.5]
    start_type = 'norm'
    start_hyperparams = [10, 2, 5]
    end_type = 'uniform'
    end_hyperparams = [20, 3, 5]
    
    points = generate_random_points(n_points, 
                                    cluster_centers, 
                                    cluster_stds, 
                                    start_type, 
                                    start_hyperparams, 
                                    end_type, 
                                    end_hyperparams)
    
    # Check the number of generated points
    assert len(points) == n_points
    
    # Check if each point is an instance of the Point class
    for point in points:
        assert isinstance(point, Point)

def test_generate_params():
    dist_type = 'norm'
    hyperparams = [10, 2, 5]
    params = generate_params(dist_type, hyperparams)
    
    # Check the length of the parameters list
    assert len(params) == 2
    
    dist_type = 'uniform'
    params = generate_params(dist_type, hyperparams)
    
    # Check the length of the parameters list
    assert len(params) == 2

if __name__ == '__main__':
    pytest.main()

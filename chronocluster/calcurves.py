#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster

import pandas as pd


def download_intcal20():
    url = "https://intcal.org/curves/intcal20.14c"
    intcal20 = pd.read_csv(url, skiprows=10, delimiter=",")
    intcal20.columns = ["calbp", "c14bp", "c14_sigma", "f14c", "f14c_sigma"]

    # Create the dictionary in the required format
    intcal20_dict = {
        "calbp": intcal20["calbp"].values,
        "c14bp": intcal20["c14bp"].values,
        "c14_sigma": intcal20["c14_sigma"].values,
    }
    return intcal20_dict


# Example usage to create initial intcal20 dictionary
intcal20 = download_intcal20()

# Users can add their custom curves to this dictionary
calibration_curves = {"intcal20": intcal20}

# Users can update this dictionary with their custom curves
# Example: calibration_curves['custom_curve'] = custom_curve_data

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-08 22:02
# @Author  : Jasontang
# @Site    : 
# @File    : run.py
# @ToDo    : 

import matplotlib.pyplot as plt
import numpy as np
import requests
import json


def show_images(result_images, filenames, probs, captions=None):
    result_images = list(result_images)
    filenames = list(filenames)
    probs = list(probs)
    #     captions = list(captions)
    nimages = np.array(result_images).shape[0]
    index = 0
    rows = 1 if (nimages < 3) else np.ceil(nimages / 3.0)
    print("rows:", rows)
    fig = plt.figure(figsize=(50, 50))
    for i in range(nimages):
        plt.subplot(rows, 3, i + 1)
        img = plt.imread(result_images[i])
        plt.imshow(img)
        plt.title("prob:" + str(probs[i]), fontsize=45)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("result.jpg")
    print("The result already in the result.jpg")
    plt.show()


url = "http://0.0.0.0:5000/imagecaption/do"
try:
    result = requests.post(url)
    result = json.loads(result.content.decode('utf-8'))
    if result['status'] == 'ok':
        print(result)
        resultInfo = json.loads(result["resultInfo"])
        img_result = json.loads(result["img_results"])
        # print("caption\t\t source_img\t\t prob\t\t filename\t\t result_img") for caption, source_img, prob,
        # filename, result_img in zip(resultInfo["caption"].values(), resultInfo["image_files"].values(), resultInfo[
        # "prob"].values(), img_result.keys(), img_result.values()): print(caption, source_img, prob, filename,
        # result_img)
        show_images(img_result.values(), img_result.keys(), resultInfo["prob"].values())
    else:
        print("网络错误，请稍后再试")
except Exception as e:
    print(e)

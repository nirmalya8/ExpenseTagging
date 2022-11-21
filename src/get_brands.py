#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
def get_brands_json():
    brands_file = 'Misc/brands.txt'
    categories_file = 'Misc/categories.txt'

    with open(brands_file) as f:
        brands = [line.rstrip() for line in f]
    
    with open(categories_file) as f:
        categories = [line.rstrip() for line in f]
    
    print(len(brands), len(categories))
    brands_dict = {}
    for i in range(len(brands)):
        brands_dict[str(i)] = {"name": brands[i], "category": categories[i]}
    
    json_file = 'brands.json'
    print(brands_dict)

    with open(json_file, 'w') as fp:
        json.dump(brands_dict, fp)
get_brands_json()

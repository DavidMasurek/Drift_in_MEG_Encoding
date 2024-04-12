import json

# Read crop_dict from json
crop_dict_file = open("crop_dict.json")
crop_dict_string = crop_dict_file.read()
crop_dict = json.loads(crop_dict_string)

#print(crop_dict["sessions"]["1"]["trials"]["1"]["timepoints"]["0.2829999999990105"])
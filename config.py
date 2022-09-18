import json
import dLux

def read_json(json: str) -> dict:
    with open(json) as json_file:
        contents = json.load(json_file)
    return contents


def decode_object(obj: dict) -> object:
    if obj in dLux.__all__:
         exec(f"{obj}(**{})")

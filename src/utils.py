import json


def save_args_to_file(args, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(vars(args), output_file, indent=4)
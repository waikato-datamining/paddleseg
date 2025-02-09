import argparse
import os
import traceback
import yaml
from typing import Optional, List, Any

from paddleseg.cvlibs.config import Config


def check_file(file_type: str, path: Optional[str]):
    """
    Checks whether file exists and does not point to a directory.
    Raises an exception otherwise.

    :param file_type: the string describing the file, used in the exceptions
    :type file_type: str
    :param path: the path to check, checks ignored if None
    :type path: str
    """
    if path is not None:
        if not os.path.exists(path):
            raise IOError("%s not found: %s" % (file_type, path))
        if os.path.isdir(path):
            raise IOError("%s points to a directory: %s" % (file_type, path))


def is_bool(s: str) -> bool:
    """
    Checks whether the string is a boolean value.

    :param s: the string to check
    :type s: str
    :return: True if a boolean
    :rtype: bool
    """
    try:
        bool(s)
        return True
    except:
        return False


def is_int(s: str) -> bool:
    """
    Checks whether the string is an int value.

    :param s: the string to check
    :type s: str
    :return: True if an int
    :rtype: bool
    """
    try:
        int(s)
        return True
    except:
        return False


def is_float(s: str) -> bool:
    """
    Checks whether the string is a float value.

    :param s: the string to check
    :type s: str
    :return: True if a float
    :rtype: bool
    """
    try:
        float(s)
        return True
    except:
        return False


def set_value(config: dict, path: List[str], value: Any):
    """
    Sets the value in the YAML config according to its path.

    :param config: the config dictionary to update
    :type config: dict
    :param path: the list of path elements to use for navigating the hierarchical dictionary
    :type path: list
    :param value: the value to use
    """
    current = config
    found = False
    for i in range(len(path)):
        if path[i] in current:
            if i < len(path) - 1:
                current = current[path[i]]
            else:
                found = True
                if isinstance(current[path[i]], bool):
                    current[path[i]] = bool(value)
                elif isinstance(current[path[i]], int):
                    current[path[i]] = int(value)
                elif isinstance(current[path[i]], float):
                    current[path[i]] = float(value)
                elif isinstance(current[path[i]], list):
                    values = value.split(",")
                    # can we infer type?
                    if len(current[path[i]]) > 0:
                        if isinstance(current[path[i]][0], bool):
                            current[path[i]] = [bool(x) for x in values]
                        elif isinstance(current[path[i]][0], int):
                            current[path[i]] = [int(x) for x in values]
                        elif isinstance(current[path[i]][0], float):
                            current[path[i]] = [float(x) for x in values]
                        else:
                            current[path[i]] = values
                else:
                    current[path[i]] = value
        elif path[i].startswith("[") and path[i].endswith("]") and isinstance(current, list):
            index = int(path[i][1:len(path[i])-1])
            if index < len(current):
                current = current[index]
        else:
            # not present, we'll just add it
            if i == len(path) - 1:
                print("Adding option: %s" % (str(path)))
                if is_bool(value):
                    current[path[i]] = bool(value)
                elif is_int(value):
                    current[path[i]] = int(value)
                elif is_float(value):
                    current[path[i]] = float(value)
                else:
                    current[path[i]] = value
                found = True
            break
    if not found:
        print("Failed to locate path in config: %s" % str(path))


def remove_value(config: dict, path: List[str]):
    """
    Removes the value from the YAML config according to its path.

    :param config: the config dictionary to update
    :type config: dict
    :param path: the list of path elements to use for navigating the hierarchical dictionary
    :type path: list
    """
    current = config
    removed = False
    for i in range(len(path)):
        if path[i] in current:
            if i < len(path) - 1:
                current = current[path[i]]
            else:
                del current[path[i]]
                removed = True
        elif path[i].startswith("[") and path[i].endswith("]") and isinstance(current, list):
            index = int(path[i][1:len(path[i])-1])
            if index < len(current):
                current = current[index]
        else:
            break
    if not removed:
        print("Failed to locate path in config, cannot remove: %s" % str(path))


def export(input_file: str, output_file: str, train_annotations: str = None, val_annotations: str = None,
           num_classes: int = None, batch_size: int = None, num_iters: int = None,
           additional: List[str] = None, remove: List[str] = None):
    """
    Exports the config file while updating specified parameters.

    :param input_file: the template YAML config file to load and modify
    :type input_file: str
    :param output_file: the YAML file to store the updated config data in
    :type output_file: str
    :param train_annotations: the text file with the training annotations/images relation, ignored if None
    :type train_annotations: str
    :param val_annotations: the text file with the validation annotations/images relation, ignored if None
    :type val_annotations: str
    :param num_classes: the number of classes in the dataset, ignored if None
    :type num_classes: int
    :param batch_size: the batch size to use, ignored if None
    :type batch_size: int
    :param num_iters: the number of epochs to train, ignored if None
    :type num_iters: int
    :param additional: the list of additional parameters to set, format: PATH:VALUE, with PATH being the dot-notation path through the YAML parameter hierarchy in the file; if VALUE is to update a list, then the elements must be separated by comma
    :type additional: list
    :param remove: the list of parameters to remove, format: PATH, with PATH being the dot-notation path through the YAML parameter hierarchy in the file
    :type remove: list
    """
    # some sanity checks
    check_file("Config file", input_file)
    check_file("Training annotations", train_annotations)
    check_file("Validation annotations", val_annotations)
    if (num_classes is not None) and (num_classes < 1):
        num_classes = None

    # load template
    print("Loading config from: %s" % input_file)
    config = Config(input_file)
    config = config.dic

    if train_annotations is not None:
        set_value(config, ["train_dataset", "type"], "Dataset")
        set_value(config, ["train_dataset", "dataset_root"], os.path.dirname(train_annotations))
        set_value(config, ["train_dataset", "train_path"], train_annotations)
        if num_classes is not None:
            set_value(config, ["train_dataset", "num_classes"], num_classes)

    if val_annotations is not None:
        set_value(config, ["val_dataset", "type"], "Dataset")
        set_value(config, ["val_dataset", "dataset_root"], os.path.dirname(val_annotations))
        set_value(config, ["val_dataset", "val_path"], val_annotations)
        if num_classes is not None:
            set_value(config, ["val_dataset", "num_classes"], num_classes)

    if batch_size is not None:
        set_value(config, ["batch_size"], batch_size)

    if num_iters is not None:
        set_value(config, ["iters"], num_iters)

    if num_classes is not None:
        set_value(config, ["model", "num_classes"], num_classes)

    if additional is not None:
        for add in additional:
            if ":" in add:
                path_str, value = add.split(":")
                path = path_str.split(".")
                set_value(config, path, value)
            else:
                print("Invalid format for additional parameter, expected PATH:VALUE but found: %s" % add)

    if remove is not None:
        for rem in remove:
            path = rem.split(".")
            remove_value(config, path)

    print("Saving config to: %s" % output_file)
    with open(output_file, "w") as fp:
        yaml.dump(config, fp)


def main(args=None):
    """
    Performs the bash.bashrc generation.
    Use -h to see all options.

    :param args: the command-line arguments to use, uses sys.argv if None
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description='Exports a PaddleSeg config file and updates specific fields with user-supplied values.',
        prog="paddleseg_export_config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", metavar="FILE", required=True, help="The PaddleClass YAML config file template to export.")
    parser.add_argument("-o", "--output", metavar="FILE", required=True, help="The YAML file to store the exported config file in.")
    parser.add_argument("-t", "--train_annotations", metavar="FILE", required=False, help="The text file with the labels for the training data (images are expected to be located below that directory).")
    parser.add_argument("-v", "--val_annotations", metavar="FILE", required=False, help="The text file with the labels for the validation data (images are expected to be located below that directory).")
    parser.add_argument("-c", "--num_classes", metavar="NUM", required=False, type=int, help="The number of classes in the dataset.")
    parser.add_argument("-I", "--num_iters", metavar="NUM", required=False, type=int, help="The number of iterations to train.")
    parser.add_argument("-a", "--additional", metavar="PATH:VALUE", required=False, help="Additional parameters to override; format: PATH:VALUE, with PATH representing the dot-notation path through the parameter hierarchy in the YAML file, if VALUE is to update a list, then the elements must be separated by comma.", nargs="*")
    parser.add_argument("-r", "--remove", metavar="PATH", required=False, help="Parameters to remove; format: PATH, with PATH representing the dot-notation path through the parameter hierarchy in the YAML file", nargs="*")
    parsed = parser.parse_args(args=args)
    export(parsed.input, parsed.output,
           train_annotations=parsed.train_annotations, val_annotations=parsed.val_annotations,
           num_classes=parsed.num_classes, num_iters=parsed.num_iters,
           additional=parsed.additional, remove=parsed.remove)


def sys_main():
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    :rtype: int
    """

    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())

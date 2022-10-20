"""
Save util taken from stable_baselines
used to serialize data (class parameters) of model classes
"""
import base64
import functools
import io
import json
import os
import pathlib
import pickle
import warnings
import zipfile
from typing import Any, Dict, Optional, Tuple, Union

import cloudpickle
import jax
import numpy as np
import haiku as hk
import jax.numpy as jnp

from sb3_jax.common.type_aliases import JnpDict 
from sb3_jax.common.utils import get_system_info


def recursive_getattr(obj: Any, attr: str, *args) -> Any:
    """Recursive version of getattr"""
    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recursive_setattr(obj: Any, attr: str, val: Any) -> None:
    """Recursive version of setattr."""
    pre, _, post = attr.rpartition(".")
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)


def is_json_serializable(item: Any) -> bool:
    """Test if an object is serializable into JSON."""
    # Try with try-except struct.
    json_serializable = True
    try:
        _ = json.dumps(item)
    except TypeError:
        json_serializable = False
    return json_serializable


def data_to_json(data: Dict[str, Any]) -> str:
    """Turn data (class parameters) into a JSON string for storing. """
    # First, check what elements can not be JSONfied,
    # and turn them into byte-strings
    serializable_data = {}
    for data_key, data_item in data.items():
        # See if object is JSON serializable
        if is_json_serializable(data_item):
            # All good, store as it is
            serializable_data[data_key] = data_item
        else:
            # Not serializable, cloudpickle it into
            # bytes and convert to base64 string for storing.
            # Also store type of the class for consumption
            # from other languages/humans, so we have an
            # idea what was being stored.
            base64_encoded = base64.b64encode(cloudpickle.dumps(data_item)).decode()

            # Use ":" to make sure we do
            # not override these keys
            # when we include variables of the object later
            cloudpickle_serialization = {
                ":type:": str(type(data_item)),
                ":serialized:": base64_encoded,
            }

            # Add first-level JSON-serializable items of the
            # object for further details (but not deeper than this to
            # avoid deep nesting).
            # First we check that object has attributes (not all do,
            # e.g. numpy scalars)
            if hasattr(data_item, "__dict__") or isinstance(data_item, dict):
                # Take elements from __dict__ for custom classes
                item_generator = data_item.items if isinstance(data_item, dict) else data_item.__dict__.items
                for variable_name, variable_item in item_generator():
                    # Check if serializable. If not, just include the
                    # string-representation of the object.
                    if is_json_serializable(variable_item):
                        cloudpickle_serialization[variable_name] = variable_item
                    else:
                        cloudpickle_serialization[variable_name] = str(variable_item)

            serializable_data[data_key] = cloudpickle_serialization
    json_string = json.dumps(serializable_data, indent=4)
    return json_string


def json_to_data(json_string: str, custom_objects: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Turn JSON serialization of class-parameters back into dictionary."""
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects argument must be a dict or None")

    json_dict = json.loads(json_string)
    # This will be filled with deserialized data
    return_data = {}
    for data_key, data_item in json_dict.items():
        if custom_objects is not None and data_key in custom_objects.keys():
            # If item is provided in custom_objects, replace
            # the one from JSON with the one in custom_objects
            return_data[data_key] = custom_objects[data_key]
        elif isinstance(data_item, dict) and ":serialized:" in data_item.keys():
            # If item is dictionary with ":serialized:"
            # key, this means it is serialized with cloudpickle.
            serialization = data_item[":serialized:"]
            # Try-except deserialization in case we run into
            # errors. If so, we can tell bit more information to
            # user.
            try:
                base64_object = base64.b64decode(serialization.encode())
                deserialized_object = cloudpickle.loads(base64_object)
            except (RuntimeError, TypeError):
                warnings.warn(
                    f"Could not deserialize object {data_key}. "
                    + "Consider using `custom_objects` argument to replace "
                    + "this object."
                )
            return_data[data_key] = deserialized_object
        else:
            # Read as it is
            return_data[data_key] = data_item
    return return_data


@functools.singledispatch
def open_path(path: Union[str, pathlib.Path, io.BufferedIOBase], mode: str, verbose: int = 0, suffix: Optional[str] = None):
    if not isinstance(path, io.BufferedIOBase):
        raise TypeError("Path parameter has invalid type.", io.BufferedIOBase)
    if path.closed:
        raise ValueError("File stream is closed.")
    mode = mode.lower()
    try:
        mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
    except KeyError:
        raise ValueError("Expected mode to be either 'w' or 'r'.")
    if ("w" == mode) and not path.writable() or ("r" == mode) and not path.readable():
        e1 = "writable" if "w" == mode else "readable"
        raise ValueError(f"Expected a {e1} file.")
    return path


@open_path.register(str)
def open_path_str(path: str, mode: str, verbose: int = 0, suffix: Optional[str] = None) -> io.BufferedIOBase:
    return open_path(pathlib.Path(path), mode, verbose, suffix)


@open_path.register(pathlib.Path)
def open_path_pathlib(path: pathlib.Path, mode: str, verbose: int = 0, suffix: Optional[str] = None) -> io.BufferedIOBase:
    if mode not in ("w", "r"):
        raise ValueError("Expected mode to be either 'w' or 'r'.")

    if mode == "r":
        try:
            path = path.open("rb")
        except FileNotFoundError as error:
            if suffix is not None and suffix != "":
                newpath = pathlib.Path(f"{path}.{suffix}")
                if verbose == 2:
                    warnings.warn(f"Path '{path}' not found. Attempting {newpath}.")
                path, suffix = newpath, None
            else:
                raise error
    else:
        try:
            if path.suffix == "" and suffix is not None and suffix != "":
                path = pathlib.Path(f"{path}.{suffix}")
            if path.exists() and path.is_file() and verbose == 2:
                warnings.warn(f"Path '{path}' exists, will overwrite it.")
            path = path.open("wb")
        except IsADirectoryError:
            warnings.warn(f"Path '{path}' is a folder. Will save instead to {path}_2")
            path = pathlib.Path(f"{path}_2")
        except FileNotFoundError:  # Occurs when the parent folder doesn't exist
            warnings.warn(f"Path '{path.parent}' does not exist. Will create it.")
            path.parent.mkdir(exist_ok=True, parents=True)

    # if opening was successful uses the identity function
    # if opening failed with IsADirectory|FileNotFound, calls open_path_pathlib
    #   with corrections
    # if reading failed with FileNotFoundError, calls open_path_pathlib with suffix

    return open_path(path, mode, verbose, suffix)


def save_to_zip_file(
    save_path: Union[str, pathlib.Path, io.BufferedIOBase],
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
) -> None:
    """Save model data to a zip archive."""
    if data is not None:
        serialized_data = data_to_json(data)
        data_path = open_path(os.path.join(save_path, "data"), "w", verbose=verbose, suffix="zip")
        with zipfile.ZipFile(data_path , mode="w") as archive:
            archive.writestr("data", serialized_data)

    for file_name, dict_ in params.items():
        path = os.path.join(save_path, f"file_name.pnz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_jax_params(save_path + "/" + file_name + ".npz", dict_)


def save_to_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase], obj: Any, verbose: int = 0) -> None:
    with open_path(path, "w", verbose=verbose, suffix="pkl") as file_handler:
        pickle.dump(obj, file_handler, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_zip_file(
    load_path: Union[str, pathlib.Path, io.BufferedIOBase],
    load_data: bool = True,
    custom_objects: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    print_system_info: bool = False,
) -> JnpDict:
    """Load model data from a .zip archive,"""
    namelist = os.listdir(load_path)
     
    params, data = {}, None
    for file_path in namelist:
        if "npz" in file_path:
            params[file_path[0:-4]] = load_jax_params(os.path.join(load_path, file_path))
        if "data" in file_path:
            data_path = open_path(os.path.join(load_path, file_path), "r", verbose=verbose)
            with zipfile.ZipFile(data_path) as archive:
                json_data = archive.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects)

    return data, params
      

def load_from_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase], verbose: int = 0) -> Any:
    with open_path(path, "r", verbose=verbose, suffix="pkl") as file_handler:
        return pickle.load(file_handler)


def save_jax_params(path: str, params: hk.Params) -> None:
    with open(path, 'wb') as fp:
        return pickle.dump(params, fp)

def load_jax_params(path: str) -> hk.Params:
    with open(path, 'rb') as fp: 
        return pickle.load(fp)

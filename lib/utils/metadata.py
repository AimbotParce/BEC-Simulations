import inspect
import json

import jax.numpy as jnp


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def toJSON(obj):
    res = None
    if inspect.isfunction(obj) or inspect.ismodule(obj):
        return None
    elif isinstance(obj, (list, tuple, set)):
        res = []
        for item in obj:
            res.append(toJSON(item))
    elif isinstance(obj, dict):
        res = {}
        for key, value in obj.items():
            res[key] = toJSON(value)
    elif isinstance(obj, jnp.ndarray):
        res = obj.tolist()
    elif is_jsonable(obj):
        res = obj
    else:
        res = str(obj)

    return res

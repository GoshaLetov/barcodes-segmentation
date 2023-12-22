import importlib
import typing


def load_object(object_path: str) -> typing.Any:
    object_path_list = object_path.rsplit(sep='.', maxsplit=1)

    if len(object_path_list) == 2:
        module_name, object_name = object_path_list
    else:
        raise AttributeError(
            f'Parameter `object_path` value should match `package.object` pattern. Got {object_path}',
        )

    module = importlib.import_module(module_name)

    try:
        return getattr(module, object_name)
    except AttributeError:
        raise AttributeError(
            'Object `{object_name}` cannot be loaded from `{module_name}`.'.format(
                object_name=object_name,
                module_name=module_name,
            ),
        )

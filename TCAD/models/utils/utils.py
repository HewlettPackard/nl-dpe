def get_module_device_and_dtype(module):
    '''
        Get the device and type of a module, all layers in the module must have the same device and dtype.
    '''
    parameters = list(module.parameters())
    if len(parameters) == 0:
        raise ValueError(
            f"Module {module.__class__.__qualname__} has no parameters, thus dtype or device can not "
            f"be inferred"
        )

    devices = [p.device for p in parameters]
    types = [p.dtype for p in parameters]
    if not all(d == devices[0] for d in devices):
        raise ValueError(
            f"Module {module.__class__.__qualname__} has different parameters on different device"
        )

    if not all(t == types[0] for t in types):
        raise ValueError(
            f"Module {module.__class__.__qualname__} has parameters on different types"
        )
    return devices[0], types[0]


def _parent_name(target):
    '''
        Turn 'bar.foo.bar' into ['bar.foo', 'bar'].
    '''
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


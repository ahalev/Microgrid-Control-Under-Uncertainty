from collections import UserDict


class Namespacify(UserDict):
    def __init__(self, in_dict, name=''):
        self.name = name

        for key in in_dict.keys():
            if isinstance(in_dict[key], dict):
                in_dict[key] = Namespacify(in_dict[key], name=key)

        super().__init__(in_dict)

    def update(self, *args, **kwargs):
        return nested_dict_update(self, *args, nest_namespacify=True, **kwargs)

    def pprint(self, indent=0):
        print("{}{}:".format(' ' * indent, self.name))

        indent += 4

        for k, v in self.items():
            if k == "name":
                continue
            if isinstance(v, Namespacify):
                v.pprint(indent)
            else:
                print("{}{}: {}".format(' ' * indent, k, v))

    def __dir__(self):
        rv = set(super().__dir__())
        rv = rv | set(self.keys())
        return sorted(rv)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __xor__(self, other):
        diff = {}
        for k, v in self.items():
            assert k in other
            if v != other[k]:
                if isinstance(v, Namespacify):
                    diff[k] = v.__xor__(other[k])
                else:
                    diff[k] = v

        return Namespacify(diff, name=self.name)


def nested_dict_update(nested_dict, *args, nest_namespacify=False, **kwargs):
    if args:
        if len(args) != 1 or not isinstance(args[0], (dict, UserDict)):
            raise TypeError('Invalid arguments')
        elif kwargs:
            raise TypeError('Cannot pass both args and kwargs.')

        d = args[0]
    else:
        d = kwargs

    for k, v in d.items():
        if isinstance(v, (dict, UserDict)):
            if k in nested_dict:
                nested_dict[k].update(v)
            else:
                nested_dict[k] = Namespacify(v, name=k) if nest_namespacify else v
        else:
            nested_dict[k] = v

    return nested_dict

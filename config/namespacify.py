from collections import UserDict


class Namespacify(UserDict):
    def __init__(self, name, in_dict):
        self.name = name

        for key in in_dict.keys():
            if isinstance(in_dict[key], dict):
                in_dict[key] = Namespacify(key, in_dict[key])

        super().__init__(in_dict)

    def update(self, *args, **kwargs):
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
                if k in self:
                    self[k].update(v)
                else:
                    self[k] = Namespacify(k, v)
            else:
                super().update({k: v})

        return self

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

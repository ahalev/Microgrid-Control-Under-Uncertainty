from collections import UserDict


class Namespacify(UserDict):
    def __init__(self, name, in_dict):
        self.name = name

        for key in in_dict.keys():
            if isinstance(in_dict[key], dict):
                in_dict[key] = Namespacify(key, in_dict[key])

        super().__init__(in_dict)
        self.__dict__.update(in_dict)

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

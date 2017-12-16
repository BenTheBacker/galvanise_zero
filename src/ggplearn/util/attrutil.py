''' XXX move out of this repo '''

import sys

import attr
import json


class AttrDict(dict):
    def __init__(self, *args, **kwds):
        dict.__init__(self, *args, **kwds)
        self._enabled = True

    def _add_clz_info(self, name, obj):
        clz = obj.__class__
        k = "%s__clz__" % name
        v = (clz.__module__, clz.__name__)
        self[k] = v

    def _add_clz_info_list(self, name, obj):
        clz = obj.__class__
        k = "%s__clzlist__" % name
        v = (clz.__module__, clz.__name__)
        self[k] = v

    def __setitem__(self, k, v):
        if self._enabled:
            if isinstance(v, (list, tuple)):
                self._do_list(k, v)
                return

            if attr.has(v):
                self._add_clz_info(k, v)

                # this recurses via AttrDict
                as_dict = attr.asdict(v, recurse=False, dict_factory=AttrDict)
                dict.__setitem__(self, k, as_dict)
                return

        dict.__setitem__(self, k, v)

    def _do_list(self, k, v):
        assert isinstance(v, (list, tuple))

        # anything to do?
        if not any(attr.has(i) for i in v):
            dict.__setitem__(self, k, v)
            return

        # check all the same type or not mixed
        if sum(issubclass(type(i), type(v[0])) for i in v) != len(v):
            raise Exception("Bad list %s" % v)

        self._add_clz_info_list(k, v[0])
        as_list = [attr.asdict(i, recurse=False, dict_factory=AttrDict) for i in v]
        dict.__setitem__(self, k, as_list)


def asdict_plus(obj):
    res = AttrDict()
    res['obj'] = obj
    return res


def _fromdict_plus(d):
    # disable or we end up adding back in the ...__clz__ keys
    if isinstance(d, AttrDict):
        d._enabled = False
    for k in d.keys():
        if "__clz__" in k:
            # get clz and remove ...__clz__ key from dict
            mod, clz_name = d.pop(k)
            clz = getattr(sys.modules[mod], clz_name)

            # recurse
            k = k.replace('__clz__', '')
            new_v = _fromdict_plus(d[k])

            # build object and replace in current dict
            d[k] = clz(**new_v)

        if "__clzlist__" in k:
            # get clz and remove ...__clz__ key from dict
            mod, clz_name = d.pop(k)
            clz = getattr(sys.modules[mod], clz_name)

            # recurse
            k = k.replace('__clzlist__', '')
            value = d[k]

            assert isinstance(value, (list, tuple))
            recurse_v = [_fromdict_plus(i) for i in value]

            # build object and replace in current dict
            d[k] = [clz(**i) for i in recurse_v]

    return d


def fromdict_plus(d):
    res = _fromdict_plus(d)
    assert 'obj' in res
    assert len(res) == 1
    return res['obj']


def attr_to_json(obj, **kwds):
    assert attr.has(obj)
    return json.dumps(asdict_plus(obj), **kwds)


def json_to_attr(buf, **kwds):
    d = json.loads(buf, **kwds)
    return fromdict_plus(d)


def pprint(obj):
    assert attr.has(obj)
    from pprint import pprint
    pprint(attr.asdict(obj))

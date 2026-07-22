import torch
import torch.nn.functional as F

try:
    from .dsf_symbols import CHAIN_G, K
except ImportError:
    from dsf_symbols import CHAIN_G, K


class Frame:
    __slots__ = ("h", "q", "r")

    def __init__(self, h, q, r):
        self.h = h
        self.q = q
        self.r = r

    def fork(self, h=None, q=None, r=None):
        return Frame(
            self.h if h is None else h,
            self.q if q is None else q,
            self.r if r is None else r,
        )


def _m(o, n):
    return getattr(o, n)


def _cos(a, b):
    return F.cosine_similarity(a, b, dim=1, eps=1e-6).unsqueeze(1)


def _cat(a, b):
    return torch.cat([a, b, _cos(a, b)], dim=1)


def _run(o, v, seq):
    for n in seq:
        v = _m(o, n)(v)
    return v


def _pre(o, f):
    return f.fork(
        h=_m(o, K.B0)(_m(o, K.A0)(f.h)),
        q=_m(o, K.B1)(f.q * _m(o, K.A1)(f.q)),
    )


def _gate(o, f):
    return f.fork(r=_run(o, _cat(f.h, f.q), CHAIN_G))


def _cross(o, f):
    return f.fork(h=f.h + _m(o, K.X)(f.h, f.q * f.r))


def _post(o, f):
    h = _m(o, K.C)(f.h)
    h = h * _m(o, K.P)(h)
    return f.fork(h=h + _m(o, K.Z)(h))


def flow(o, a, b):
    f = Frame(a, b, None)
    for step in (_pre, _gate, _cross, _post):
        f = step(o, f)
    return f.h

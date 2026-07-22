import torch
import torch.nn.functional as F

try:
    from .dfenet_symbols import R
except ImportError:
    from dfenet_symbols import R


class Pack:
    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a=None, b=None, c=None, d=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def t(self):
        return self.a, self.b, self.c, self.d


def _m(o, n):
    return getattr(o, n)


def _cat(*x):
    return torch.cat(x, 1)


def _up(x, ref=None, size=None):
    return F.interpolate(
        x, size=ref.shape[2:] if ref is not None else size, mode="bilinear"
    )


def _fixed(x, size=384):
    return F.interpolate(x, size=size, mode="bilinear")


def _take(x, c=(32, 32)):
    return torch.split(x, list(c), dim=1)[0]


def _stem(o, x, z):
    return Pack(*_m(o, R.A[0])(x)), Pack(*_m(o, R.A[1])(z))


def _rgb(o, s):
    l1, l2, l3, l4 = s.t()
    m1 = _up(l1, size=R.Z[0])
    m2 = _up(l2, size=R.Z[1])
    m3 = _up(l3, size=R.Z[2])
    m4 = _up(l4, size=R.Z[3])
    mm512, ll512 = _m(o, R.P[3])(m4), _m(o, R.P[3])(l4)
    mm256, ll256 = _m(o, R.P[2])(m3), _m(o, R.P[2])(l3)
    mm128, ll128 = _m(o, R.P[1])(m2), _m(o, R.P[1])(l2)
    mm64, ll64 = _m(o, R.P[0])(m1), _m(o, R.P[0])(l1)
    t1 = _m(o, R.U[3])(m=mm512, l=ll512)
    t2 = _m(o, R.U[2])(m=mm256, l=ll256)
    t3 = _m(o, R.U[1])(m=mm128, l=ll128)
    t4 = _m(o, R.U[0])(m=mm64, l=ll64)
    return Pack(t1, t2, t3, t4)


def _dep(o, s):
    d1, d2, d3, d4 = s.t()
    return Pack(
        _m(o, R.Q[0])(d1),
        _m(o, R.Q[1])(d2),
        _m(o, R.Q[2])(d3),
        _m(o, R.Q[3])(d4),
    )


def _mix(o, a, b):
    t1, t2, t3, t4 = a.t()
    d1, d2, d3, d4 = b.t()
    r4 = _m(o, R.G[0])(t4, d1)
    r3 = _m(o, R.G[1])(t3, d2)
    r2 = _m(o, R.G[2])(t2, d3)
    r1 = _m(o, R.G[3])(t1, d4)
    return Pack(r1, r2, r3, r4)


def _ctx(o, r):
    r1, r2, r3, r4 = r.t()
    edge = o.EDGE(r4, r3, r2, r1)
    edge_att = torch.sigmoid(edge)
    xf_1 = _m(o, R.M[0])(r1)
    r1_up = _up(_m(o, R.W[0])(xf_1), ref=r2)
    r2_con = _m(o, R.C[0])(_cat(r2, r1_up))
    xf_2 = _m(o, R.M[1])(r2_con)
    r2_up = _up(_m(o, R.W[1])(xf_2), ref=r3)
    r3_con = _m(o, R.C[1])(_cat(r3, r2_up))
    xf_3 = _m(o, R.M[2])(r3_con)
    r3_up = _take(_up(_m(o, R.W[2])(xf_3), ref=r4))
    r4_con = _m(o, R.C[2])(_cat(_take(r4), r3_up))
    xf_4 = _m(o, R.M[3])(r4_con)
    return Pack(xf_1, xf_2, xf_3, xf_4), edge_att, r


def _pyr(o, h, r, edge_att):
    xf_1, xf_2, xf_3, xf_4 = h.t()
    _, r2, r3, r4 = r.t()
    xf_12 = _up(xf_1, ref=r2)
    xf_13 = _up(xf_1, ref=r3)
    xf_14 = _up(xf_1, ref=r4)
    xf_22 = _m(o, R.X[0])(xf_2)
    xf_23 = _up(_m(o, R.X[1])(xf_2), ref=r3)
    xf_24 = _up(_m(o, R.X[2])(xf_2), ref=r4)
    xf_33 = _m(o, R.X[3])(xf_3)
    xf_34 = _up(_m(o, R.X[4])(xf_3), ref=r4)
    z4 = _m(o, R.E[0])(xf_4 + xf_14 + xf_24 + xf_34, edge_att)
    z3 = _m(o, R.E[1])(xf_33 + xf_23 + xf_13, edge_att)
    z2 = _m(o, R.E[2])(xf_22 + xf_12, edge_att)
    z1 = _m(o, R.E[3])(xf_1, edge_att)
    return Pack(z1, z2, z3, z4)


def _decode(o, p):
    z1, z2, z3, z4 = p.t()
    y1 = _m(o, R.D[0])(z1)
    y2 = _m(o, R.D[1])(_m(o, R.B[0])(_cat(y1, z2)))
    y3 = _m(o, R.D[2])(_m(o, R.B[1])(_cat(y2, z3)))
    y4 = _m(o, R.D[3])(_m(o, R.B[2])(_cat(y3, z4)))
    return Pack(y1, y2, y3, y4)


def _out(o, y, e):
    y1, y2, y3, y4 = y.t()
    return (
        o.predict_layer_1(_fixed(y4)),
        _fixed(_m(o, R.O[0])(y3)),
        _fixed(_m(o, R.O[1])(y2)),
        _fixed(_m(o, R.O[2])(y1)),
        _fixed(e),
    )


def flow(o, x, depth):
    rgb_raw, dep_raw = _stem(o, x, depth)
    rgb = _rgb(o, rgb_raw)
    dep = _dep(o, dep_raw)
    fused = _mix(o, rgb, dep)
    ctx, edge, fused_ref = _ctx(o, fused)
    maps = _decode(o, _pyr(o, ctx, fused_ref, edge))
    result = _out(o, maps, edge)
    if o.training:
        return result
    return result + (dep_raw.a, dep_raw.c)

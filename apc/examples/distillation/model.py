import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

"""
This is a binary distillation column model, first presented by Moritz Dieh(2001),
and then modified by Rodrigo Lopez-Negrete(2013), David Thierry(2019),
Kuan-Han Lin(2023).
"""


# Overall mass balances
def _de_M_rule(m, i, k):
    if k == 1:
        return m.Mdot[i, 1] == \
               (m.L[i, 2] - m.L[i, 1] - m.V[i, 1])
    elif k == m.Ntray:
        return m.Mdot[i, m.Ntray] == \
               (m.V[i, m.Ntray - 1] - m.L[i, m.Ntray] - m.D[i])
    else:
        return m.Mdot[i, k] == \
               (m.V[i, k - 1] - m.V[i, k] +
                m.L[i, k + 1] - m.L[i, k] +
                m.feed[k])


# Component mass balance
def _de_x_rule(m, i, k):
    if k == 1:
        return m.xdot[i, 1]*m.M[i, 1] == \
                (m.L[i, 2]*(m.x[i, 2] - m.x[i, 1]) -
                 m.V[i, 1]*(m.y[i, 1] - m.x[i, 1]))
    elif k == m.Ntray:
        return m.xdot[i, m.Ntray]*m.M[i, m.Ntray] == \
                (m.V[i, m.Ntray - 1]*(m.y[i, m.Ntray - 1] - m.x[i, m.Ntray]))
    else:
        return m.xdot[i, k]*m.M[i, k] == \
                (m.V[i, k - 1]*(m.y[i, k - 1] - m.x[i, k]) +
                 m.L[i, k + 1]*(m.x[i, k + 1] - m.x[i, k]) -
                 m.V[i, k]*(m.y[i, k] - m.x[i, k]) +
                 m.feed[k]*(m.xf - m.x[i, k]))


# Reflux
def _hrc_rule(m, i):
    return m.D[i]*m.Rec[i] - m.L[i, m.Ntray] == 0


# Energy balance
def _gh_rule(m, i, k):
    if k == 1:
        return m.M[i, 1]*(
                m.xdot[i, 1]*(
                    (m.hlm0 - m.hln0)*m.T[i, 1]**3 +
                    (m.hlma - m.hlna)*m.T[i, 1]**2 +
                    (m.hlmb - m.hlnb)*m.T[i, 1] +
                    (m.hlmc - m.hlnc)
                    ) +
                m.Tdot[i, 1]*(
                    3*m.hln0*m.T[i, 1]**2 +
                    2*m.hlna*m.T[i, 1] +
                    m.hlnb +
                    m.x[i, 1]*(
                        3*(m.hlm0 - m.hln0)*m.T[i, 1]**2 +
                        2*(m.hlma - m.hlna)*m.T[i, 1] +
                        (m.hlmb - m.hlnb)
                        )
                    )
                ) == \
               (m.L[i, 2]*(m.hl[i, 2] - m.hl[i, 1]) -
                m.V[i, 1]*(m.hv[i, 1] - m.hl[i, 1]) +
                m.Qr[i]
                )

    elif k == m.Ntray:
        return m.M[i, m.Ntray]*(
                m.xdot[i, m.Ntray]*(
                    (m.hlm0 - m.hln0)*m.T[i, m.Ntray]**3 +
                    (m.hlma - m.hlna)*m.T[i, m.Ntray]**2 +
                    (m.hlmb - m.hlnb)*m.T[i, m.Ntray] +
                    (m.hlmc - m.hlnc)
                    ) +
                m.Tdot[i, m.Ntray]*(
                    3*m.hln0*m.T[i, m.Ntray]**2 +
                    2*m.hlna*m.T[i, m.Ntray] +
                    m.hlnb +
                    m.x[i, m.Ntray]*(
                        3*(m.hlm0 - m.hln0)*m.T[i, m.Ntray]**2 +
                        2*(m.hlma - m.hlna)*m.T[i, m.Ntray] +
                        (m.hlmb - m.hlnb)
                        )
                    )
                ) == \
               (m.V[i, m.Ntray - 1]*(m.hv[i, m.Ntray - 1] - m.hl[i, m.Ntray]) -
                m.Qc[i]
                )
    else:
        return m.M[i, k]*(
                 m.xdot[i, k]*(
                     (m.hlm0 - m.hln0)*(m.T[i, k]**3) +
                     (m.hlma - m.hlna)*(m.T[i, k]**2) +
                     (m.hlmb - m.hlnb)*m.T[i, k] +
                     (m.hlmc - m.hlnc)
                     ) +
                 m.Tdot[i, k]*(
                     3*m.hln0*(m.T[i, k]**2) +
                     2*m.hlna*m.T[i, k] +
                     m.hlnb +
                     m.x[i, k]*(
                         3*(m.hlm0 - m.hln0)*(m.T[i, k]**2) +
                         2*(m.hlma - m.hlna)*m.T[i, k] +
                         (m.hlmb - m.hlnb)
                         )
                     )
                ) == \
               (m.V[i, k-1]*(m.hv[i, k-1] - m.hl[i, k]) +
                m.L[i, k+1]*(m.hl[i, k+1] - m.hl[i, k]) -
                m.V[i, k]*(m.hv[i, k] - m.hl[i, k]) +
                m.feed[k]*(m.hf - m.hl[i, k])
                )


# Enthalpy for liquid
def _hkl_rule(m, i, k):
    return m.hl[i, k] == \
            m.x[i, k]*(m.hlm0*m.T[i, k]**3 +
                       m.hlma*m.T[i, k]**2 +
                       m.hlmb*m.T[i, k] +
                       m.hlmc
                       ) + \
           (1 - m.x[i, k])*(m.hln0*m.T[i, k]**3 +
                            m.hlna*m.T[i, k]**2 +
                            m.hlnb*m.T[i, k] +
                            m.hlnc
                            )


# Enthalpy for vapor
def _hkv_rule(m, i, k):
    if k < m.Ntray:
        return m.hv[i, k] == \
                m.y[i, k]*(m.hlm0*m.T[i, k]**3 +
                           m.hlma*m.T[i, k]**2 +
                           m.hlmb*m.T[i, k] +
                           m.hlmc +
                           m.r*m.Tkm*pyo.sqrt(1-(m.p[k]/m.Pkm)*(m.Tkm/m.T[i, k])**3)*
                           (m.a-
                            m.b*m.T[i, k]/m.Tkm +
                            m.c1*(m.T[i, k]/m.Tkm)**7 +
                            m.gm*(m.d-
                                  m.l*m.T[i, k]/m.Tkm +
                                  m.f*(m.T[i, k]/m.Tkm)**7)
                            )
                           ) + \
                (1 - m.y[i, k])*(m.hln0*m.T[i, k]**3 +
                                 m.hlna*m.T[i, k]**2 +
                                 m.hlnb*m.T[i, k] +
                                 m.hlnc +
                                 m.r*m.Tkn*
                                 pyo.sqrt(1-(m.p[k]/m.Pkn)*(m.Tkn/m.T[i, k])**3)*
                                 (m.a-
                                  m.b*m.T[i, k]/m.Tkn +
                                  m.c1*(m.T[i, k]/m.Tkn)**7 +
                                  m.gn*(m.d -
                                        m.l*m.T[i, k]/m.Tkn +
                                        m.f*(m.T[i, k]/m.Tkn)**7)
                                  )
                                 )
    else:
        return pyo.Constraint.Skip


# Vapor pressure by Antoine's equation
def _lpself_rule(m, i, k):
    return m.pm[i, k] == pyo.exp(m.CapAm - m.CapBm/(m.T[i, k] + m.CapCm))


# Vapor pressure by Antoine's equation
def _lpn_rule(m, i, k):
    return m.pn[i, k] == pyo.exp(m.CapAn - m.CapBn/(m.T[i, k] + m.CapCn))


# Raoult's law
def _dp_rule(m, i, k):
    return m.p[k] == m.x[i, k]*m.pm[i, k] + (1 - m.x[i, k])*m.pn[i, k]


# Derivative of T for index reduction
def _lTdot_rule(m, i, k):
    return m.Tdot[i, k]  == \
               -(m.pm[i, k] - m.pn[i, k])*m.xdot[i, k] / \
               (m.x[i, k]*
                pyo.exp(m.CapAm - m.CapBm/(m.T[i, k] + m.CapCm))*
                m.CapBm/(m.T[i, k] + m.CapCm)**2 +
                (1 - m.x[i, k])*
                pyo.exp(m.CapAn - m.CapBn/(m.T[i, k] + m.CapCn))*
                m.CapBn/(m.T[i, k] + m.CapCn)**2
                )


# Summation equation with tray efficiency(alpha)
def _gy_rule(m, i, k):
    if k == 1:
        return m.p[1]*m.y[i, 1] == m.x[i, 1]*m.pm[i, 1]
    elif k == m.Ntray:
        return pyo.Constraint.Skip
    else:
        return m.y[i, k] == \
                   m.alpha[k]*m.x[i, k]*m.pm[i, k] / m.p[k] + \
                   (1 - m.alpha[k])*m.y[i, k - 1]


# Definition of liquid volume holdup
def _dMV_rule(m, i, k):
    if k == 1:
        return m.Mv[i, 1] == m.Vm[i, 1]*m.M[i, 1]
    elif k == m.Ntray:
        return m.Mv[i, m.Ntray] == m.Vm[i, m.Ntray]*m.M[i, m.Ntray]
    else:
        return m.Mv[i, k] == m.Vm[i, k]*m.M[i, k]


# Definition of molar volume
def _dvself_rule(m, i, k):
    return m.Vm[i, k] == m.x[i, k]*(
            (1/2288)*0.2685**(1 + (1 - m.T[i, k]/512.4)**0.2453)
            ) + \
            (1 - m.x[i, k])*(
                (1/1235)*0.27136**(1 + (1 - m.T[i, k]/536.4)**0.24)
                )


# Francis weir equation for liquid flow rate
def _hyd_rule(m, i, k):
    if k == 1:
        return m.L[i, 1]*m.Vm[i, 1] == 0.166*(m.Mv[i, 1] - 8.5)**1.5

    elif k == m.Ntray:
        return m.L[i, m.Ntray]*m.Vm[i, m.Ntray] == \
                0.166*(m.Mv[i, m.Ntray] - 0.17)**1.5

    else:
        return m.L[i, k]*m.Vm[i, k] == 0.166*(m.Mv[i, k] - 0.155)**1.5


# Variable initialization rules
def _p_init(m, k):
    ptray = 9.39e+04
    if k <= m.feedTray-1:
        return _p_init(m, m.feedTray) + m.pstrip*(m.feedTray - k)
    elif m.feedTray-1 < k < m.Ntray:
        return ptray + m.prect*(m.Ntray - k)
    elif k == m.Ntray:
        return ptray


def _M_init(m, i, k):
    if k == 1:
        return 105500.0
    elif 2 <= k <= m.feedTray:
        return 3340.0+(3772.0-3340.0)/(m.feedTray-2)*(k-2)
    else:
        return 2890.0+(4650.0-2890.0)/(m.Ntray-m.feedTray-1)*(k-m.feedTray-1)


def _x_init(m, i, k):
    if 1 <= k <= 16:
        return 0.999*k/m.Ntray
    else:
        return 0.36+(0.98-0.36)/(m.Ntray-m.feedTray-1)*(k-m.feedTray-1)


def _T_init(m, i, k):
    return 336.0 + (370.0 - 336.0)/(m.Ntray-1)*(k-1)


turnpt = 11
def _pm_init(m, i, k):
    if 1 <= k <= turnpt:
        return 316040.0+(204915.0-316040.0)/(turnpt-1)*(k-1)
    elif turnpt < k <= m.feedTray-1:
        return 200560.0+(187130.0-200560.0)/(m.feedTray-turnpt-2)*(k-turnpt-1)
    else:
        return 186500.0+(95280.0-186500.0)/(m.Ntray-m.feedTray)*(k-m.feedTray)


def _pn_init(m, i, k):
    if 1 <= k <= turnpt:
        return 93820.0+(58980.0-93820.0)/(turnpt-1)*(k-1)
    elif turnpt < k <= m.feedTray-1:
        return 57500.0+(52950.0-57500.0)/(m.feedTray-turnpt-2)*(k-turnpt-1)
    else:
        return 52740.0+(23525.0-52740.0)/(m.Ntray-m.feedTray)*(k-m.feedTray)


def _l_init(m, i, k):
    if 2 <= k <= m.feedTray:
        return 83.
    elif m.feedTray+1 <= k <= m.Ntray:
        return 23.
    elif k == 1:
        return 40.0


def _y_init(m, i, k):
    if 1 <= k <= turnpt:
        return 0.064+(0.56-0.064)/(turnpt-1)*(k-1)
    elif turnpt < k <= m.feedTray-1:
        return 0.58+(0.64-0.58)/(m.feedTray-turnpt-2)*(k-turnpt-1)
    else:
        return 0.64+(0.99-0.67)/(m.Ntray-m.feedTray)*(k-m.feedTray)


def _hl_init(m, i, k):
    if 1 <= k <= turnpt:
        return 14971.0+(11185.0-14971.0)/(turnpt-1)*(k-1)
    elif turnpt < k <= m.feedTray-1:
        return 11010.0+(10500.0-11010.0)/(m.feedTray-turnpt-2)*(k-turnpt-1)
    else:
        return 10484.0+(5260-10484.0)/(m.Ntray-m.feedTray)*(k-m.feedTray)


def _Mv_init(m, i, k):
    if k==1:
        return 8.60
    elif 2 <= k <= m.feedTray-1:
        return 0.26
    else:
        return 0.20


def make_model(dynamic=True, horizon=600.0):
    m = pyo.ConcreteModel()
    m.Ntray = pyo.Param(initialize=42)
    m.tray = pyo.Set(initialize=range(1, m.Ntray+1))
    if dynamic:
        m.time = dae.ContinuousSet(bounds=(0, horizon))
    else:
        m.time = pyo.Set(initialize=[0])
    time = m.time

    m.feedTray = pyo.Param(initialize=21)
    m.feed = pyo.Param(
        m.tray,
        initialize=lambda m, k: 57.5294 if k == m.feedTray else 0.0,
        mutable=True,
    )

    # Feed mole fraction
    m.xf = pyo.Param(initialize=0.32, mutable=True)
    # Feed enthalpy
    m.hf = pyo.Param(initialize=9081.3)

    m.hlm0 = pyo.Param(initialize=2.6786e-04)
    m.hlma = pyo.Param(initialize=-0.14779)
    m.hlmb = pyo.Param(initialize=97.4289)
    m.hlmc = pyo.Param(initialize=-2.1045e04)

    m.hln0 = pyo.Param(initialize=4.0449e-04)
    m.hlna = pyo.Param(initialize=-0.1435)
    m.hlnb = pyo.Param(initialize=121.7981)
    m.hlnc = pyo.Param(initialize=-3.0718e04)

    m.r = pyo.Param(initialize=8.3147)
    m.a = pyo.Param(initialize=6.09648)
    m.b = pyo.Param(initialize=1.28862)
    m.c1 = pyo.Param(initialize=1.016)
    m.d = pyo.Param(initialize=15.6875)
    m.l = pyo.Param(initialize=13.4721)
    m.f = pyo.Param(initialize=2.615)

    m.gm = pyo.Param(initialize=0.557)
    m.Tkm = pyo.Param(initialize=512.6)
    m.Pkm = pyo.Param(initialize=8.096e06)

    m.gn = pyo.Param(initialize=0.612)
    m.Tkn = pyo.Param(initialize=536.7)
    m.Pkn = pyo.Param(initialize=5.166e06)

    m.CapAm = pyo.Param(initialize=23.48)
    m.CapBm = pyo.Param(initialize=3626.6)
    m.CapCm = pyo.Param(initialize=-34.29)

    m.CapAn = pyo.Param(initialize=22.437)
    m.CapBn = pyo.Param(initialize=3166.64)
    m.CapCn = pyo.Param(initialize=-80.15)

    m.pstrip = pyo.Param(initialize=250)
    m.prect = pyo.Param(initialize=190)

    m.p = pyo.Param(m.tray, initialize=_p_init)
    m.alpha = pyo.Param(
        m.tray, initialize=lambda m, k: 0.62 if k <= m.feedTray else 0.35
    )


    # Liquid hold-up
    m.M = pyo.Var(m.time, m.tray, initialize=_M_init)

    # Mole-fraction
    m.x = pyo.Var(m.time, m.tray, initialize=_x_init)

    if dynamic:
        m.Mdot = dae.DerivativeVar(m.M, initialize=0.0)
        m.xdot = dae.DerivativeVar(m.x, initialize=0.0)
    else:
        m.Mdot = pyo.Var(m.time, m.tray, initialize=0.0)
        m.xdot = pyo.Var(m.time, m.tray, initialize=0.0)
        m.Mdot[...].fix(0.0)
        m.xdot[...].fix(0.0)


    # Tray temperature
    m.T = pyo.Var(m.time, m.tray, initialize=_T_init)
    m.Tdot = pyo.Var(m.time, m.tray, initialize=1e-05)  # not really a der_var

    # Saturation pressure
    m.pm = pyo.Var(m.time, m.tray, initialize=_pm_init)
    m.pn = pyo.Var(m.time, m.tray, initialize=_pn_init)

    # Vapor mole flowrate
    m.V = pyo.Var(m.time, m.tray, initialize=44.0)

    # Liquid mole flowrate
    m.L = pyo.Var(m.time, m.tray, initialize=_l_init)

    # Vapor mole fraction
    m.y = pyo.Var(m.time, m.tray, initialize=_y_init)

    # Liquid enthalpy
    m.hl = pyo.Var(m.time, m.tray, initialize=_hl_init)

    # Vapor enthalpy
    m.hv = pyo.Var(m.time, m.tray, initialize=5e+04)
    # Condenser heat duty
    m.Qc = pyo.Var(m.time, initialize=1.6e06)
    # Distillate
    m.D = pyo.Var(m.time, initialize=18.33)
    # Molar volume
    m.Vm = pyo.Var(m.time, m.tray, initialize=6e-05)

    # Liquid volume holdup
    m.Mv = pyo.Var(m.time, m.tray, rule=_Mv_init)

    # Reflux ratio
    m.Rec = pyo.Var(m.time, initialize=1.20)
    # Re-boiler heat duty
    m.Qr = pyo.Var(m.time, initialize=1.65E+06)

    m.de_M = pyo.Constraint(m.time, m.tray, rule=_de_M_rule)
    m.de_x = pyo.Constraint(m.time, m.tray, rule=_de_x_rule)

    m.hrc = pyo.Constraint(m.time, rule=_hrc_rule)
    m.gh = pyo.Constraint(m.time, m.tray, rule=_gh_rule)
    m.hkl = pyo.Constraint(m.time, m.tray, rule=_hkl_rule)
    m.hkv = pyo.Constraint(m.time, m.tray, rule=_hkv_rule)
    m.lpself = pyo.Constraint(m.time, m.tray, rule=_lpself_rule)
    m.lpn = pyo.Constraint(m.time, m.tray, rule=_lpn_rule)
    m.dp = pyo.Constraint(m.time, m.tray, rule=_dp_rule)
    m.lTdot = pyo.Constraint(m.time, m.tray, rule=_lTdot_rule)
    m.gy = pyo.Constraint(m.time, m.tray, rule=_gy_rule)
    m.dMV = pyo.Constraint(m.time, m.tray, rule=_dMV_rule)
    m.hyd = pyo.Constraint(m.time, m.tray, rule=_hyd_rule)
    m.dvself = pyo.Constraint(m.time, m.tray, rule=_dvself_rule)

    return m


def create_model_bounds(m):
    state_bounds = {
        "M": (1.0, 1.0e+07),
        "T": (200.0, 500.0),
        "pm": (1.0, 5.0e+07),
        "pn": (1.0, 5.0e+07),
        "L": (0.0, 1.0e+03),
        "V": (0.0, 1.0e+03),
        "x": (0.0, 1.0),
        "y": (0.0, 1.0),
        "hl": (1.0, 1.0e+07),
        "hv": (1.0, 1.0e+07),
        "Qc": (0.0, 1.0e+08),
        "D": (0.0, 1.0e+04),
        "Vm": (0.0, 1.0e+04),
        "Mv": (0.155 + 1e-06, 1e+04),
        "Mv[*, 1]": (8.5 + 1e-06, 1e+04),
        "Mv[*, 42]": (0.17 + 1e-06, 1e+04)
    }
    u_bounds = {"Rec": (0.1, 99.999), "Qr": (0, None)}

    for key, val in {**state_bounds, **u_bounds}.items():
        var = m.find_component(pyo.ComponentUID(key))
        var.setlb(val[0])
        var.setub(val[1])


def initialize_model(m, dynamic=True, ntfe=None):
    if ntfe is not None and not dynamic:
        raise RuntimeError("Cannot provide ntfe to initialize steady model")
    elif dynamic and ntfe is None:
        ntfe = 10
    if dynamic:
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(m, nfe=ntfe, ncp=3, scheme="LAGRANGE-RADAU")

    t0 = m.time.first()

    # Fix inputs
    m.Rec.fix(1.20)
    m.Qr.fix(1.65E+06)

    if dynamic:
        # Fix initial conditions if dynamic
        m.M[t0, :].fix()
        m.x[t0, :].fix()


def create_instance(dynamic=True, horizon=None, ntfe=None):
    if horizon is None and dynamic:
        horizon = 600.
    if ntfe is None and dynamic:
        ntfe = 10
    m = make_model(horizon=horizon, dynamic=dynamic)
    initialize_model(m, ntfe=ntfe, dynamic=dynamic)
    return m


def main():
    # Make sure steady and dynamic models are square, structurally
    # nonsingular models.
    m_steady = create_instance(dynamic=False)
    steady_igraph = IncidenceGraphInterface(m_steady)
    assert len(steady_igraph.variables) == len(steady_igraph.constraints)
    steady_vdmp, steady_cdmp = steady_igraph.dulmage_mendelsohn()
    assert not steady_vdmp.unmatched and not steady_cdmp.unmatched

    m = create_instance(horizon=600.0, ntfe=10)
    igraph = IncidenceGraphInterface(m)
    assert len(igraph.variables) == len(igraph.constraints)
    vdmp, cdmp = igraph.dulmage_mendelsohn()
    assert not vdmp.unmatched and not cdmp.unmatched


if __name__ == "__main__":
    main()
from pyomo.dae.flatten import slice_component_along_sets

from pyomo.common.collections import ComponentMap
from pyomo.core.base.expression import Expression
from pyomo.core.base.param import Param
from pyomo.core.base.set import Set
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.component import Component
from pyomo.core.expr.relational_expr import EqualityExpression

from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.modeling.cost_expressions import (
    get_penalty_from_time_varying_target,
)

'''
All functions are taken from Robert Parker's pyomo branch:contrib-mpc-mhe-mutparam.
Robby and I are working together to publish these in pyomo's main branch.
Kuan-Han Lin 06/22/2023
'''


def get_parameters_from_variables(
    variables,
    time,
    ctype=Param,
):
    n_var = len(variables)
    init_dict = {
        (i, t): var[t].value for i, var in enumerate(variables) for t in time
    }
    var_set, comp = _get_indexed_parameters(
        n_var, time, ctype=ctype, initialize=init_dict
    )
    return var_set, comp


def _get_indexed_parameters(n, time, ctype=Param, initialize=None):
    range_set = Set(initialize=range(n))
    if ctype is Param:
        # Create a mutable parameter
        comp = ctype(range_set, time, mutable=True, initialize=initialize)
    elif ctype is Var:
        # Create a fixed variables
        comp = ctype(range_set, time, initialize=initialize)
        # comp.fix() KHL: cannot fix a Var before construct it
    return range_set, comp


def get_constraint_residual_expression(
    constraints,
    time,
    weight_data=None,
    # TODO: Option for norm (including no norm)
):
    cuids = [
        get_indexed_cuid(con, (time,)) for con in constraints
    ]
    # Here I changed Robby's code and assume the model for time and constraints
    # is the same. Robby put this function under DynamicModelInterface to
    # aviod this.
    constraints = [time.model().find_component(cuid) for cuid in cuids]

    if weight_data is None:
        weight_data = ScalarData(
            ComponentMap((var, 1.0) for con in constraints)
        )
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    for con in constraints:
        if not weight_data.contains_key(con):
            raise KeyError(
                "Tracking weight does not contain a key for"
                " constraint %s" % con
            )
    n_con = len(constraints)
    con_set = Set(initialize=range(n_con))
    resid_expr_list = []
    for con in constraints:
        resid_expr_dict = {}
        for t in time:
            expr = con[t].expr
            if isinstance(expr, EqualityExpression):
                resid_expr_dict[t] = (con[t].body - con[t].upper)
            elif con.upper is None:
                resid_expr_dict[t] = (con[t].lower - con[t].body)
            elif con.lower is None:
                resid_expr_dict[t] = (con[t].body - con[t].upper)
            else:
                raise RuntimeError(
                    "Cannot construct a residual expression from a ranged"
                    " inequality. Error encountered processing the expression"
                    " of constraint %s" % con[t].name
                )
        resid_expr_list.append(resid_expr_dict)
    # NOTE: In KH's implementation, using error vars enforces that constraint
    # residuals are constant throughout a sampling period. Is this necessary?
    # Supposing that it is, we can achieve the same thing by imposing piecewise
    # constant constraints on these expressions.
    weights = [weight_data.get_data_from_key(con) for con in constraints]
    def resid_expr_rule(m, i, t):
        return weights[i]*resid_expr_list[i][t]**2
    resid_expr = Expression(con_set, time, rule=resid_expr_rule)
    return con_set, resid_expr


def get_disturbed_constraint_and_residual_expression(
    time,
    constraints,
    con_set,
    disturb_var,
    weight_data=None,
):
    cuids = [
        get_indexed_cuid(con, (time,)) for con in constraints
    ]
    constraints = [time.model().find_component(cuid) for cuid in cuids]

    if weight_data is None:
        weight_data = ScalarData(
            ComponentMap((var, 1.0) for con in constraints)
        )
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    for con in constraints:
        if not weight_data.contains_key(con):
            raise KeyError(
                "Tracking weight does not contain a key for"
                " constraint %s" % con
            )
    def _disturbed_con_rule(m, i, t):
        con = constraints[i][t]
        if isinstance(con.expr, EqualityExpression):
            return con.body + disturb_var[i, t] == 0.0
        else:
            raise RuntimeError(
                "Cannot construct a disturbed constraint. Error encountered"
                "processing the expression"
                " of constraint %s" % con[t].name
            )
    disturbed_con = Constraint(con_set, time, rule=_disturbed_con_rule)

    weights = [weight_data.get_data_from_key(con) for con in constraints]
    def resid_expr_rule(m, i, t):
        return weights[i]*disturb_var[i, t]**2
    resid_expr = Expression(con_set, time, rule=resid_expr_rule)

    return disturbed_con, resid_expr


def slice_components(time, components):
    if isinstance(components, Component):
        components = (components,)
    slices = []
    for comp in components:
        slices.extend(
            slc for idx, slc in slice_component_along_sets(
                comp, (time,)
            )
        )
    return slices


def get_tracking_cost_from_target_trajectory(
    time_set,
    target_data,
    time=None,
    variables=None,
    weight_data=None,
    context=None,
):
    if time is None:
        # time = self.time
        time = time_set
    if isinstance(target_data, (Var, Param, Expression)):
        if variables is None:
            raise RuntimeError(
                "Variables must be provided if we are using a Param or"
                " Expression as an argument"
            )
        # target_data = TimeSeriesData.from_pyomo_components(
        #     variables,
        #     target_data,
        #     time,
        #     # time_set=self.time,
        #     time_set=time_set,
        #     context=context,
        # )

        def from_pyomo_components(
            keys, values, time, time_set=None, context=None
        ):
            # We assume values is an indexed Variable, Param, or Expression
            # indexed by integer indices into the list of keys (variables)
            # and the time set.
            data = ComponentMap([
                (var, [values[i, t] for t in time]) for i, var in enumerate(keys)
            ])
            return TimeSeriesData(data, time, time_set=time_set, context=context)
        target_data = from_pyomo_components(
            variables,
            target_data,
            time,
            # time_set=self.time,
            time_set=time_set,
            context=context,
        )
    elif not isinstance(target_data, TimeSeriesData):
        # target_data = TimeSeriesData(*target_data, time_set=self.time)
        target_data = TimeSeriesData(*target_data, time_set=time_set)
    if variables is None:
        # Use variables provided by the target trajectory.
        # NOTE: Nondeterministic order in Python < 3.7
        variables = [
            # self.model.find_component(key)
            time_set.model().find_component(key)
            for key in setpoint_data.get_data().keys()
        ]
    else:
        # Variables were provided. These could be anything. Process them
        # to get time-indexed variables on the model.
        variables = [
            # self.model.find_component(
            #     get_indexed_cuid(var, (self.time,))
            # ) for var in variables
            time_set.model().find_component(
                get_indexed_cuid(var, (time_set,))
            ) for var in variables
        ]
    return get_penalty_from_time_varying_target(
        variables,
        time,
        target_data,
        weight_data=weight_data,
    )
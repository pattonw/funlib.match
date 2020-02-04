import pylp
import networkx as nx

import logging
import itertools
from copy import deepcopy
from typing import Hashable, Tuple, List, Dict, Iterable, Optional, Union

from .get_constraints import get_constraints

logger = logging.getLogger(__name__)

Node = Hashable
Edge = Tuple[Node, Node]
Matchable = Union[Node, Edge]


class GraphToTreeMatcher:
    def __init__(
        self,
        graph: nx.Graph,
        tree: nx.DiGraph,
        node_match_costs: Iterable[Tuple[Node, Node, float]],
        edge_match_costs: Iterable[Tuple[Edge, Edge, float]],
        use_gurobi: bool = True,
        create_constraints: bool = True,
    ):
        if isinstance(graph, nx.DiGraph):
            self.undirected_graph = graph.to_undirected()
            # TODO: support graph as nx.DiGraph
            # note: some operations such as graph.neighbors only return successors
            # breaking some constraints.
            self.graph = self.undirected_graph.to_directed(graph)
        elif isinstance(graph, nx.Graph):
            self.undirected_graph = graph
            self.graph = graph.to_directed()
        self.tree = tree

        # TODO: do we even need this constraint? Couldn't we do arbitrary
        # subgraph matching. The only difference between ours and the solution
        # in the thesis we based this off of is matching "chains"
        assert nx.is_directed_acyclic_graph(self.tree), (
            "cannot match an arbitrary source to an arbitrary target. "
            + "target graph should be a DAG."
        )

        self.use_gurobi = use_gurobi

        self.node_match_costs = list(node_match_costs)
        self.edge_match_costs = list(edge_match_costs)

        self.objective = None
        self.constraints = pylp.LinearConstraints()

        self.__create_inidicators()
        if create_constraints:
            self.__create_constraints()
        self.__create_objective()

    def g2ts(self, target: Matchable) -> Dict:
        """
        Get all possible matches from G to T for a given target
        in the form of a dictionary
        """
        return self.g2t_match_indicators.get(target, {})

    def t2gs(self, target: Matchable) -> Dict:
        """
        Get all possible matches from T to G for a given target
        in the form of a dictionary
        """
        return self.t2g_match_indicators.get(target, {})

    def t2g(self, target: Matchable, query: Matchable) -> Optional[int]:
        """
        Get the indicator for a target (in T), query pair (in G). None if it doesn't exist.
        """
        return self.t2gs(target).get(query, None)

    def g2t(self, target: Matchable, query: Matchable) -> Optional[int]:
        """
        Get the indicator for a target (in G), query (in T) pair. None if it doesn't exist.
        """
        return self.t2g(query, target)

    def match(self,) -> Tuple[List[Tuple[Node, Node]], List[Tuple[Edge, Edge]], float]:
        """
        Return a Tuple containing a list of node matching tuples, a list of edge matching
        tuples, and a float indicating the cost of the matching.
        """

        solution = self.solve()
        logger.debug(
            f"Found optimal solution with score: {self._score_solution(solution)}"
        )

        edge_matches = []
        for target in self.graph.edges():
            matched = False
            for match, ind in self.g2ts(target).items():
                if solution[ind] > 0.5:
                    edge_matches.append((target, match))
                    matched = True
            if not matched:
                edge_matches.append((target, None))

        node_matches = []
        for target in self.graph.nodes():
            for match, ind in self.g2ts(target).items():
                if solution[ind] > 0.5:
                    node_matches.append((target, match))
                    matched = True

        return node_matches, edge_matches, self._score_solution(solution)

    def solve(self):
        """
        Solves the matching problem and returns a solution if possible.
        If problem is impossible, raises a ValueError
        """
        logger.debug(f"Creating Solver!")
        if self.use_gurobi:
            solver = pylp.create_linear_solver(pylp.Preference.Gurobi)
            # set num threads sometimes causes an error. See issue #5 on
            # github.com/funkey/pylp
            # solver.set_num_threads(1)
        else:
            solver = pylp.create_linear_solver(pylp.Preference.Scip)
            # don't set num threads. It leads to a core dump
        solver.initialize(self.num_variables, pylp.VariableType.Binary)
        solver.set_timeout(120)

        solver.set_objective(self.objective)

        logger.debug(f"Starting Solve!")

        solver.set_constraints(self.constraints)
        solution, message = solver.solve()
        logger.debug(f"Finished solving!, got message ({message})")
        if "NOT" in message:
            raise ValueError(message)

        return solution

    def _score_solution(self, solution) -> float:
        """
        Get the total cost of a particular solution
        """
        total = 0
        for target in itertools.chain(self.graph.nodes(), self.graph.edges()):
            for l, i in self.g2ts(target).items():
                if solution[i] > 0.5:
                    total += self.match_indicator_costs[i]
        return total

    def __create_inidicators(self):
        """
        Creates binary indicators:
        For all i,j in V(G), V(T), there exists an x_{i,j} s.t.
            x_{i,j} = 1 if node i matches to node j, else 0
        For all kl,mn in E(G), E(T), there exists an y_{kl, mn} s.t. 
            y_{kl,mn} = 1 if edge kl matches to edge mn, else 0
        """

        self.num_variables = 0

        self.g2t_match_indicators = {}
        self.t2g_match_indicators = {}
        self.match_indicator_costs = []
        self.num_variables = len(self.node_match_costs) + len(self.edge_match_costs)

        for i, (source, target, cost) in enumerate(
            itertools.chain(self.node_match_costs, self.edge_match_costs)
        ):

            u2t_node_indicators = self.g2t_match_indicators.setdefault(source, {})
            u2t_node_indicators[target] = i
            v2g_node_indicators = self.t2g_match_indicators.setdefault(target, {})
            v2g_node_indicators[source] = i

            self.match_indicator_costs.append(cost)

    def add_constraint(self, constraints, relation, value, key):
        print(key)
        for constraint in constraints:
            c = pylp.LinearConstraint()
            for match_indicator, weight in constraint:
                c.set_coefficient(match_indicator, weight)
            c.set_relation(getattr(pylp.Relation, relation))
            c.set_value(value)
            self.constraints.add(c)

    def __create_constraints(self):
        """
        constraints based on the implementation described here:
        https://hal.archives-ouvertes.fr/hal-00726076/document
        pg 11
        Pierre Le Bodic, Pierre Héroux, Sébastien Adam, Yves Lecourtier. An integer
        linear program for substitution-tolerant subgraph isomorphism and its use for
        symbol spotting in technical drawings.
        Pattern Recognition, Elsevier, 2012, 45 (12), pp.4214-4224. ffhal-00726076
        """

        all_constraints = get_constraints(
            self.graph, self.tree, self.node_match_costs, self.edge_match_costs
        )

        self.constraints = pylp.LinearConstraints()
        for key, (constraints, relation, value) in all_constraints.items():
            logger.debug(f"{key}: {len(constraints)}")
            self.add_constraint(constraints, relation, value, key)

    def __create_objective(self):

        self.objective = pylp.LinearObjective(self.num_variables)

        for i, c in enumerate(self.match_indicator_costs):
            self.objective.set_coefficient(i, c)

    def enforce_expected_assignments(
        self, expected_assignments: Iterable[Tuple[Matchable, Matchable]]
    ):
        """
        Force a specific set of matchings to be made in any solution.
        """
        expected_assignment_constraint = pylp.LinearConstraint()
        failed_matchings = []
        num_assignments = 0
        for match_a, match_b in expected_assignments:
            match_ind = self.g2t(match_a, match_b)
            logger.debug(
                f"Enforcing match {(match_a, match_b)} with match index {match_ind}"
            )
            if match_ind is None:
                # This is most likely an edge matching since edges don't have an
                # explicit indicator for None matchings
                if match_a in self.g2t_match_indicators and match_b is None:
                    # if there is no None match available, then this must be an edge matching
                    # enfore unmatched edge
                    for match, ind in self.g2ts(match_a).items():
                        expected_assignment_constraint.set_coefficient(ind, -1)
                elif match_a in self.g2t_match_indicators and match_b is not None:
                    logger.debug(f"Matching {(match_a, match_b)} is impossible!")
                    failed_matchings.append((match_a, match_b))
                elif match_a not in self.g2t_match_indicators:
                    logger.debug(
                        f"Attempting to enforce matching for {match_a} which has no possible matches!"
                    )
                    failed_matchings.append((match_a, match_b))
            else:
                expected_assignment_constraint.set_coefficient(match_ind, 1)
                num_assignments += 1
        expected_assignment_constraint.set_relation(pylp.Relation.Equal)
        expected_assignment_constraint.set_value(num_assignments)
        self.constraints.add(expected_assignment_constraint)
        logger.warning(
            f"{len(failed_matchings)} of {len(expected_assignments)} were not enforcable!"
        )


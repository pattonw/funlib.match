import pylp
import networkx as nx

import logging
import itertools
from copy import deepcopy
from typing import Hashable, Tuple, List, Dict, Iterable, Optional, Union

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

        self.node_match_costs = node_match_costs
        self.edge_match_costs = edge_match_costs

        self.objective = None
        self.constraints = None

        self.__create_inidicators()
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

        no_matches = [(graph_n, None, 0) for graph_n in self.graph.nodes()]

        self.g2t_match_indicators = {}
        self.t2g_match_indicators = {}
        self.match_indicator_costs = []
        self.num_variables = (
            len(no_matches) + len(self.node_match_costs) + len(self.edge_match_costs)
        )

        for i, (u, v, cost) in enumerate(
            itertools.chain(self.node_match_costs, self.edge_match_costs, no_matches)
        ):

            u2t_node_indicators = self.g2t_match_indicators.setdefault(u, {})
            u2t_node_indicators[v] = i
            v2g_node_indicators = self.t2g_match_indicators.setdefault(v, {})
            v2g_node_indicators[u] = i

            self.match_indicator_costs.append(cost)

    def __add_one_to_one_t2g_node_constraint(self):
        """
        (2b) Every node in T must match to exactly one node in G
        """
        for tree_n in self.tree.nodes():
            constraint = pylp.LinearConstraint()
            for match_indicator in self.t2gs(tree_n).values():
                constraint.set_coefficient(match_indicator, 1)
            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(1)
            self.constraints.add(constraint)

    def __add_many_to_one_t2g_edge_constraint(self):
        """
        modified
        (2c): Every edge in T must match to **at least one** edge in G

        For strict subgraph isomorphisms, this would be exactly 1,
            but we want to allow "chains"
        """

        for tree_e in self.tree.edges():

            constraint = pylp.LinearConstraint()
            for match_indicator in self.t2gs(tree_e).values():
                constraint.set_coefficient(match_indicator, -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(-1)
            self.constraints.add(constraint)

    def __add_g_node_must_match_constraint(self):
        """
        modified
        (2d) Every node in G must match

        A possible match for every node in G is None
        """
        for graph_n in self.graph.nodes():
            logger.debug(
                f"2d) {graph_n} must match to one of {list(self.g2ts(graph_n).keys())}"
            )
            constraint = pylp.LinearConstraint()
            for match_indicator in self.g2ts(graph_n).values():
                constraint.set_coefficient(match_indicator, 1)
            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(1)
            self.constraints.add(constraint)

    def __add_branch_topology_constraint(self):
        """
        For every pair of matched nodes j, k in V(G), V(T),
            (2e) Every edge targeting k in T must match to an edge targeting j in G
            (2f) Every edge originating from k in T must match to an edge originating from j in G
        """
        for graph_n in self.graph.nodes():
            tree_nodes = [x for x in self.g2ts(graph_n).keys() if x is not None]
            for tree_n in tree_nodes:
                node_match_indicator = self.g2t(graph_n, tree_n)

                # (2e)
                for tree_out_e in self.tree.out_edges(tree_n):
                    logger.debug(
                        f"2e) Out edge of {graph_n} must match to {tree_out_e} if {graph_n} matches {tree_n}"
                    )

                    constraint = pylp.LinearConstraint()
                    constraint.set_coefficient(node_match_indicator, 1)
                    for graph_out_e in self.graph.out_edges(graph_n):
                        g2t_e_indicator = self.g2t(graph_out_e, tree_out_e)
                        if g2t_e_indicator is not None:
                            constraint.set_coefficient(g2t_e_indicator, -1)
                    # Note we use LessEqual so that if node_match_indicator = 0, which will often be
                    # the case, we still allow matching.
                    constraint.set_relation(pylp.Relation.LessEqual)
                    constraint.set_value(0)
                    self.constraints.add(constraint)

                # (2f)
                for tree_in_e in self.tree.in_edges(tree_n):
                    logger.debug(
                        f"2f) In edge of {graph_n} must match to {tree_in_e} if {graph_n} matches {tree_n}"
                    )

                    constraint = pylp.LinearConstraint()
                    constraint.set_coefficient(node_match_indicator, 1)
                    for graph_in_e in self.graph.in_edges(graph_n):
                        g2t_e_indicator = self.g2t(graph_in_e, tree_in_e)
                        if g2t_e_indicator is not None:
                            constraint.set_coefficient(g2t_e_indicator, -1)
                    constraint.set_relation(pylp.Relation.LessEqual)
                    constraint.set_value(0)
                    self.constraints.add(constraint)

    def __add_solid_chain_constraint(self):
        """
        EXTRA CONSTRAINT, NOT FROM PAPER:
        The previous constraints are enough to cover isomorphisms, but we need to model chains
        in G representing edges in S
        Consider:
        
        T:      A---------------------------------------B
        G:      a--b--c---------------------------d--e--f

        Under previous constraints, matching AB to ab and ef would be sufficient.

        Consider a node i in G and an edge kl in T. 
        If i matches to k, then we know
        there should be exactly 1 edge originating from i that matches to kl, and 0
        edges targeting i should match to kl.
        Similarly if i matches to l, then there should be exactly 1 edge targeting i
        that matches to kl, and 0 edges originating from i should match to kl.
        Finally, if i matches to neither k nor l, then to maintain a chain, we must
        have exactly 1 edge originating from i match to kl *and* exactly 1 edge
        targeting i to match to kl.
        """

        for graph_n in self.graph.nodes():

            possible_edges = set(
                tree_e
                for tree_n in self.g2ts(graph_n)
                for tree_e in self.tree.edges(tree_n)
            )
            for tree_e in possible_edges:
                equality_constraint = pylp.LinearConstraint()
                for graph_in_e in self.graph.in_edges(graph_n):
                    indicator = self.g2t(graph_in_e, tree_e)
                    if indicator is not None:
                        # -1 if an in edge matches
                        equality_constraint.set_coefficient(indicator, -1)
                for graph_out_e in self.graph.out_edges(graph_n):
                    indicator = self.g2t(graph_out_e, tree_e)
                    if indicator is not None:
                        # +1 if an out edge matches
                        equality_constraint.set_coefficient(indicator, 1)

                for tree_n, tree_n_indicator in self.g2ts(graph_n).items():
                    # tree_e must be an out edge
                    if tree_n == tree_e[0]:
                        equality_constraint.set_coefficient(tree_n_indicator, -1)
                    # tree_e must be an in edge
                    if tree_n == tree_e[1]:
                        equality_constraint.set_coefficient(tree_n_indicator, 1)

                equality_constraint.set_relation(pylp.Relation.Equal)
                equality_constraint.set_value(0)
                self.constraints.add(equality_constraint)

    def __add_degree_constraint(self):
        """
        Now that we allow chains, it is possible that multiple
        chains pass through the same node, or a chain might pass through
        a branch point. To avoid this we add a degree constraint.

        given x_ij for all i in V(G) and j in V(T)
        given y_ab_cd for all (a, b) in E(G) and (c, d) in E(T)
        let degree(None) = 2
        For every node i in V(G)
          let N = SUM(degree(c)*x_ic) for all c in V(T) Union None
          let y = SUM(y_ai_cd) + SUM(y_ia_cd)
              for all a adjacent to i, and all (c,d) in E(T)
          y - N <= 0
        """
        for graph_n in self.graph.nodes():
            degree_constraint = pylp.LinearConstraint()
            for tree_n, tree_n_indicator in self.g2ts(graph_n).items():
                d = 2 if tree_n is None else self.tree.degree(tree_n)
                degree_constraint.set_coefficient(tree_n_indicator, -d)
            # TODO: support graph being a nx.DiGraph
            for neighbor in self.graph.neighbors(graph_n):
                for adj_edge_indicator in self.g2ts((graph_n, neighbor)).values():
                    degree_constraint.set_coefficient(adj_edge_indicator, 1)
                for adj_edge_indicator in self.g2ts((neighbor, graph_n)).values():
                    degree_constraint.set_coefficient(adj_edge_indicator, 1)

            degree_constraint.set_relation(pylp.Relation.LessEqual)
            degree_constraint.set_value(0)
            self.constraints.add(degree_constraint)

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

        self.constraints = pylp.LinearConstraints()

        self.__add_one_to_one_t2g_node_constraint()

        self.__add_many_to_one_t2g_edge_constraint()

        self.__add_g_node_must_match_constraint()

        self.__add_branch_topology_constraint()

        self.__add_solid_chain_constraint()

        self.__add_degree_constraint()

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
        num_assignments = 0
        for match_a, match_b in expected_assignments:
            expected_assignment_constraint.set_coefficient(
                self.g2t(match_a, match_b), 1
            )
            num_assignments += 1
        expected_assignment_constraint.set_relation(pylp.Relation.Equal)
        expected_assignment_constraint.set_value(num_assignments)
        self.constraints.add(expected_assignment_constraint)

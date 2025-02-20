from flow.networks import Network
from flow.core.params import InitialConfig, TrafficLightParams
import random
import numpy as np
from numpy import random as np_random

SCALING = 1

ADDITIONAL_NET_PARAMS = {
    "length": 200,
    "lanes": 2,
    "speed_limit": 30,
}

class Intersection(Network):
    """Definizione di una rete personalizzata per SUMO."""

    def __init__(self, name, vehicles, net_params, 
                 initial_config=InitialConfig(), traffic_lights=TrafficLightParams()):
        """Inizializza la rete."""

        self.nodes_dict = dict()

        super().__init__(name, vehicles, net_params, 
                         initial_config, traffic_lights)

    def specify_nodes(self, net_params):
        """Definisci i nodi della rete."""
        nodes = [
            {"id": "J0", "x": 0.0, "y": 200.0},
            {"id": "J1", "x": 0.0, "y": 0.0},
            {"id": "J2", "x": -200.0, "y": 0.0},
            {"id": "J3", "x": 200.0, "y": 0.0},
            {"id": "J4", "x": 0.0, "y": -200.0},
        ]

        for node in nodes:
            self.nodes_dict[node['id']] = np.array([node['x'] * SCALING, 
                                             node['y'] * SCALING])
            
        for node in nodes:
            node['x'] = node['x'] * SCALING
            node['y'] = node['y'] * SCALING
            
        return nodes

    def specify_edges(self, net_params):
        """Definisci gli archi della rete."""
        edges = [
            {"id": "L0", "from": "J0", "to": "J1", "numLanes": 2, "length": 200, 'type': 'edgeType'},
            {"id": "L1", "from": "J1", "to": "J0", "numLanes": 2, "length": 200, 'type': 'edgeType'},
            {"id": "L2", "from": "J3", "to": "J1", "numLanes": 2, "length": 200, 'type': 'edgeType'},
            {"id": "L3", "from": "J1", "to": "J3", "numLanes": 2, "length": 200, 'type': 'edgeType'},
            {"id": "L4", "from": "J4", "to": "J1", "numLanes": 2, "length": 200, 'type': 'edgeType'},
            {"id": "L5", "from": "J1", "to": "J4", "numLanes": 2, "length": 200, 'type': 'edgeType'},
            {"id": "L6", "from": "J2", "to": "J1", "numLanes": 2, "length": 200, 'type': 'edgeType'},
            {"id": "L7", "from": "J1", "to": "J2", "numLanes": 2, "length": 200, 'type': 'edgeType'}
        ]

        for edge in edges:
            if edge["to"] not in self.nodes_dict or edge["from"] not in self.nodes_dict:
                raise KeyError(f"Edge {edge['id']} refers to unknown node '{edge['to']}' or '{edge['from']}'")
            
            edge['length'] = np.linalg.norm(
                    self.nodes_dict[edge['to']] -
                    self.nodes_dict[edge['from']])
    
        return edges

    
    def specify_routes(self, net_params):
        """Definisci le rotte per tutti gli archi e rotte multiple."""
        edges = [
            "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"
        ]
        
        # Rotte singole per ogni arco
        rts = {edge: [edge] for edge in edges}
        
        # Aggiungi rotte multiple
        rts.update({
            "route_WE": ["L6", "L3"],
            "route_WN": ["L6", "L1"],
            "route_WS": ["L6", "L5"],
            "route_EW": ["L2", "L7"],
            "route_ES": ["L2", "L5"],
            "route_EN": ["L2", "L1"],
            "route_NE": ["L0", "L3"],
            "route_NS": ["L0", "L5"],
            "route_NW": ["L0", "L7"],
            "route_SW": ["L4", "L7"],
            "route_SN": ["L4", "L1"],
            "route_SE": ["L4", "L3"],
        })
        return rts
    
    def specify_connections(self, net_params):
        """Define connections based on the provided data."""
        connections = {}
        connections['J1'] = [
            {'from': 'L0', 'to': 'L7', 'fromLane': 0, 'toLane': 0},
            {'from': 'L0', 'to': 'L5', 'fromLane': 0, 'toLane': 0},
            {'from': 'L0', 'to': 'L3', 'fromLane': 1, 'toLane': 1},
            {'from': 'L2', 'to': 'L1', 'fromLane': 0, 'toLane': 0},
            {'from': 'L2', 'to': 'L7', 'fromLane': 0, 'toLane': 0},
            {'from': 'L2', 'to': 'L7', 'fromLane': 1, 'toLane': 1},
            {'from': 'L2', 'to': 'L5', 'fromLane': 1, 'toLane': 1},
            {'from': 'L2', 'to': 'L5', 'fromLane': 1, 'toLane': 1},
            {'from': 'L4', 'to': 'L3', 'fromLane': 0, 'toLane': 0},
            {'from': 'L4', 'to': 'L1', 'fromLane': 0, 'toLane': 0},
            {'from': 'L4', 'to': 'L7', 'fromLane': 1, 'toLane': 1},
            {'from': 'L6', 'to': 'L1', 'fromLane': 1, 'toLane': 0},
            {'from': 'L6', 'to': 'L5', 'fromLane': 0, 'toLane': 0},
            {'from': 'L6', 'to': 'L3', 'fromLane': 0, 'toLane': 0},
            ]
        return connections
    
    def specify_types(self, net_params):
        """Define default edge types."""
        return [{"id": "edgeType", "speed": 30}]

    def specify_edge_starts(self):
        """Define randomized edge starts for the simulation."""
        length = 0
        edge_starts = []
        
        # Ottieni gli archi e randomizza l'ordine
        edges = self.specify_edges(None)
        random.shuffle(edges)
        
        for edge in edges:
            offset = random.uniform(-0.5, 0.5)
            edge_starts.append((edge["id"], max(0, length + offset)))
            length += edge["length"]
        
        return edge_starts
    
    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """
        Genera posizioni iniziali casuali per i veicoli all'interno della rete.

        Parameters
        ----------
        net_params : flow.core.params.NetParams
            Parametri specifici della rete.
        initial_config : flow.core.params.InitialConfig
            Configurazione iniziale della rete.
        num_vehicles : int
            Numero di veicoli da posizionare.

        Returns
        -------
        start_positions : list of tuple (str, float)
            Lista di posizioni iniziali dei veicoli [(edge_id, position), ...].
        start_lanes : list of int
            Lista di corsie iniziali dei veicoli.
        start_speeds : list of float
            Lista di velocità iniziali dei veicoli.
        """
        return net_params.additional_params["start_positions"], net_params.additional_params["start_lanes"]
            

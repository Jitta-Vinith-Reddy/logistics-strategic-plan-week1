"""
Route Optimization (Google OR-Tools) - pseudocode/skeleton
FreshCart Supplies - Logistics Data Analysis (Week 1)
"""
from ortools.constraint_solver import routing_enums, pywrapcp

distance_matrix = compute_distance_matrix(dc_location, store_locations)
manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, 0)
routing = pywrapcp.RoutingModel(manager)


def distance_cb(i, j):
    return distance_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]


routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_cb))
routing.AddDimensionWithVehicleCapacity(
    routing.RegisterTransitCallback(distance_cb), 0, vehicle_capacities, True, "Capacity"
)

params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = routing_enums.FirstSolutionStrategy.PATH_CHEAPEST_ARC
solution = routing.SolveWithParameters(params)

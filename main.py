# from src.simulation import Simulation

# def main():
#     sim = Simulation()
#     sim.create_cars(10)
#     sim.create_servers(10)
#     sim.load_tasks('data/task_data.csv')
    
#     print("Running the simulation...")
#     sim.offload_tasks()

#     print("Locations of cars and servers")
#     sim.plot_locations()

#     print("Plotting the results...")
#     sim.plot_results()

# if __name__ == "__main__":
#     main()

from src.simulation import Simulation
import random

def main():
    sim = Simulation()
    sim.create_cars(10)
    sim.create_servers(10)
    # Generate 50-200 tasks per car instead of loading from CSV
    for car in sim.cars:
        car.generate_tasks(random.randint(50, 200))
    
    print("=" * 60)
    print("RUNNING COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 60)
    
    print("\n1. Running DQL (Deep Q-Learning) Algorithm...")
    sim.offload_tasks()  # Runs the original DQL algorithm

    print("\n2. Plotting locations and DQL results...")
    sim.plot_locations()
    sim.plot_results()

    print("\n3. Running Wu et al. (2024) Baseline Algorithm...")
    sim.wu_baseline_offload_tasks()
    wu_results = {
        "successful": sim.wu_successful_offloads,
        "failed": sim.wu_failed_offloads,
        "distances": sim.wu_distances
    }

    print("\n4. Running WFIT (Worst Fit) Algorithm...")
    worst_results = sim.worst_offload_tasks()

    print("\n5. Running Random Algorithm...")
    sim.random_offload_tasks()
    random_results = {
        "successful": sim.random_successful_offloads,
        "failed": sim.random_failed_offloads,
        "distances": sim.random_distances
    }

    print("\n6. Generating comprehensive comparison analysis...")
    original_results = {
        "successful_offloads": sim.successful_offloads,
        "failed_offloads": sim.failed_offloads,
        "distances": sim.distances
    }
    
    sim.plot_comparison(
        original_results=original_results,
        worst_results=worst_results,
        wu_results=wu_results,
        random_results=random_results
    )
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("Check the 'outputs/' directory for generated charts and analysis.")
    print("=" * 60)

if __name__ == "__main__":
    main()

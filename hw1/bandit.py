import numpy as np
import matplotlib.pyplot as plt

# Set parameters
num_arms = 10         # 10-armed bandit
alpha = 0.1           # Constant step-size for one method
time_steps = 10000    # Number of time steps
runs = 3000         # Number of independent runs
sigma_walk = 0.01     # Standard deviation for random walk

# Epsilon values to test
epsilons = [0, 0.01, 0.1]  # Fully greedy, ε=0.1, ε=0.01

# Function to run the experiment
# epsilon: value to use for the epsilon-greedy action selection
# sample_avg: Boolean representing if action value is being used
def run_experiment(epsilon, sample_avg=True):
    rewards_over_time = np.zeros(time_steps)  # Track rewards over time
    optimal_action_counts = np.zeros(time_steps) # Track optimal actions
    switch_action_counts = np.zeros(time_steps) # Track action switches

    for run in range(runs):
        if sample_avg and run == 0:
            print(f"Running sample average method... ")
        elif run == 0:
            print(f"Running constant step size method... ")
            
        if run % 100 == 0: 
                print(f"Run {run}/{runs}")

        q_true = np.zeros(num_arms)  # True values start at 0
        q_estimates = np.zeros(num_arms)  # Initial estimates (all 0)
        action_counts = np.zeros(num_arms)  # Count for sample average method

        last_action = None # Use to track most recent action

        for t in range(time_steps):
            # Find optimal action
            optimal_action = np.argmax(q_true)

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(num_arms)  # Explore
            else:
                action = np.argmax(q_estimates)  # Exploit best known action

            # Count action switches
            if last_action is not None and action != last_action:
                switch_action_counts[t] += 1
            last_action = action  # Update last action

            # Check if action is optimal
            if action == optimal_action:
                optimal_action_counts[t] += 1

            # Get reward (true value + noise)
            reward = np.random.normal(q_true[action], 0.5)
            rewards_over_time[t] += reward  # Accumulate reward for averaging

            # Update estimates
            if sample_avg:
                # Sample-Average Method: Q(a) = sum(rewards) / count
                action_counts[action] += 1
                q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]
            else:
                # Constant Step-Size Method: Q(a) = Q(a) + α * (reward - Q(a))
                q_estimates[action] += alpha * (reward - q_estimates[action])

            # Nonstationary environment: update true values randomly
            q_true += np.random.normal(0, sigma_walk, num_arms)

    # Compute % optimal action selection but ignore first 100 steps
    optimal_action_percentage = (optimal_action_counts[100:] / runs) * 100

    # Compute % action switches but ignore first 100 steps
    action_switch_percentage = (switch_action_counts[100:] / runs) * 100

    # Average rewards over runs
    return (rewards_over_time / runs), optimal_action_percentage, action_switch_percentage


def main():
    results_rewards = {}
    results_optimal = {}
    results_switches = {}

    # Run experiments for different epsilon values
    for epsilon in epsilons:
        rewards, optimal_actions, switches = run_experiment(epsilon, sample_avg=True)
        results_rewards[f"Sample Avg"] = rewards
        results_optimal[f"Sample Avg"] = optimal_actions
        results_switches[f"Sample Avg"] = switches

        rewards, optimal_actions, switches = run_experiment(epsilon, sample_avg=False)
        results_rewards[f"Const Step"] = rewards
        results_optimal[f"Const Step"] = optimal_actions
        results_switches[f"Const Step"] = switches


    # Plot Average Reward results
    print("Plotting...")
    plt.figure(figsize=(10, 6))
    for label, rewards in results_rewards.items():
        plt.plot(rewards, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.suptitle("Average Reward vs. Time Step", fontsize=15, y=0.96, fontweight='bold')
    plt.title("(Sigma walk: 0.01, Number of runs: 3000, Action Selection: 0.01-greedy)", fontsize=9)
    plt.legend()
    plt.show()

    # Plot Optimal Action results   
    plt.figure(figsize=(10, 6))
    for label, optimal in results_optimal.items():
        plt.plot(optimal, label=label)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.suptitle("Percentage of Optimal Action vs. Time Step", fontsize=15, y=0.96, fontweight='bold')
    plt.title("(Sigma walk: 0.01, Number of runs: 3000, Action Selection: 0.01-greedy)", fontsize=9)
    plt.legend()
    plt.show()

    # Plot Action Switches results  
    plt.figure(figsize=(10, 6))
    for label, switch in results_switches.items():
        plt.plot(switch, label=label)
    plt.xlabel("Steps")
    plt.ylabel("% Action Switch")
    plt.suptitle("% Action Switch vs. Time Step", fontsize=15, y=0.96, fontweight='bold')
    plt.title("(Sigma walk: 0.01, Number of runs: 3000, Action Selection: 0.1-greedy)", fontsize=9)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
import os
from stable_baselines3 import PPO
from tyche_core.env import TycheTradeEnv

def start_training():
    print("--- Project Tyche: Initiating Aggressive Training ---")
    
    # 1. Create folders for project organization
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 2. Initialize the updated Environment
    # Ensure you have already updated tyche_core/env.py with the 10-feature logic
    try:
        env = TycheTradeEnv()
        print("✅ Environment initialized with Price & Volume features.")
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return

    # 3. Define the AI Brain (PPO)
    # Using a slightly higher learning rate (0.001) to "kickstart" the agent
    # We use 'MlpPolicy' which is a standard Multi-Layer Perceptron neural net
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001, 
        tensorboard_log="./tyche_logs/"
    )

    # 4. Deep Training Session
    # 100,000 steps allows the agent to see the 5-day dataset multiple times
    print("--- Tyche is now entering the market (100,000 steps) ---")
    model.learn(total_timesteps=100000)

    # 5. Save the Progress
    model_path = "models/tyche_v1"
    model.save(model_path)
    print(f"✅ Training Complete! Brain saved at: {model_path}")

if __name__ == "__main__":
    # IMPORTANT: Delete your old models/tyche_v1.zip before running this
    # because the input shape has changed from 5 to 10 features.
    start_training()
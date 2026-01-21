from simworld.communicator.unrealcv import UnrealCV
from simworld.communicator.communicator import Communicator
from simworld.agent.humanoid import Humanoid
from simworld.utils.vector import Vector
from simworld.llm.base_llm import BaseLLM
from simworld.local_planner.local_planner import LocalPlanner
from simworld.llm.a2a_llm import A2ALLM

class Agent:
    def __init__(self, goal):
        self.goal = goal
        self.llm = BaseLLM("gpt-4o")
        self.system_prompt = f"You are an intelligent agent in a 3D world. Your goal is to: {self.goal}."

    def action(self, obs):
        prompt = f"{self.system_prompt}\n You are currently at: {obs}\nWhat is your next action?"
        action = self.llm.generate_text(system_prompt=self.system_prompt, user_prompt=prompt)
        return action

class Environment:
    def __init__(self, comm: Communicator):
        self.comm = comm
        self.agent: Humanoid | None = None
        self.action_planner = None
        self.agent_name: str | None = None
        self.target: Vector | None = None
        self.action_planner_llm = A2ALLM(model_name="gpt-4o-mini")

    def reset(self):
        """Clear the UE scene and (re)spawn the humanoid and target."""
        # Clear spawned objects
        self.comm.clear_env()

        # Blueprint path for the humanoid agent to spawn in the UE level
        agent_bp = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C"

        # Initial spawn position and facing direction for the humanoid (2D)
        spawn_location, spawn_forward = Vector(0, 0), Vector(0, 1)
        self.agent = Humanoid(spawn_location, spawn_forward)
        self.action_planner = LocalPlanner(agent=self.agent, model=self.action_planner_llm, rule_based=False)

        # Spawn the humanoid agent in the Unreal world
        self.comm.spawn_agent(self.agent, name=None, model_path=agent_bp, type="humanoid")

        # Define a target position the agent is encouraged to move toward (example value)
        self.target = Vector(1000, 0)

        # Return initial observation (optional, but RL-style)
        observation = self.comm.get_camera_observation(self.agent.camera_id, "lit")

        return observation

    def step(self, action):
        """Use action planner to execute the given action."""
        # Parse the action text and map it to the action space
        primitive_actions = self.action_planner.parse(action)

        self.action_planner.execute(primitive_actions)

        # Get current location from UE (x, y, z) and convert to 2D Vector
        location = Vector(*self.comm.unrealcv.get_location(self.agent)[:2])

        # Camera observation for RL
        observation = self.comm.get_camera_observation(self.agent.camera_id, "lit")

        # Reward: negative Euclidean distance in 2D plane
        reward = -location.distance(self.target)

        return observation, reward

if __name__ == "__main__":
    # Connect to the running Unreal Engine instance via UnrealCV
    ucv = UnrealCV()
    comm = Communicator(ucv)

    # Create the environment wrapper
    agent = Agent(goal='Go to (1700, -1700) and pick up GEN_BP_Box_1_C.')
    env = Environment(comm)

    obs = env.reset()

    # Roll out a short trajectory
    for _ in range(100):
        action = agent.action(obs)
        obs, reward = env.step(action)
        print(f"obs: {obs}, reward: {reward}")
        # Plug this into your RL loop / logging as needed
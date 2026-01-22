from simworld.communicator.unrealcv import UnrealCV
from simworld.communicator.communicator import Communicator
from simworld.agent.humanoid import Humanoid
from simworld.llm.base_llm import BaseLLM
from simworld.utils.vector import Vector
from simworld.config import Config
from simworld.map.map import Map
from simworld.llm.a2a_llm import A2ALLM
from simworld.local_planner.local_planner import LocalPlanner
from pathlib import Path

class AgentEgo:
    def __init__(self, name:str=None, goal:str=None):
        self.name = name
        self.goal = goal
        self.llm = BaseLLM(model_name="gpt-4o-mini")
        self.system_prompt = f"You are an intelligent agent in a 3D world. Your goal is to: {self.goal}."

    def action(self, obs):
        prompt = f"{self.system_prompt}\n You are currently at: {obs}\nWhat is your next goal?"
        return self.llm.generate_text(system_prompt=self.system_prompt, user_prompt=prompt)

class Environment:
    def __init__(self, config:Config=None):
        self.config = config
        self.ucv = UnrealCV(port=9000)
        self.comm = Communicator(self.ucv)
        self.config = Config()
        self.map = Map(self.config)
        self.objects = self.ucv.get_objects()
        self.action_planner_llm = A2ALLM(model_name="gpt-4o-mini")
        self._map_initialized = False

    def _resolve_roads_path(self) -> str:
        roads_path = Path(self.config['map.input_roads'])
        if not roads_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            roads_path = repo_root / roads_path
        if roads_path.exists():
            return str(roads_path)
        fallback = Path(__file__).resolve().parents[1] / "data" / "example_city" / "roads.json"
        return str(fallback)

    def _ensure_map_initialized(self):
        if self._map_initialized or self.map.nodes:
            return
        roads_path = self._resolve_roads_path()
        self.map.initialize_map_from_file(roads_file=roads_path)
        self._map_initialized = True

    def reset(self):
        self._ensure_map_initialized()
        print(self.map.nodes)
        agent = self.spawn_humanoid()#agent_bp="/Game/TrafficSystem/Pedestrian/Base_Pedestrian.Base_Pedestrian_C")
        self.action_planner = LocalPlanner(agent=agent, model=self.action_planner_llm, rule_based=False)
        self.target = Vector(1000, 0)
        return agent

    def spawn_humanoid(self, name:str=None, agent_bp:str=None, position:Vector=Vector(0, 0), direction:Vector=Vector(1, 0)):
        if agent_bp is None:
            agent_bp = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C"
        agent = Humanoid(
            communicator=self.comm,
            position=position,
            direction=direction,
            config=self.config,
            map=self.map
        )
        self.comm.spawn_agent(agent, name=name, model_path=agent_bp, type="humanoid")
        self.humanoid_name = self.comm.get_humanoid_name(agent.id)
        return agent

    def step(self, action):
        primitive_actions = self.action_planner.parse(action)
        self.action_planner.execute(primitive_actions)

        loc_3d = self.ucv.get_location(self.humanoid_name)
        location = Vector(loc_3d[0], loc_3d[1])
        reward = -location.distance(self.target)
        return location, reward

    def close(self):
        self.ucv.disconnect()
        self.comm.disconnect()
        self.map.close()
        self.objects.close()
        self.action_planner.close()
        self.action_planner_llm.close()




if __name__ == "__main__":
    agent = AgentEgo(goal="Go to (1700, -1700) and pick up GEN_BP_Box_1_C.")
    env = Environment()
    obs = env.reset()
    for _ in range(100):
        action = agent.action(obs)
        print(f"action: {action}")
        obs, reward = env.step(action)
        print(f"obs: {obs}, reward: {reward}")

    env.close()
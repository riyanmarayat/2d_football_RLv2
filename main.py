from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
import numpy as np
import math

FIELD_WIDTH = 100  # Default lebar lapangan yang dipakai 100 meter
FIELD_HEIGHT = 75
NUM_AGENTS = 22
FRICTION = {
    'dry-grass-standard': 0.985,
    'wet-grass': 0.995,
    'very-wet-grass': 0.96,
    'rough-sandy': 0.92,
    'synthetic': 0.98,
    'ice-superslippery': 0.999,
}

class FootballRLEnvironment(ParallelEnv):
    metadata = {
        "name": "football_env_v0",
        "render_modes": ["human", "none"],
    }
    ball_list_states = {
        'pos_x': None,
        'pos_y': None,
        'vel_x': None,
        'vel_y': None,
        'radius': 0.11, # radius bola standar dalam meter
        'friction': None,
        'controlled_by': None,
    }
    team_list_states = {
        'team_field': None,
        'score': None,
        'shots': None,
        'shots_on_target': None,
        'passes': None,
    }
    agents_list_states = {
        'team': None,
        'pos_me_x': None,
        'pos_me_y': None,
        'vel_me_x': None,
        'vel_me_y': None,
        'dist_to_ball': None,
        'has_ball': None,
        'zone': None,
        'last_action': None,
        'is_pressured': None,
        'goal': None,
        'assist': None,
    }
    field_list_states = {
        'width': None,
        'height': None,
        'outer_margin': 3.0, # meter
        'total_width': None,
        'total_height': None,
        'condition': 'dry-grass-standard',
    }

    def __init__(self, render_mode="none", num_agents=NUM_AGENTS, field_height=FIELD_HEIGHT, field_width=FIELD_WIDTH,
                 field_condition='dry-grass-standard', agents=None):
        # Env params
        self.render_mode = render_mode
        self.num_agents = num_agents
        self.dt = 1.0 / 15 # Time step in seconds dibagi 15 fps
        self.last_goal = None
        self.team_states = [self.team_list_states.copy() for _ in range(2)]
        self.step_count = 0

        # Field params
        self.field_states = self.field_list_states.copy()
        self.field_states['condition'] = field_condition

        # Goal post params
        self.goal_width =  7.32
        self.goal_height = 2.44
        self.left_goal = {
            'x': self.field_outer_margin,
            'y_top': (self.field_height / 2) + self.field_outer_margin - (self.goal_width / 2),
            'y_bottom': (self.field_height / 2) + self.field_outer_margin + (self.goal_width / 2),
        }
        self.right_goal = {
            'x': self.field_width + self.field_outer_margin,
            'y_top': (self.field_height / 2) + self.field_outer_margin - (self.goal_width / 2),
            'y_bottom': (self.field_height / 2) + self.field_outer_margin + (self.goal_width / 2),
        }

        # Ball params
        self.ball_states = self.ball_list_states.copy()
        self.ball_states['friction'] = self._get_friction(field_condition)

        # Agent params
        self.agents_action = [None] * self.num_agents
        self.agents_state = [None] * self.num_agents
        if agents is None:
            self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        else:
            self.agents = agents
        # self.agent_data = [self.agent_list_states.copy() for _ in range(self.num_agents)]
        # self.initial_state = [self.agent_list_states.copy() for _ in range(self.num_agents)]

        # Action list (tanpa referensi variabel yang belum ada)
        self.action_list = [
            {'type': 'idle', 'target': None},
            {'type': 'move', 'target': None},
            {'type': 'move_up', 'target': None},
            {'type': 'move_up_right', 'target': None},
            {'type': 'move_right', 'target': None},
            {'type': 'move_down_right', 'target': None},
            {'type': 'move_down', 'target': None},
            {'type': 'move_down_left', 'target': None},
            {'type': 'move_left', 'target': None},
            {'type': 'move_up_left', 'target': None},
            {'type': 'sprint', 'target': None},
            {'type': 'sprint_up', 'target': None},
            {'type': 'sprint_up_right', 'target': None},
            {'type': 'sprint_right', 'target': None},
            {'type': 'sprint_down_right', 'target': None},
            {'type': 'sprint_down', 'target': None},
            {'type': 'sprint_down_left', 'target': None},
            {'type': 'sprint_left', 'target': None},
            {'type': 'sprint_up_left', 'target': None},
            {'type': 'dribble', 'target': None},
            {'type': 'pass_short', 'target': None},
            {'type': 'pass_through', 'target': None},
            {'type': 'lob_pass', 'target': None},
            {'type': 'shoot_power', 'target': None},
            {'type': 'shoot_placed', 'target': None},
        ]

        # Action space for all agents
        self.action_space = {
            agent: Discrete(len(self.action_list)) for agent in self.agents
        }

        # States space for all agents
        """
        state index 0-21 untuk tiap agent
        """
        self.states = []
        for i in range(self.num_agents):
            self.states[i] = self.agents_list_states.copy()
        self.initial_state = [] # ISI INITIAL STATES FOR ALL AGENTS

        self.observation_spaces = {} # Perlu buat fungsi get_observation


    def reset(self, seed=None, options=None):
        """
        Reset the env to starting values
        :param seed:
        :param options:
        :return:  observation of all agents and infos
        """
        # Reset agents and ball positions
        self.step_count = 0
        for i in range(self.num_agents):
            self.states[i]['pos_me_x'] = self.initial_state[i]['pos_me_x']
            self.states[i]['pos_me_y'] = self.initial_state[i]['pos_me_y']
            self.states[i]['vel_me_x'] = 0
            self.states[i]['vel_me_y'] = 0
            self.states[i]['dist_to_ball'] = None # Harus diisi
            self.states[i]['has_ball'] = False
            self.states[i]['zone'] = self.initial_state[i]['zone']
            self.states[i]['last_action'] = None
            self.states[i]['is_pressured'] = False
            self.states[i]['goal'] = 0
            self.states[i]['assist'] = 0

        # Reset  field state
        self.field_states['width'] = 100 # meter
        self.field_states['height'] = 75 # meter
        self.field_states['total_width'] = self.field_states['width'] + (2 * self.field_states['outer_margin'])
        self.field_states['total_height'] = self.field_states['height'] + (2 * self.field_states['outer_margin'])
        self.field_states['condition'] = 'dry-grass-standard' # default condition

        # Reset ball position
        self.ball_states['pos_x'] = (self.field_width / 2) + self.field_outer_margin
        self.ball_states['pos_y'] = (self.field_height / 2) + self.field_outer_margin
        self.ball_states['vel_x'] = 0
        self.ball_states['vel_y'] = 0
        self.ball_states['friction'] = self._get_friction(self.field_states['condition'])
        self.ball_states['controlled_by'] = None

        for i in range(2):
            self.team_states[i]['score'] = 0
            self.team_states[i]['shots'] = 0
            self.team_states[i]['shots_on_target'] = 0
            self.team_states[i]['passes'] = 0


        return self.observation_spaces, self.states

    def step(self, action):
        """
        Jalankan satu langkah simulasi:
            1. tentukan siapa pengendali bola (ball_controller)
            2. susun urutuan pemain untuk aksi
            3. Untuk setiap pemain dalam urutan:
                - Ambil state
                - tentutan aksi agent
                - Apply action untuk menerapkan aksi tiap agent
            4. Perbarui fisika bola dan deteksi gol atau out-of-bounds
            5. Catat snapshot
        :param action:
        :return:
        """
        self.step_count += 1

        # Menentukan siapa yang mengontrol bola
        player_order = []
        index_player_order = []
        """
        Jika tidak ada yang mengontrol bola, cari pemain terdekat dengan bola.
        Jika ada yang mengontrol bola, cari pemaain terdekat dengan pengendali bola.
        """
        if self.ball_controller is None:
            # Cari pemain yang paling dekat dengan bola
            player_order = sorted(self.states, key=lambda p: math.hypot(p[]))


    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_space[agent]

    def _get_friction(self, condition):
        return FRICTION.get(condition, 0.985)  # Default friction for 'dry-grass-standard'

    def _get_observation_space(self):

        raise NotImplementedError


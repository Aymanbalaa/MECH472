
// maximum number of robots in the simulation
const int N_MAX = 100;

// maximum string size for obstacle file names
const int S_MAX = 50;

class robot_system {
	public:

	// robot system simulation time
	double t;

	// number of robots in the simulation
	int N_robot;

	// array of pointers to robot objects
	robot *P[N_MAX];

	// simulated image width and height
	double width, height;

	// obstacle locations and sizes
	int N_obs;
	double x_obs[N_MAX];
	double y_obs[N_MAX];

	// mode = 0 - single player mode (manual opponent)
	// mode = 1 - two player mode, player #1
	// mode = 2 - two player mode, player #2
	int mode;

	// constructor -- set default simulation parameters
	robot_system(double D, double Lx, double Ly, 
		double Ax, double Ay, double alpha_max, int n_robot);
	
	~robot_system();
	
	void sim_step(double dt);
};


int activate_simulation(double width, double height,
	double x_obs[], double y_obs[], int N_obs,
	char robot_file[], char opponent_file[], char background_file[],
	char obstacle_file[][S_MAX], double D, double Lx, double Ly, 
	double Ax, double Ay, double alpha_max, int n_robot);
	
int deactivate_simulation();

int set_inputs(int pw_l, int pw_r, int pw_laser, int laser, 
				double max_speed);
	
int set_opponent_inputs(int pw_l, int pw_r, int pw_laser, int laser, 
				double max_speed);	

int acquire_image_sim(image &rgb);

int set_simulation_mode(int mode);

int set_robot_position(double x, double y, double theta);

int set_opponent_position(double x, double y, double theta);

int draw_image(image &a, double theta, double ia, double ja, 
			   image &b, double ib, double jb, image &output,
			   int blend = 0);

int get_image(image &a, image &b, double ib, double jb, double theta);

int draw_laser(robot *P, image &rgb);

int draw_point_rgb_laser(image &rgb, int ip, int jp, int R, int G, int B);

int wait_for_player();

int join_player();

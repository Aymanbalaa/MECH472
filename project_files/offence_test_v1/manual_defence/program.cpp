
// MECH 472/663 - MANUAL DEFENCE (player 2)
// WASD drives robot_B. Used as the human-controlled sparring partner
// against the auto_offence in this folder.
//
// mode = 2 (two player, this process is player #2, calls join_player)
//
// Obstacles, robot starts and N_obs MUST match auto_offence/program.cpp
// exactly. Any mismatch silently breaks the simulator's shared-memory
// state and you'll see the wrong arena.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <Windows.h>

using namespace std;

#define KEY(c) ( GetAsyncKeyState((int)(c)) & (SHORT)0x8000 )

#include "image_transfer.h"
#include "vision.h"
#include "robot.h"
#include "vision_simulation.h"
#include "timer.h"
#include "update_simulation.h"

extern robot_system *S1;

int main()
{
	const double PI = 3.14159265;

	// ---- sim setup (must match auto_offence) ----
	double width1 = 640, height1 = 480;
	const int N_obs = 2;
	double x_obs[N_obs] = { 200.0, 440.0 };
	double y_obs[N_obs] = { 180.0, 300.0 };
	char obstacle_file[N_obs][S_MAX] = {
		"obstacle_black.bmp", "obstacle_green.bmp"
	};
	double D = 121.0, Lx = 31.0, Ly = 0.0, Ax = 37.0, Ay = 0.0;
	double alpha_max = PI / 2.0;
	int n_robot = 2;

	int pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
	double max_speed = 200.0;

	cout << "\n=== MANUAL DEFENCE (player 2) ===";
	cout << "\nW/S = forward/back   A/D = turn CCW/CW";
	cout << "\nSpace = fire laser (held, for testing the offence)";
	cout << "\nX = exit";
	cout << "\nPress space to begin.";
	pause();

	activate_vision();
	activate_simulation(width1, height1,
		x_obs, y_obs, N_obs,
		"robot_A.bmp", "robot_B.bmp", "background.bmp",
		obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);

	set_simulation_mode(2);
	set_robot_position(140, 400, 0.0);          // robot_B start
	set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

	int width = 640, height = 480;
	image rgb;
	rgb.type = RGB_IMAGE; rgb.width = width; rgb.height = height;
	allocate_image(rgb);

	// player 2 joins the player 1 process
	join_player();

	double tc0 = high_resolution_time(), tc;
	int frame = 0;

	while (1) {

		acquire_image_sim(rgb);
		tc = high_resolution_time() - tc0;
		frame++;

		// reset to neutral every frame, then add WASD
		pw_l = 1500; pw_r = 1500;

		// drive convention (matches the sign used by the offence):
		//   forward => pw_l<1500, pw_r>1500
		//   CCW spin => pw_l>1500, pw_r>1500
		if (KEY('W')) { pw_l -= 250; pw_r += 250; }   // forward
		if (KEY('S')) { pw_l += 250; pw_r -= 250; }   // backward
		if (KEY('A')) { pw_l += 200; pw_r += 200; }   // CCW
		if (KEY('D')) { pw_l -= 200; pw_r -= 200; }   // CW

		// fire laser while space is held - lets you test offence's
		// reaction by trying to hit it back. Held-down, not one-shot.
		laser = KEY(' ') ? 1 : 0;

		// clamp servos
		if (pw_l > 2000) pw_l = 2000;
		if (pw_l < 1000) pw_l = 1000;
		if (pw_r > 2000) pw_r = 2000;
		if (pw_r < 1000) pw_r = 1000;

		set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

		// only player 1 calls view_rgb_image - leaving it out here
		// avoids double-display / window twitching, same as v3 defence

		// Status print. In two-player mode, each process sees its own
		// robot at S1->P[1] and the opponent at S1->P[2] (both views
		// re-index "self" to slot 1). So in mode 2 here, the defence's
		// own pose is at P[1], not P[2]. Reading P[2] would print the
		// OFFENCE's pose -- looks plausible but is wrong telemetry.
		if (frame % 60 == 0) {
			double rx = S1->P[1]->x[2];
			double ry = S1->P[1]->x[3];
			double rt = S1->P[1]->x[1];
			double ox = S1->P[2]->x[2];
			double oy = S1->P[2]->x[3];
			cout << "\n[t=" << (int)tc << "s]"
			     << " me=(" << (int)rx << "," << (int)ry << ")"
			     << " th=" << (int)(rt*180.0/PI) << "deg"
			     << " opp=(" << (int)ox << "," << (int)oy << ")"
			     << " pw=" << pw_l << "/" << pw_r
			     << " laser=" << laser;
		}

		if (KEY('X')) break;
	}

	free_image(rgb);
	deactivate_vision();
	deactivate_simulation();
	cout << "\ndone.\n";
	return 0;
}

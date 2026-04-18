
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>

#include <Windows.h>

using namespace std;

// KEY(c) returns true if the keyboard key 'c' is currently held down.
// Used to read WASD input for the opponent robot in real time.
#define KEY(c) ( GetAsyncKeyState((int)(c)) & (SHORT)0x8000 )

#include "image_transfer.h"

// vision.h provides image processing functions:
// copy, lowpass_filter, scale, erode, dialate, label_image,
// draw_point_rgb, etc.
#include "vision.h"

#include "robot.h"

// vision_simulation.h provides the simulated arena:
// activate_simulation, set_inputs, set_opponent_inputs,
// acquire_image_sim, set_robot_position, set_opponent_position, etc.
#include "vision_simulation.h"

#include "timer.h"

#include "update_simulation.h"

// S1 is the global robot_system object created inside the simulation library.
// We access it here to read the true simulation state (e.g. robot heading).
extern robot_system *S1;

// ============================================================
// FORWARD DECLARATIONS
// ============================================================
// These tell the compiler that the functions exist somewhere
// below in this file, so they can be called from main().

// features(): given a labelled blob, compute its centroid (ic, jc),
// pixel area, and average RGB colour from the original image.
int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave);

// calculate_HSV(): convert an RGB colour to Hue/Saturation/Value.
// Professor's convention: value = max(R,G,B), sat = (max-min)/max,
// hue in degrees [0, 360).
void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value);

// detect_objects(): runs over every labelled blob, classifies each one
// as belonging to the robot, the opponent, or an obstacle, updates
// the tracked centre positions, and draws coloured dots on the display.
int detect_objects(image &a, image &rgb0, image &label, int nlabels,
                   double robot_ic[], double robot_jc[], int &n_robot_blobs,
                   double &robot_center_ic, double &robot_center_jc,
                   double opp_ic[], double opp_jc[], int &n_opp_blobs,
                   double &opp_center_ic, double &opp_center_jc,
                   double obs_ic[], double obs_jc[], int &n_obs_detected,
                   image &rgb, int frame_count);

// ============================================================
// MAIN
// ============================================================

int main()
{
	// ---- General variables ----
	double x0, y0, theta0, max_speed, opponent_max_speed;
	int pw_l, pw_r, pw_laser, laser;       // servo pulse widths for our robot
	double width1, height1;
	int n_robot;
	double x_obs[50] = { 0.0 }, y_obs[50] = { 0.0 };
	double D, Lx, Ly, Ax, Ay, alpha_max;
	double tc, tc0;                         // clock time (unused but reserved)
	int mode;
	int pw_l_o, pw_r_o, pw_laser_o, laser_o; // servo pulse widths for opponent

	// ---- Vision tracking variables ----
	// These store the detected centroids of our robot's blobs each frame.
	// A robot sprite has two coloured markers, so up to 10 blobs are tracked.
	double robot_ic[10], robot_jc[10]; // individual blob positions (row, col)
	int    n_robot_blobs;              // how many blobs were found for our robot
	double robot_center_ic, robot_center_jc; // averaged centre of all robot blobs

	// Same tracking variables for the opponent robot.
	double opp_ic[10], opp_jc[10];
	int    n_opp_blobs;
	double opp_center_ic, opp_center_jc;

	// Obstacle centroid positions detected from vision each frame.
	double obs_ic[10], obs_jc[10];
	int    n_obs_detected;

	// ---- Arena setup ----
	// We have 2 obstacles; each needs a BMP image file and an (x, y) position
	// in the 640x480 arena.
	const int N_obs = 2;

	char obstacle_file[N_obs][S_MAX] = {
		"obstacle_black.bmp",   // obstacle 0 -- black square, upper area
		"obstacle_red.bmp"      // obstacle 1 -- red square, lower-right area
	};

	x_obs[0] = 280; y_obs[0] = 160;  // obstacle 0 position (pixels)
	x_obs[1] = 480; y_obs[1] = 320;  // obstacle 1 position (pixels)

	// Simulation image size (pixels) -- fixed at 640x480
	width1  = 640;
	height1 = 480;

	// ---- Robot physical model parameters ----
	// These describe the geometry of the simulated robot body.

	D = 121.0; // shaft length: distance between the two drive wheels (pixels)

	// Laser/gripper position in the robot's LOCAL coordinate frame.
	// In local coordinates the robot always points in the +x direction.
	Lx = 31.0; // how far forward the laser is from the robot centre
	Ly = 0.0;  // how far sideways (0 = centred)

	// Offset from the robot image centre to the axis of rotation
	// (the midpoint between the two wheels) in local coordinates.
	Ax = 37.0;
	Ay = 0.0;

	alpha_max = 3.14159/2; // maximum laser sweep angle (90 degrees, in radians)

	// n_robot = 2 means we have both robots active (our robot + the opponent).
	n_robot = 2;

	cout << "\npress space key to begin program.";
	pause();

	// ---- Activate libraries ----
	// The regular vision library must be activated before the simulation library.
	activate_vision();

	activate_simulation(width1, height1,
		x_obs, y_obs, N_obs,
		"robot_A.bmp", "robot_B.bmp", "background.bmp",
		obstacle_file, D, Lx, Ly,
		Ax, Ay, alpha_max, n_robot);

	// mode 0 = single-player / manual opponent (we control the opponent via WASD).
	mode = 0;
	set_simulation_mode(mode);

	// ---- Initial robot positions ----
	// Our robot starts on the left side of the arena, facing right (theta=0).
	x0 = 140;
	y0 = 240;
	theta0 = 0;
	set_robot_position(x0, y0, theta0);

	// Opponent starts on the right side, facing left (theta = pi).
	set_opponent_position(500, 240, 3.14159);

	// ---- Initial servo commands ----
	// Pulse width 1500 us = neutral (stopped).
	// pw_r > 1500 drives the right wheel forward.
	// pw_l < 1500 drives the left wheel forward (left servo is physically flipped).
	pw_l = 1500;
	pw_r = 1500;
	pw_laser = 1500;
	laser = 0;

	max_speed          = 180; // our robot's maximum wheel speed (pixels/s)
	opponent_max_speed = 100; // opponent's maximum wheel speed (pixels/s)

	set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

	pw_l_o = 1500; pw_r_o = 1500; pw_laser_o = 1500; laser_o = 0;
	set_opponent_inputs(pw_l_o, pw_r_o, pw_laser_o, laser_o, opponent_max_speed);

	// ---- Allocate image memory ----
	// All images must be allocated before use.

	image rgb;          // final display image (colour, shown to the user)
	int height = 480, width = 640;

	rgb.type   = RGB_IMAGE;
	rgb.width  = width;
	rgb.height = height;
	allocate_image(rgb);

	// a, b  -- greyscale working images used at each stage of the pipeline
	// rgb0  -- copy of the raw camera image, kept for colour averaging
	// label -- output of label_image(): each pixel stores which blob it belongs to
	image a, b, rgb0, label;

	a.type = GREY_IMAGE;    a.width = width;  a.height = height;
	b.type = GREY_IMAGE;    b.width = width;  b.height = height;
	rgb0.type = RGB_IMAGE;  rgb0.width = width; rgb0.height = height;
	label.type = LABEL_IMAGE; label.width = width; label.height = height;

	allocate_image(a);
	allocate_image(b);
	allocate_image(rgb0);
	allocate_image(label);

	// measure initial clock time
	tc0 = high_resolution_time();

	// Seed the centre estimates so the first frame's blob assignment has a
	// reasonable starting point (proximity to previous frame centre).
	robot_center_ic = x0;  robot_center_jc = y0;
	opp_center_ic   = 500; opp_center_jc   = 240;

	int nlabels; // number of connected blobs found by label_image()

	// ============================================================
	// MAIN CONTROL LOOP
	// ============================================================
	// Runs continuously. Each iteration:
	//   1. Grab a new simulated camera frame.
	//   2. Run the vision pipeline to detect robots and obstacles.
	//   3. Compute the desired heading using a potential field.
	//   4. Drive the robot toward the hiding waypoint.
	//   5. Read keyboard input and move the opponent.

	while(1) {

		// Step the physics simulation and render the new camera frame into rgb.
		acquire_image_sim(rgb);

		// ============================================================
		// VISION PIPELINE
		// ============================================================
		// Goal: take the raw colour image and produce a list of blob
		// centroids labelled as "robot", "opponent", or "obstacle".

		static int frame_count = 0;
		frame_count++;

		// Keep an untouched copy of the raw frame for colour averaging later.
		// rgb0 is never modified after this point.
		copy(rgb, rgb0);

		// Convert the colour image to greyscale so we can apply
		// intensity-based filters.
		copy(rgb, a);

		// Lowpass (blur) filter: smooths out single-pixel noise so that
		// erode/dilate work more reliably.
		lowpass_filter(a, b);
		copy(b, a);

		// Scale (contrast stretch): maps the darkest pixel to 0 and the
		// brightest to 255, making colour differences easier to threshold.
		scale(a, b);
		copy(b, a);

		// HSV threshold: decide for every pixel whether it belongs to a
		// coloured/dark object (foreground = 255) or the plain background (0).
		// A pixel is foreground if:
		//   - its saturation > 0.20  (it has strong colour -- robot markers, red obstacle)
		//   - OR its brightness < 50 (it is very dark -- black obstacle)
		// The background (wooden floor) is low-saturation and bright, so it maps to 0.
		{
			ibyte *prgb = rgb0.pdata; // pointer into the raw RGB image
			ibyte *pb   = b.pdata;   // pointer into the output binary image
			int npix = width * height;
			double h, s, v;
			for(int k = 0; k < npix; k++, prgb += 3) {
				int Bk = prgb[0], Gk = prgb[1], Rk = prgb[2]; // BGR byte order
				calculate_HSV(Rk, Gk, Bk, h, s, v);
				if( s > 0.20 || v < 50 ) pb[k] = 255; // foreground
				else                     pb[k] = 0;   // background
			}
		}
		copy(b, a);

		// Double erode: shrinks every white region by ~2 pixels on each side.
		// This removes tiny noise blobs that are too small to be real objects.
		erode(a, b); copy(b, a);
		erode(a, b); copy(b, a);

		// Double dilate: expands every remaining white region back to roughly
		// its original size. Together with the erode, this is called "opening"
		// and it cleans up the binary mask without losing large blobs.
		dialate(a, b); copy(b, a);
		dialate(a, b); copy(b, a);

		// Label connected components: assign a unique integer ID (1, 2, 3, ...)
		// to each separate white blob. nlabels = total number of blobs found.
		label_image(a, label, nlabels);

		// Classify all blobs, update centre estimates, and annotate the display.
		// After this call:
		//   robot_center_ic/jc  = pixel position of our robot
		//   opp_center_ic/jc    = pixel position of the opponent
		//   obs_ic[]/obs_jc[]   = pixel positions of each detected obstacle
		detect_objects(a, rgb0, label, nlabels,
		               robot_ic, robot_jc, n_robot_blobs,
		               robot_center_ic, robot_center_jc,
		               opp_ic, opp_jc, n_opp_blobs,
		               opp_center_ic, opp_center_jc,
		               obs_ic, obs_jc, n_obs_detected,
		               rgb, frame_count);

		// Save annotated and binary debug images on the very first frame only.
		if(frame_count == 1) {
			save_rgb_image("output.bmp", rgb);
			copy(a, rgb);
			save_rgb_image("binary.bmp", rgb);
			copy(rgb0, rgb);
			cout << "\ntest images saved: output.bmp and binary.bmp";
		}

		// ============================================================
		// ROBOT FRONT POSITION
		// ============================================================
		// Instead of using the robot's centre pixel, we project 65 px
		// forward along the robot's current heading to get the "front" position.
		// All force and waypoint calculations are done from this point so that
		// the robot's nose -- not its belly -- is what navigates.

		double rx = robot_center_ic; // robot centre (col = x in image)
		double ry = robot_center_jc; // robot centre (row = y in image)

		// sim_theta is the true heading angle from the physics engine (radians).
		// Positive = counter-clockwise from the +x (right) axis.
		double sim_theta = S1->P[1]->x[1];

		const double front_offset = 65.0; // pixels from centre to front of robot
		double fx = rx + front_offset * cos(sim_theta); // front x (col)
		double fy = ry + front_offset * sin(sim_theta); // front y (row)

		// ============================================================
		// HIDING SPOT SELECTION
		// ============================================================
		// For each detected obstacle, compute a "hiding spot": a point
		// placed hide_dist pixels BEHIND the obstacle along the line
		// from the opponent to the obstacle.
		//
		// Example: opponent at (500,240), obstacle at (280,160).
		//   The direction from opponent to obstacle is normalised, then
		//   we go hide_dist further past the obstacle, placing us in
		//   the obstacle's shadow from the opponent's point of view.
		//
		// The robot then navigates toward the best hiding spot.

		double hide_ic = fx, hide_jc = fy; // default: stay put
		double dx, dy, dist;

		const double hide_dist = 110.0; // pixels beyond the obstacle centre

		if(n_obs_detected > 0) {

			// Compute hiding spot for every visible obstacle.
			double hx_all[10], hy_all[10];
			for(int k = 0; k < n_obs_detected; k++) {
				double ox   = obs_ic[k];              // obstacle column
				double oy   = obs_jc[k];              // obstacle row
				double vx   = ox - opp_center_ic;     // vector: opponent -> obstacle
				double vy   = oy - opp_center_jc;
				double vlen = sqrt(vx*vx + vy*vy) + 1e-6; // length of that vector
				// hiding spot = obstacle centre + hide_dist in the same direction
				hx_all[k] = ox + hide_dist * vx / vlen;
				hy_all[k] = oy + hide_dist * vy / vlen;
				// clamp to arena boundaries so we don't navigate off-screen
				if(hx_all[k] <  10) hx_all[k] =  10;
				if(hx_all[k] > 630) hx_all[k] = 630;
				if(hy_all[k] <  10) hy_all[k] =  10;
				if(hy_all[k] > 470) hy_all[k] = 470;
			}

			// Score each hiding spot and pick the best one.
			// Score = (squared distance from opponent) - 0.3 * (squared distance from robot front)
			//
			// This means:
			//   - We strongly prefer spots that are FAR from the opponent (safest).
			//   - We apply a small penalty for spots that are far from our current position
			//     so we don't chase a hiding spot on the opposite side of the arena
			//     when a nearby one is almost as good.
			//   - As the opponent moves toward one obstacle, the other obstacle's score
			//     rises and the robot naturally switches target.
			double best_score = -1e9;
			for(int k = 0; k < n_obs_detected; k++) {
				double d_opp = (hx_all[k] - opp_center_ic)*(hx_all[k] - opp_center_ic)
				             + (hy_all[k] - opp_center_jc)*(hy_all[k] - opp_center_jc);
				double d_rob = (hx_all[k] - fx)*(hx_all[k] - fx)
				             + (hy_all[k] - fy)*(hy_all[k] - fy);
				double score = d_opp - 0.3 * d_rob;
				if(score > best_score) {
					best_score = score;
					hide_ic = hx_all[k];
					hide_jc = hy_all[k];
				}
			}
		}

		// ============================================================
		// LINE-OF-SIGHT CHECK
		// ============================================================
		// Determines whether the opponent can currently "see" our robot.
		// We cast a line segment from the opponent centre to our robot front.
		// If any obstacle centroid lies within obs_radius pixels of that segment,
		// the obstacle is blocking the view -- the robot is HIDDEN.
		//
		// in_los = true  --> opponent can see us (exposed) --> keep moving
		// in_los = false --> we are hidden behind an obstacle --> stop

		bool in_los = true;
		const double obs_radius = 5.0; // how close to the segment counts as "blocked"

		for(int k = 0; k < n_obs_detected; k++) {
			double ax = opp_center_ic, ay = opp_center_jc; // segment start (opponent)
			double bx = fx,           by = fy;             // segment end (robot front)
			double px = obs_ic[k],    py = obs_jc[k];      // obstacle centre

			// Find the closest point on the segment [A,B] to point P.
			// t is the normalised distance along the segment (clamped to [0,1]).
			double sdx = bx - ax, sdy = by - ay;
			double len2 = sdx*sdx + sdy*sdy + 1e-6;
			double t = ((px - ax)*sdx + (py - ay)*sdy) / len2;
			if(t < 0.0) t = 0.0;
			if(t > 1.0) t = 1.0;
			double cx = ax + t*sdx, cy = ay + t*sdy; // closest point on segment

			// Distance from obstacle centre to the closest point on the LOS segment.
			double d = sqrt((px - cx)*(px - cx) + (py - cy)*(py - cy));

			if(d < obs_radius) { in_los = false; break; } // obstacle blocks the view
		}

		// ============================================================
		// VECTOR VISUALISATION
		// ============================================================
		// Draw a line from the robot front to the hiding spot so we can
		// see the navigation target on screen.
		//   Yellow line = robot is exposed (opponent has line of sight)
		//   Green line  = robot is hidden (obstacle is blocking the view)
		{
			int vR = in_los ? 255 : 0; // red channel: 255=yellow, 0=green
			int vG = 255;
			int vB = 0;
			int n_steps = 80;
			for(int s = 0; s <= n_steps; s++) {
				double t2 = (double)s / n_steps;
				int pi = (int)(fx + t2*(hide_ic - fx));
				int pj = (int)(fy + t2*(hide_jc - fy));
				if(pi >= 0 && pi < width && pj >= 0 && pj < height)
					draw_point_rgb(rgb, pi, pj, vR, vG, vB);
			}
		}

		// ============================================================
		// POTENTIAL FIELD -- computes desired_heading
		// ============================================================
		// The robot is pushed and pulled by three types of virtual forces:
		//
		//   1. ATTRACTIVE force (Ka): pulls the robot toward the hiding spot.
		//      Always a unit vector -- constant magnitude regardless of distance.
		//
		//   2. OBSTACLE REPULSION (Kr): pushes the robot away from obstacle centres.
		//      Falls off with distance^3 so it is only strong up close.
		//      Prevents the robot from colliding with obstacles while navigating.
		//
		//   3. BOUNDARY REPULSION (Kb): pushes the robot away from arena edges.
		//      Falls off with distance^2 -- keeps the robot from getting trapped
		//      in corners or driving off the edge.
		//
		// The vector sum of all forces gives the desired travel direction.

		double Fx = 0, Fy = 0;

		// Attractive force: unit vector pointing from robot front to hiding spot.
		double Ka = 4.0;
		dx   = hide_ic - fx;
		dy   = hide_jc - fy;
		dist = sqrt(dx*dx + dy*dy) + 1e-6;
		Fx  += Ka * dx / dist; // unit vector scaled by Ka
		Fy  += Ka * dy / dist;

		// Obstacle repulsion: 1/r^2 force away from each obstacle centre.
		double Kr = 15000.0;
		for(int k = 0; k < n_obs_detected; k++) {
			dx = fx - obs_ic[k]; // vector from obstacle to robot front
			dy = fy - obs_jc[k];
			double dist2 = dx*dx + dy*dy + 1e-6;
			Fx += Kr * dx / (dist2 * sqrt(dist2)); // magnitude = Kr / r^2
			Fy += Kr * dy / (dist2 * sqrt(dist2));
		}

		// Boundary repulsion: pushes away from all four arena edges.
		// The closer the robot is to an edge, the stronger the push.
		double Kb = 15000.0;
		Fx += Kb / (fx*fx + 1);                      // left edge
		Fx -= Kb / ((640.0-fx)*(640.0-fx) + 1);      // right edge
		Fy += Kb / (fy*fy + 1);                       // top edge
		Fy -= Kb / ((480.0-fy)*(480.0-fy) + 1);      // bottom edge

		// The direction of the total force vector is the desired travel heading.
		double desired_heading = atan2(Fy, Fx); // angle in radians, [-pi, pi]

		// ============================================================
		// ROBOT HEADING ESTIMATION
		// ============================================================
		// The robot sprite has two coloured markers. The line drawn between
		// their two centroids gives us the robot's axis, but we don't know
		// which end is the front (180 degree ambiguity).
		//
		// Fix: compute the heading both ways (raw and raw+180), and pick
		// whichever one is closer to the desired_heading we just calculated.
		// This resolves the ambiguity using the direction we want to travel.
		//
		// If fewer than 2 blobs are detected, fall back to the simulation's
		// true heading value (S1->P[1]->x[1]).

		double rtheta;
		if(n_robot_blobs >= 2) {
			// angle of the axis joining blob[0] to blob[1]
			double raw = atan2(robot_jc[1] - robot_jc[0],
			                   robot_ic[1] - robot_ic[0]);
			// error if we use raw as-is
			double err1 = desired_heading - raw;
			while(err1 >  3.14159) err1 -= 2*3.14159;
			while(err1 < -3.14159) err1 += 2*3.14159;
			// error if we flip 180 degrees
			double err2 = desired_heading - (raw + 3.14159);
			while(err2 >  3.14159) err2 -= 2*3.14159;
			while(err2 < -3.14159) err2 += 2*3.14159;
			// pick whichever interpretation has smaller angular error
			rtheta = (fabs(err1) <= fabs(err2)) ? raw : raw + 3.14159;
		} else {
			rtheta = S1->P[1]->x[1]; // fallback to simulation truth
		}

		// Heading error: how far off we currently are from desired_heading.
		// Wrapped to [-pi, pi] so we always take the shorter turn direction.
		double heading_error = desired_heading - rtheta;
		while(heading_error >  3.14159) heading_error -= 2*3.14159;
		while(heading_error < -3.14159) heading_error += 2*3.14159;

		// ============================================================
		// WAYPOINT NAVIGATOR (state machine)
		// ============================================================
		// The navigator has two phases and several stop conditions:
		//
		//   STOP conditions (override everything):
		//     - Robot is hidden (in_los = false) --> hold position
		//     - Robot front is within arrive_dist of waypoint --> hold position
		//
		//   Phase 0 -- ROTATE:
		//     Spin in place until aligned with desired_heading.
		//     Both servos receive the same signed pulse so the robot turns
		//     without translating.
		//     Transition to Phase 1 once heading error < angle_thresh.
		//
		//   Phase 1 -- DRIVE:
		//     Drive forward toward the waypoint with gentle heading correction.
		//     Forward speed is proportional to distance (slows down as it arrives).
		//     If heading drifts too far off, revert to Phase 0 to re-align.

		static int wp_state = 0; // 0 = rotate, 1 = drive

		const double angle_thresh = 0.20; // ~11 degrees -- "good enough" alignment
		const double arrive_dist  = 30.0; // pixels -- "close enough" to waypoint

		// distance from robot front to the selected hiding spot
		double dist_to_wp = sqrt((hide_ic - fx)*(hide_ic - fx) +
		                         (hide_jc - fy)*(hide_jc - fy));

		if(!in_los) {
			// Hidden behind an obstacle -- stop and hold.
			// Reset to rotate state so we re-align if the opponent moves us out.
			pw_r = 1500; pw_l = 1500;
			wp_state = 0;

		} else if(dist_to_wp < arrive_dist) {
			// Reached the waypoint -- stop and let the hiding spot update next frame.
			pw_r = 1500; pw_l = 1500;
			wp_state = 0;

		} else if(wp_state == 0) {
			// Phase 0: rotate in place.
			// turn is proportional to heading_error; clamped so we don't spin too fast.
			if(fabs(heading_error) < angle_thresh) {
				// Aligned -- transition to driving phase.
				wp_state = 1;
				pw_r = 1500; pw_l = 1500;
			} else {
				int turn = (int)(300.0 * heading_error);
				if(turn >  450) turn =  450;
				if(turn < -450) turn = -450;
				// Both wheels receive the same turn delta so the robot spins in place.
				// (right servo: +turn = forward; left servo: +turn = backward because
				//  the left servo is physically reversed -- this creates rotation.)
				pw_r = 1500 + turn;
				pw_l = 1500 + turn;
			}

		} else {
			// Phase 1: drive toward waypoint with heading correction.
			if(fabs(heading_error) > angle_thresh * 2.5) {
				// Drifted too far off course -- revert to rotate phase.
				wp_state = 0;
				pw_r = 1500; pw_l = 1500;
			} else {
				// Forward speed proportional to remaining distance (smooth approach).
				int fwd = (int)(3.5 * dist_to_wp);
				if(fwd > 480) fwd = 480;
				if(fwd <  80) fwd =  80;
				// Gentle correction: small differential applied on top of forward speed.
				int turn = (int)(180.0 * heading_error);
				if(turn >  200) turn =  200;
				if(turn < -200) turn = -200;
				// Differential drive:
				//   pw_r > 1500 --> right wheel forward
				//   pw_l < 1500 --> left wheel forward (reversed servo)
				// Adding turn to both steers the robot.
				pw_r = 1500 + fwd + turn;
				pw_l = 1500 - fwd + turn;
			}
		}

		// Clamp all servo commands to the safe operating range [1000, 2000] us.
		if(pw_r > 2000) pw_r = 2000;
		if(pw_r < 1000) pw_r = 1000;
		if(pw_l > 2000) pw_l = 2000;
		if(pw_l < 1000) pw_l = 1000;

		// ============================================================
		// OPPONENT KEYBOARD CONTROL (WASD)
		// ============================================================
		// The human player drives the opponent robot using WASD keys.
		//   W = both wheels forward  (drive forward)
		//   S = both wheels backward (drive backward)
		//   A = right forward, left backward (turn left in place)
		//   D = right backward, left forward (turn right in place)
		//
		// Note: because the left servo is physically reversed, "left wheel forward"
		// means pw_l_o < 1500 and "left wheel backward" means pw_l_o > 1500.

		pw_r_o = 1500;
		pw_l_o = 1500;
		if(KEY('W')) { pw_r_o += 200; pw_l_o -= 200; } // forward
		if(KEY('S')) { pw_r_o -= 200; pw_l_o += 200; } // backward
		if(KEY('A')) { pw_r_o += 200; pw_l_o += 200; } // turn left
		if(KEY('D')) { pw_r_o -= 200; pw_l_o -= 200; } // turn right
		if(pw_r_o > 2000) pw_r_o = 2000;
		if(pw_r_o < 1000) pw_r_o = 1000;
		if(pw_l_o > 2000) pw_l_o = 2000;
		if(pw_l_o < 1000) pw_l_o = 1000;

		// Send commands to both robots.
		set_opponent_inputs(pw_l_o, pw_r_o, pw_laser_o, laser_o, opponent_max_speed);
		set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

		// Display mode 1: non-blocking -- show latest frame without waiting.
		v_mode = 1;
		view_rgb_image(rgb, v_mode);
	}

	// ---- Cleanup ----
	// Free all allocated image memory before the program exits.
	free_image(a);
	free_image(b);
	free_image(rgb0);
	free_image(label);
	free_image(rgb);

	deactivate_vision();
	deactivate_simulation();

	cout << "\ndone.\n";

	return 0;
}

// ============================================================
// detect_objects()
// ============================================================
// Called once per frame after label_image().
// Loops over every blob (label 1..nlabels), calls features() to get
// its centroid and average colour, then classifies it:
//
//   SMALL blob (area < size_threshold):
//     Could be one of the two robot markers or two opponent markers.
//     Classified by proximity: whichever robot centre from the PREVIOUS
//     frame is closer gets the blob assigned to it.
//     This handles the case where a robot produces 2 separate blobs
//     (e.g. two coloured patches on the sprite).
//
//   LARGE blob (area >= size_threshold):
//     Treated as an obstacle. Further classified by HSV colour so we
//     know which obstacle it is (black, red, orange, green, blue).
//
// After classification, robot/opponent centre positions are updated
// by averaging all blobs belonging to each entity.
// Coloured dots are drawn on the display image for debugging:
//   Green  dot = robot blob
//   Cyan   dot = opponent blob
//   White  dot = obstacle centre
//   Red    dot = averaged robot centre
//   Magenta dot = averaged opponent centre

int detect_objects(image &a, image &rgb0, image &label, int nlabels,
                   double robot_ic[], double robot_jc[], int &n_robot_blobs,
                   double &robot_center_ic, double &robot_center_jc,
                   double opp_ic[], double opp_jc[], int &n_opp_blobs,
                   double &opp_center_ic, double &opp_center_jc,
                   double obs_ic[], double obs_jc[], int &n_obs_detected,
                   image &rgb, int frame_count)
{
	// Blobs smaller than min_area pixels are noise -- ignore them.
	const double min_area       = 200.0;
	// Blobs larger than size_threshold are obstacles; smaller ones are robot parts.
	const double size_threshold = 2500.0;

	double ic, jc, area, R_ave, G_ave, B_ave;
	double hue, sat, value;
	int    R, G, B;
	const char *blob_name;

	// Reset blob counts for this frame.
	n_robot_blobs  = 0;
	n_opp_blobs    = 0;
	n_obs_detected = 0;

	for(int i_label = 1; i_label <= nlabels; i_label++) {

		// Compute centroid, area, and average colour for this blob.
		features(a, rgb0, label, i_label, ic, jc, area, R_ave, G_ave, B_ave);

		if(area < min_area) continue; // skip noise

		// Convert average colour to HSV for classification.
		calculate_HSV((int)R_ave, (int)G_ave, (int)B_ave, hue, sat, value);

		if(area < size_threshold) {

			// Small blob: assign to robot or opponent based on which one
			// this blob is closest to (using last frame's centre as reference).
			double d_robot = (ic - robot_center_ic)*(ic - robot_center_ic)
			               + (jc - robot_center_jc)*(jc - robot_center_jc);
			double d_opp   = (ic - opp_center_ic)*(ic - opp_center_ic)
			               + (jc - opp_center_jc)*(jc - opp_center_jc);

			if(d_robot <= d_opp) {
				blob_name = "robot";
				if(n_robot_blobs < 10) {
					robot_ic[n_robot_blobs] = ic;
					robot_jc[n_robot_blobs] = jc;
					n_robot_blobs++;
				}
				R = 0; G = 255; B = 0; // green dot
			} else {
				blob_name = "opponent";
				if(n_opp_blobs < 10) {
					opp_ic[n_opp_blobs] = ic;
					opp_jc[n_opp_blobs] = jc;
					n_opp_blobs++;
				}
				R = 0; G = 255; B = 255; // cyan dot
			}

		} else {

			// Large blob: classify by colour to identify which obstacle it is.
			// These ranges are tuned to the HSV colours of the obstacle BMPs.
			if(value < 50) {
				blob_name = "obstacle_black";   // very dark = black obstacle
			} else if((hue >= 340 || hue <= 20) && sat > 0.4) {
				blob_name = "obstacle_red";     // red hue wraps around 0/360
			} else if(hue > 20 && hue <= 45 && sat > 0.5) {
				blob_name = "obstacle_orange";
			} else if(hue > 80 && hue <= 160 && sat > 0.3) {
				blob_name = "obstacle_green";
			} else if(hue > 190 && hue <= 260 && sat > 0.3) {
				blob_name = "obstacle_blue";
			} else {
				blob_name = "unknown";
			}

			if(n_obs_detected < 10) {
				obs_ic[n_obs_detected] = ic;
				obs_jc[n_obs_detected] = jc;
				n_obs_detected++;
			}
			R = 255; G = 255; B = 255; // white dot
		}

		// Draw a dot at the blob centroid on the display image.
		draw_point_rgb(rgb, (int)ic, (int)jc, R, G, B);

		// On the first frame only: print blob info to the console for debugging.
		if(frame_count == 1) {
			cout << "\nlabel " << i_label << " [" << blob_name << "]";
			cout << "  centroid: (" << (int)ic << ", " << (int)jc << ")";
			cout << "  HSV: (" << (int)hue << " deg, " << sat
			     << ", " << (int)value << ")";
			cout << "  area: " << (int)area;
		}
	}

	if(frame_count == 1) cout << "\nnlabels = " << nlabels;

	// Update robot centre: average position of all robot blobs this frame.
	if(n_robot_blobs >= 1) {
		robot_center_ic = 0; robot_center_jc = 0;
		for(int k = 0; k < n_robot_blobs; k++) {
			robot_center_ic += robot_ic[k];
			robot_center_jc += robot_jc[k];
		}
		robot_center_ic /= n_robot_blobs;
		robot_center_jc /= n_robot_blobs;
		draw_point_rgb(rgb, (int)robot_center_ic, (int)robot_center_jc,
		               255, 0, 0); // red dot = our robot's averaged centre
	}

	// Update opponent centre: average position of all opponent blobs this frame.
	if(n_opp_blobs >= 1) {
		opp_center_ic = 0; opp_center_jc = 0;
		for(int k = 0; k < n_opp_blobs; k++) {
			opp_center_ic += opp_ic[k];
			opp_center_jc += opp_jc[k];
		}
		opp_center_ic /= n_opp_blobs;
		opp_center_jc /= n_opp_blobs;
		draw_point_rgb(rgb, (int)opp_center_ic, (int)opp_center_jc,
		               255, 0, 255); // magenta dot = opponent's averaged centre
	}

	return 0;
}

// ============================================================
// features()
// ============================================================
// Computes the centroid, area, and average RGB colour of one labelled blob.
//
// Parameters:
//   a       - binary GREY_IMAGE mask (not used for data here, kept for API compatibility)
//   rgb     - original RGB image -- used to sample pixel colours
//   label   - LABEL_IMAGE: each pixel value is the blob ID it belongs to
//   label_i - which blob ID to analyse
//
// Outputs:
//   ic, jc  - centroid column and row (floating point pixel coordinates)
//   area    - number of pixels in the blob
//   R_ave, G_ave, B_ave - average colour of the blob from the original image
//
// How it works:
//   Scan every pixel. If the label image says this pixel belongs to label_i,
//   add its (i, j) coordinate to the running sum (weighted equally = centroid)
//   and accumulate its RGB values for the colour average.

int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave)
{
	ibyte  *p, *pc;
	i2byte *pl;
	i4byte  i, j, k, width, height;
	double  mi, mj, m, rho, EPS = 1e-7, n;
	double  R_sum = 0, G_sum = 0, B_sum = 0;
	int     Rv, Gv, Bv;

	// Sanity checks: both images must be the same size and correct types.
	if(rgb.height != label.height || rgb.width != label.width) {
		cout << "\nerror in features: sizes of rgb, label are not the same!";
		return 1;
	}
	if(rgb.type != RGB_IMAGE || label.type != LABEL_IMAGE) {
		cout << "\nerror in features: input types are not valid!";
		return 1;
	}

	p  = rgb.pdata;            // pointer to RGB pixel data
	pl = (i2byte *)label.pdata; // pointer to label pixel data (2 bytes per pixel)

	width  = rgb.width;
	height = rgb.height;

	mi = mj = m = n = 0.0; // moment accumulators

	for(j = 0; j < height; j++) {
		for(i = 0; i < width; i++) {
			if(pl[j*width+i] == label_i) { // this pixel belongs to our blob

				k  = i + width*j;    // flat pixel index
				pc = p + 3*k;        // pointer to this pixel's BGR bytes

				// Note: OpenCV / BMP stores bytes in B-G-R order.
				Bv = *pc;
				Gv = *(pc+1);
				Rv = *(pc+2);

				R_sum += Rv;
				G_sum += Gv;
				B_sum += Bv;

				n++;       // pixel count (= area)

				rho = 1;   // uniform weighting (every pixel counts equally)
				m   += rho;
				mi  += rho * i; // accumulate column coordinate
				mj  += rho * j; // accumulate row coordinate
			}
		}
	}

	// Centroid = weighted sum / total weight.
	ic = mi / (m + EPS);
	jc = mj / (m + EPS);

	// Average colour = colour sum / pixel count.
	R_ave = R_sum / (n + EPS);
	G_ave = G_sum / (n + EPS);
	B_ave = B_sum / (n + EPS);

	// Area = total number of pixels in the blob.
	area = n;

	return 0;
}

// ============================================================
// calculate_HSV()
// ============================================================
// Converts an integer RGB colour [0-255 each] to HSV using the
// professor's convention:
//
//   value = max(R, G, B)           range: [0, 255]
//   sat   = (max - min) / max      range: [0, 1]   (0 = grey, 1 = fully saturated)
//   hue   = colour angle in degrees range: [0, 360)
//
// Hue calculation follows the standard hexagonal colour-wheel formula:
//   If max is R: hue = 60 * (G-B)/delta
//   If max is G: hue = 60 * ((B-R)/delta + 2)
//   If max is B: hue = 60 * ((R-G)/delta + 4)
// Negative hue values are wrapped to [0, 360) by adding 360.

void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value)
{
	int max, min, delta;
	double H;

	max = min = R;

	if(G > max) max = G;
	if(B > max) max = B;

	if(G < min) min = G;
	if(B < min) min = B;

	delta = max - min; // colour spread

	value = max; // brightness = brightest channel

	// Saturation: how "grey" vs "colourful" the pixel is.
	if(delta == 0) {
		sat = 0.0; // pure grey -- no colour
	} else {
		sat = (double)delta / value;
	}

	// Hue: which colour on the wheel.
	if(delta == 0) {
		H = 0; // grey pixels have undefined hue -- set to 0 by convention
	} else if(max == R) {
		H = (double)(G - B) / delta;
	} else if(max == G) {
		H = (double)(B - R) / delta + 2;
	} else {
		H = (double)(R - G) / delta + 4;
	}

	hue = 60 * H;
	if(hue < 0) hue += 360; // wrap negative hue back into [0, 360)
}

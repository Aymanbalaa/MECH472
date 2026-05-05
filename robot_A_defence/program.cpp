// DEFENSE ROBOT OF COLOR BLUE AND ORANGE
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <Windows.h>

using namespace std;

#define KEY(c) ( GetAsyncKeyState((int)(c)) & (SHORT)0x8000 )

#include "image_transfer.h"
#include "vision.h"
#include "robot.h"
#include "vision_simulation.h"
#include "timer.h"
#include "update_simulation.h"

extern robot_system* S1;

// these are the functions used one for the centroids one for the detection of the different markers and obstacles and one for the hsv calculations
int features(image& a, image& rgb, image& label, int label_i,
	double& ic, double& jc, double& area,
	double& R_ave, double& G_ave, double& B_ave);

int detect_objects(image& a, image& rgb0, image& label, int nlabels,
	double robot_ic[], double robot_jc[], int& n_robot_blobs,
	double& robot_center_ic, double& robot_center_jc,
	double opp_ic[], double opp_jc[], int& n_opp_blobs,
	double& opp_center_ic, double& opp_center_jc,
	double obs_ic[], double obs_jc[], int& n_obs_detected,
	image& rgb, int frame_count);

void calculate_HSV(int R, int G, int B, double& hue, double& sat, double& value);


int main()
{
	const double PI = 3.14159265;

	// the next few lines are the lines that setup the simulationm parameters and they are exactly used by the professor code in his classes
	double width1 = 640, height1 = 480;
	const int N_obs = 2;
	double x_obs[N_obs] = { 250, 400.0 };
	double y_obs[N_obs] = { 240, 240.0 };
	char obstacle_file[N_obs][S_MAX] = {
		"obstacle_black.bmp", "obstacle_black.bmp"
	};
	double D = 121.0, Lx = 31.0, Ly = 0.0, Ax = 37.0, Ay = 0.0;
	double alpha_max = PI / 2.0;
	int n_robot = 2;

	int pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
	double max_speed = 200.0;

	// here we are declaring the robot opponent  and obstacle centorids and the nuumber of detected objects of each class and they are stored in the array that will be used in detect objects
	double robot_ic[10], robot_jc[10];
	int    n_robot_blobs;
	double robot_center_ic, robot_center_jc;

	double opp_ic[10], opp_jc[10];
	int    n_opp_blobs;
	double opp_center_ic, opp_center_jc;

	double obs_ic[10], obs_jc[10];
	int    n_obs_detected;

	// persistent obstacle positions (survive temporary occlusion)
	double known_obs_ic[10], known_obs_jc[10];
	int    n_known_obs = 0;

	int width = 640, height = 480;

	cout << "\nPLAYER 1 - DEFENSE - RED AND GREEN ROBOT";
	cout << "\nPress space to begin.";
	pause();
	// here all the activation of the vision simulation functions so that everything is called in with the correct parameters and correct order
	activate_vision();
	activate_simulation(width1, height1, x_obs, y_obs, N_obs, "robot_A.bmp", "robot_B.bmp", "background.bmp", obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);

	set_simulation_mode(1);
	set_robot_position(500, 100, PI);
	set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

	// here the images are declared  and allocated for the vision identifying its width and height and they are used to scan the entire area of the simulation
	image rgb, a, b, rgb0, label;

	rgb.type = RGB_IMAGE;   rgb.width = width; rgb.height = height;
	a.type = GREY_IMAGE;  a.width = width; a.height = height;
	b.type = GREY_IMAGE;  b.width = width; b.height = height;
	rgb0.type = RGB_IMAGE;   rgb0.width = width; rgb0.height = height;
	label.type = LABEL_IMAGE; label.width = width; label.height = height;

	allocate_image(rgb);
	allocate_image(a);
	allocate_image(b);
	allocate_image(rgb0);
	allocate_image(label);
	// here are the initial robot position that can be changed if we wnat different simulation results just to be safe they are used later on
	robot_center_ic = 500; robot_center_jc = 100;
	opp_center_ic = 140; opp_center_jc = 240;

	int nlabels;
	static int frame_count = 0;

	wait_for_player();

	double tc0 = high_resolution_time(), tc;

	//thats the main loop where all the functions are called in and all the detection and driving happens
	while (1) {

		acquire_image_sim(rgb);
		tc = high_resolution_time() - tc0;
		frame_count++;

		copy(rgb, rgb0);

		//hsv is firstly used here to differentiate between the background white/grey and the obstacles/markers and having a saturatioon above 0.2 makes it easier to detect
// the colorful markers/objects and the value below 50 cptures the blcak and the dark objects
		{
			ibyte* prgb = rgb0.pdata;
			ibyte* pa = a.pdata;
			int npix = width * height;
			double hue, sat, value;
			for (int k = 0; k < npix; k++, prgb += 3) {
				int Bk = prgb[0], Gk = prgb[1], Rk = prgb[2];
				calculate_HSV(Rk, Gk, Bk, hue, sat, value);
				if (sat > 0.20 || value < 50) pa[k] = 255; else pa[k] = 0;
			}
		}
		//this filtering sequence was tested multiple times and it was the best way to eleminate all the noise and the unwanted pixels and a lot of different combinations were tested and this is the finl one
		erode(a, b);   copy(b, a);
		erode(a, b);   copy(b, a);
		dialate(a, b); copy(b, a);
		dialate(a, b); copy(b, a);
		// once the hsv seperated everything now you can detect and label all the available markers and obstacles and diffrentiate between them
		label_image(a, label, nlabels);
		detect_objects(a, rgb0, label, nlabels, robot_ic, robot_jc, n_robot_blobs, robot_center_ic, robot_center_jc, opp_ic, opp_jc, n_opp_blobs, opp_center_ic, opp_center_jc, obs_ic, obs_jc, n_obs_detected, rgb, frame_count);

		//this block of code is used in case color detection failed it will be able to diffrentiate between the 2 markers and all other objects detected by checking the distance betwween the 2 markers stays the same
		static double prev_marker_dist = -1.0;
		if (n_robot_blobs >= 2) {
			double dx = robot_ic[0] - robot_ic[1];
			double dy = robot_jc[0] - robot_jc[1];
			double marker_dist = sqrt(dx * dx + dy * dy);
			if (prev_marker_dist >= 0.0 && fabs(marker_dist - prev_marker_dist) > 20.0)
				n_robot_blobs = 0;
			else
				prev_marker_dist = marker_dist;
		}
		//this is used to check if the robot goes over an aobstacle for over than one frame because when it goes over an obstackle the orientation the marker and the obstacle will all disapear
		// it will be used down stairs to command the robot to stop if the number of lost frames is more than 10 which means that the robot is either outside the arena or on top of an obstacle
		static int robot_lost_frames = 0;
		if (n_robot_blobs >= 1) robot_lost_frames = 0;
		else                    robot_lost_frames++;

		bool opp_visible = (n_opp_blobs >= 1);

		// the main purpose of this block is to detect the obstacles at the first frame which is the cleanest state of the obstacles and saves it in memory because the obstacles dont actually move so now if the oponent drives
// over the obstacle it will still remeber its position and calculate the hiding spot accordingly
		for (int j = 0; j < n_obs_detected; j++) {
			int bk = -1; double bd2 = 60.0 * 60.0;
			for (int k = 0; k < n_known_obs; k++) {
				double d2 = (obs_ic[j] - known_obs_ic[k]) * (obs_ic[j] - known_obs_ic[k])
					+ (obs_jc[j] - known_obs_jc[k]) * (obs_jc[j] - known_obs_jc[k]);
				if (d2 < bd2) { bd2 = d2; bk = k; }
			}
			if (bk >= 0) {
				known_obs_ic[bk] = obs_ic[j];
				known_obs_jc[bk] = obs_jc[j];
			}
			else if (n_known_obs < N_obs) {
				known_obs_ic[n_known_obs] = obs_ic[j];
				known_obs_jc[n_known_obs] = obs_jc[j];
				n_known_obs++;
			}
		}
		//this block of code calculates the nose position of the robot using rx and ry and adding lx which was already defiend which
// is multiplied by cos and sin to calculate it based on the robots orientation
		double rx = robot_center_ic, ry = robot_center_jc;
		static double rtheta_smooth = PI;
		static int    rtheta_init = 0;
		double nose_x = rx + Lx * cos(rtheta_smooth);
		double nose_y = ry + Lx * sin(rtheta_smooth);

		//this is how many pixels the robot will go to behind the obstacle so basically 160 pixels after the end of an obstacle
		const double hide_offset = 160.0;

		//this block of code was not present before but while running the tests we figured out that the robot back end i getting caught by the attacker so we did an los check where
		bool los_on_bottom = false;
		if (n_robot_blobs >= 1 && n_opp_blobs >= 2 && opp_visible) {
			int lowest_marker_idx = 0;
			for (int m = 1; m < n_robot_blobs; m++)
				if (robot_jc[m] > robot_jc[lowest_marker_idx]) lowest_marker_idx = m;// this choses the centroid which is at the bottom of the robot basicaly it is choosing the marker with the the lower centroid which usually identifies as the bottom marker

			double opp_raw = atan2(opp_jc[1] - opp_jc[0], opp_ic[1] - opp_ic[0]);// this is the raw angle of the opponent but not knowing which is the front and which is the back
			double dir_us = atan2(robot_center_jc - opp_center_jc, robot_center_ic - opp_center_ic); // and this is the angle between the opponent and the defence robot
			double er1 = dir_us - opp_raw;// this is the error between the 2 angles
			while (er1 > PI) er1 -= 2 * PI;
			while (er1 < -PI) er1 += 2 * PI;// these 2 are used to wrap the error betwee pi and - pi
			double er2 = dir_us - (opp_raw + PI);// this is the same error but on the opposite side +pi
			while (er2 > PI) er2 -= 2 * PI;
			while (er2 < -PI) er2 += 2 * PI;//same here
			double opp_hdg;
			if (fabs(er1) <= fabs(er2)) opp_hdg = opp_raw;
			else                        opp_hdg = opp_raw + PI;// using absolute value of this which ever error is the smallest its the angle chosen for the opponent heading

			double bx = robot_ic[lowest_marker_idx] - opp_center_ic;
			double by = robot_jc[lowest_marker_idx] - opp_center_jc;//this is the vector from opponent to the bottom marker of the robot
			double rc = cos(opp_hdg), rs = sin(opp_hdg);// and this is the vector of oponent heading
			double proj = bx * rc + by * rs;// using the dot prodoct of both you can know if the robot is in front or at the back or if you are close or not
			double perp = fabs(bx * rs - by * rc);//this the cross product which will give you the perpendiclar distance of these 2 vectors which should be in a 30 pixel tolerance

			bool los_blocked = false;
			for (int k = 0; k < n_known_obs; k++) {
				double ox = known_obs_ic[k] - opp_center_ic;
				double oy = known_obs_jc[k] - opp_center_jc;
				double obs_proj = ox * rc + oy * rs;
				double obs_perp = fabs(ox * rs - oy * rc);// the same logic here is used as before but we are checking it with the obstacle
				if (obs_proj > 0.0 && obs_proj < proj && obs_perp < 60.0) {// here is obstacle in fromt of the opponent and if the obstacle is in between the robot and oppoenent and if the obbstacle is within 60 pixels of the perpendicalr distance it means that the obstacle is blocking the line of sight
					los_blocked = true;
					break;
				}
			}
			los_on_bottom = (!los_blocked && proj > 0.0 && perp < 30.0);// and this is where the los works with a tolerance of 30 pixels
		}


		//this block of code computes the best hiding spot dynamically because in one the professors lectures he did a waypoint logic to the middle of the arena so i used that logic and applied it so that this waypoint can change dynamically depending on obstacle position and opponent position
		int best_obs = -1;
		double best_score = -1.0;
		double target_x = nose_x, target_y = nose_y;
		for (int k = 0; k < n_known_obs; k++) {
			double obs_x = known_obs_ic[k], obs_y = known_obs_jc[k];
			double dir_x = obs_x - opp_center_ic, dir_y = obs_y - opp_center_jc; // vector from opponent to obstacle
			double dist = sqrt(dir_x * dir_x + dir_y * dir_y) + 1e-6;//distance form opponnent to the obstacle
			double hide_x = obs_x + hide_offset * dir_x / dist; // so dividing the vector by its distance makes it a unit vector and then multiplying it by the 160 pixels extends the distance of the center of the obstacle by 160 in the x direction the same for y
			double hide_y = obs_y + hide_offset * dir_y / dist;
			if (hide_x < 80) hide_x = 80;
			if (hide_x > 560) hide_x = 560;
			if (hide_y < 80) hide_y = 80;
			if (hide_y > 400) hide_y = 400; // these are used so that the waypoint doesnt go outside the arena
			double dist_robot = sqrt((hide_x - nose_x) * (hide_x - nose_x) + (hide_y - nose_y) * (hide_y - nose_y)) + 1.0;
			double dist_opp = sqrt((hide_x - opp_center_ic) * (hide_x - opp_center_ic) + (hide_y - opp_center_jc) * (hide_y - opp_center_jc));
			double score = dist_opp / dist_robot;// here if the score is above 1 it means the opponenet is far and if it is lower than 1 robot is close and the higher the score the better the hiding spot
			if (score > best_score) { best_score = score; best_obs = k; target_x = hide_x; target_y = hide_y; }// since the score is high the hiding spot will take the corresponding values
		}
		static double hide_ic = -1.0, hide_jc = -1.0;
		if (hide_ic < 0.0) { hide_ic = target_x; hide_jc = target_y; }
		else { hide_ic = 0.92 * hide_ic + 0.08 * target_x; hide_jc = 0.92 * hide_jc + 0.08 * target_y; }// in order to not make the hiding spot jump around becasue it happened during testing the simulation this line of code that researched about basically hold 92% of the previous
		// posyion of the hiding spot and 8% of the new hiding spot and sisnce its a loop it makes changing the hiding spot a lot smoother and not sudden

		// the potential field explained in the final uses the forces to calculate the desired heading and calcualting the optimal angle to avoid all obstacles and boundries and go towards the hiding spot
		double Fx = 0, Fy = 0;
		double dx = hide_ic - nose_x, dy = hide_jc - nose_y;
		double dist = sqrt(dx * dx + dy * dy) + 1e-6;
		Fx += 4.0 * dx / dist; Fy += 4.0 * dy / dist;     // Ka=4.0 attraction

		for (int k = 0; k < n_known_obs; k++) {
			dx = nose_x - known_obs_ic[k]; dy = nose_y - known_obs_jc[k];
			double d2 = dx * dx + dy * dy + 1e-6;
			Fx += 15000.0 * dx / (d2 * sqrt(d2));         // Kr=15000 1/r^3 repulsion
			Fy += 15000.0 * dy / (d2 * sqrt(d2));
		}

		Fx += 5000.0 / (nose_x * nose_x + 1);            // Kb=5000 boundary repulsion
		Fx -= 5000.0 / ((640.0 - nose_x) * (640.0 - nose_x) + 1);
		Fy += 5000.0 / (nose_y * nose_y + 1);
		Fy -= 5000.0 / ((480.0 - nose_y) * (480.0 - nose_y) + 1);

		// desired heading is calculculated using atan2 of the total forces in x and y
		double desired_heading = atan2(Fy, Fx);
		//the heading orientation logic was used exactly the same as previously the same logic was used
		double rtheta;
		if (n_robot_blobs >= 2) {

			double blob_angle = atan2(robot_jc[1] - robot_jc[0], robot_ic[1] - robot_ic[0]);

			double diff_normal = desired_heading - blob_angle;
			while (diff_normal > PI) diff_normal -= 2 * PI;
			while (diff_normal < -PI) diff_normal += 2 * PI;
			double diff_flipped = desired_heading - (blob_angle + PI);
			while (diff_flipped > PI) diff_flipped -= 2 * PI;
			while (diff_flipped < -PI) diff_flipped += 2 * PI;
			if (fabs(diff_normal) <= fabs(diff_flipped)) rtheta = blob_angle;
			else                                         rtheta = blob_angle + PI;

			if (!rtheta_init) { rtheta_smooth = rtheta; rtheta_init = 1; }
			double angle_step = rtheta - rtheta_smooth;
			while (angle_step > PI) angle_step -= 2 * PI;
			while (angle_step < -PI) angle_step += 2 * PI;

			rtheta_smooth += 0.5 * angle_step;// this is the same logic as before too the angle keeps incresoing by 50% of its value everytime to make tuurning and driving smoother
			rtheta = rtheta_smooth;
		}
		else {
			// if the marker is hnot visible in one of the frames it holds the last known angle
			if (rtheta_init) rtheta = rtheta_smooth;
			else             rtheta = 0.0;
		}

		//now here using the angles calculated with the pontential field and the robots angle we use the same logic too and we call it heading error which the differene between the robots angle and the optimal angle
		double heading_error = desired_heading - rtheta;
		while (heading_error > PI) heading_error -= 2 * PI;
		while (heading_error < -PI) heading_error += 2 * PI;

		// this is here the driving logic
		double dist_to_wp = sqrt((hide_ic - nose_x) * (hide_ic - nose_x) + (hide_jc - nose_y) * (hide_jc - nose_y));// distance to waypoint
		int pw_r_raw = 1500, pw_l_raw = 1500;

		if (n_known_obs < 2) {
			pw_r_raw = 1500; pw_l_raw = 1500; // robots stops if the opponent is covering one of the obstacles

		}
		else if (n_opp_blobs < 1) {
			pw_r_raw = 1500; pw_l_raw = 1500; // if the opponent is out of frame it stays still too

		}
		else if (robot_lost_frames > 10) {// if the robot is not in the frame stop it too
			pw_r_raw = 1500; pw_l_raw = 1500;

		}
		else if (dist_to_wp < 30.0) {// if the robot is already at the waypoint stop it before
			pw_r_raw = 1500; pw_l_raw = 1500;

		}
		else {
			double F_mag = sqrt(Fx * Fx + Fy * Fy);// thi is the potential field magnitude
			int fwd = 0, turn = 0;

			static int spin_dir = 0;
			if (fabs(heading_error) > PI / 2.0) {// if the heading error is above 90 degrees which means the robot is in opposite direction
				if (spin_dir == 0) {
					if (heading_error >= 0) spin_dir = 1;
					else                    spin_dir = -1;
				}
				turn = spin_dir * 380;// the robot will turn any way and if the heading error is postive the robot will turn in the positive way and the opposite is the same
				fwd = 0;
			}
			else {
				spin_dir = 0; //it releases the turning
				double fwd_f = F_mag * cos(heading_error);//the robot goes towards the hiding spot now using simple the smaller the heading error the faster the robot goes
				if (fwd_f < 0.0) fwd_f = 0.0;
				fwd = (int)(100.0 * fwd_f);
				if (fwd > 430) fwd = 430;
				turn = (int)(250.0 * heading_error);// the smaller the heading error the less fatser the robot turns
				if (turn > 380) turn = 380;
				if (turn < -380) turn = -380;
			}
			pw_r_raw = 1500 + fwd + turn;// and both are combined here to have the best possible driving command
			pw_l_raw = 1500 - fwd + turn;
		}
		static double right_wheel_smoothed = 1500.0, left_wheel_smoothed = 1500.0;
		right_wheel_smoothed = 0.40 * right_wheel_smoothed + 0.60 * pw_r_raw;
		left_wheel_smoothed = 0.40 * left_wheel_smoothed + 0.60 * pw_l_raw;// same logic so that the driving is not all over the place 
		pw_r = (int)right_wheel_smoothed;
		pw_l = (int)left_wheel_smoothed;

		if (pw_r > 2000) pw_r = 2000;
		if (pw_r < 1000) pw_r = 1000;
		if (pw_l > 2000) pw_l = 2000;
		if (pw_l < 1000) pw_l = 1000;

		laser = 0;
		view_rgb_image(rgb, 2);
		set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
		if (KEY('X')) break;
	}

	free_image(rgb);
	free_image(a);
	free_image(b);
	free_image(rgb0);
	free_image(label);
	deactivate_vision();
	deactivate_simulation();
	return 0;
}

int detect_objects(image& a, image& rgb0, image& label, int nlabels, double robot_ic[], double robot_jc[], int& n_robot_blobs, double& robot_center_ic, double& robot_center_jc,
	double opp_ic[], double opp_jc[], int& n_opp_blobs, double& opp_center_ic, double& opp_center_jc, double obs_ic[], double obs_jc[], int& n_obs_detected, image& rgb, int frame_count)
{
	const double min_area = 200.0;
	const double size_threshold = 2500.0;

	double ic, jc, area, R_ave, G_ave, B_ave;
	double hue, sat, value;
	int    R, G, B;
	const char* blob_name;

	n_robot_blobs = 0;
	n_opp_blobs = 0;
	n_obs_detected = 0;

	for (int i_label = 1; i_label <= nlabels; i_label++) {

		features(a, rgb0, label, i_label, ic, jc, area, R_ave, G_ave, B_ave);
		if (area < min_area) continue;

		calculate_HSV((int)R_ave, (int)G_ave, (int)B_ave, hue, sat, value);

		if (area < size_threshold) {

			if (value < 50) continue;


			bool is_own = (((hue >= 340 || hue <= 20) && sat > 0.40) ||   // red
				(hue > 80 && hue <= 160 && sat > 0.30));    // green
			if (is_own) {
				blob_name = "robot";
				if (n_robot_blobs < 10) { robot_ic[n_robot_blobs] = ic; robot_jc[n_robot_blobs] = jc; n_robot_blobs++; }
				R = 0; G = 255; B = 0;
			}
			else {
				blob_name = "opponent";
				if (n_opp_blobs < 10) { opp_ic[n_opp_blobs] = ic; opp_jc[n_opp_blobs] = jc; n_opp_blobs++; }
				R = 0; G = 255; B = 255;
			}
		}
		else {
			// large blob = obstacle � classify by colour
			if (value < 50)                                      blob_name = "obstacle_black";
			else if ((hue >= 340 || hue <= 20) && sat > 0.4)         blob_name = "obstacle_red";
			else if (hue > 20 && hue <= 45 && sat > 0.5)           blob_name = "obstacle_orange";
			else if (hue > 80 && hue <= 160 && sat > 0.3)           blob_name = "obstacle_green";
			else if (hue > 190 && hue <= 260 && sat > 0.3)           blob_name = "obstacle_blue";
			else                                                      blob_name = "unknown";
			if (n_obs_detected < 10) { obs_ic[n_obs_detected] = ic; obs_jc[n_obs_detected] = jc; n_obs_detected++; }
			R = 255; G = 255; B = 255;
		}
	}
	if (n_robot_blobs >= 1) {
		robot_center_ic = 0; robot_center_jc = 0;
		for (int k = 0; k < n_robot_blobs; k++) {
			robot_center_ic += robot_ic[k];
			robot_center_jc += robot_jc[k];
		}
		robot_center_ic /= n_robot_blobs;
		robot_center_jc /= n_robot_blobs;
	}

	if (n_opp_blobs >= 1) {
		opp_center_ic = 0; opp_center_jc = 0;
		for (int k = 0; k < n_opp_blobs; k++) {
			opp_center_ic += opp_ic[k];
			opp_center_jc += opp_jc[k];
		}
		opp_center_ic /= n_opp_blobs;
		opp_center_jc /= n_opp_blobs;
	}

	return 0;
}


int features(image& a, image& rgb, image& label, int label_i,
	double& ic, double& jc, double& area,
	double& R_ave, double& G_ave, double& B_ave)
{
	ibyte* p, * pc;
	i2byte* pl;
	i4byte  i, j, k, width, height;
	double  mi, mj, m, rho, EPS = 1e-7, n;
	double  R_sum = 0, G_sum = 0, B_sum = 0;
	int     Rv, Gv, Bv;

	p = rgb.pdata;
	pl = (i2byte*)label.pdata;

	width = rgb.width;
	height = rgb.height;

	mi = mj = m = n = 0.0;

	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			if (pl[j * width + i] == label_i) {
				k = i + width * j;
				pc = p + 3 * k;
				Bv = *pc;
				Gv = *(pc + 1);
				Rv = *(pc + 2);
				R_sum += Rv;
				G_sum += Gv;
				B_sum += Bv;
				n++;
				rho = 1;
				m += rho;
				mi += rho * i;
				mj += rho * j;
			}
		}
	}
	ic = mi / (m + EPS);
	jc = mj / (m + EPS);
	R_ave = R_sum / (n + EPS);
	G_ave = G_sum / (n + EPS);
	B_ave = B_sum / (n + EPS);
	area = n;
	return 0;
}

void calculate_HSV(int R, int G, int B, double& hue, double& sat, double& value)
{
	int mx = R, mn = R;
	if (G > mx) mx = G; if (B > mx) mx = B;
	if (G < mn) mn = G; if (B < mn) mn = B;
	int delta = mx - mn;

	value = mx;
	if (delta == 0) sat = 0.0; else sat = (double)delta / value;

	double H;
	if (delta == 0) H = 0;
	else if (mx == R)    H = (double)(G - B) / delta;
	else if (mx == G)    H = (double)(B - R) / delta + 2;
	else                 H = (double)(R - G) / delta + 4;

	hue = 60 * H;
	if (hue < 0) hue += 360;
}

// MECH 472/663 - Player 1 OFFENCE (v2)
// Chase the opponent robot and hit it with the laser.
// mode = 1 (two player, player #1 controls robot_A)
//
// Vision: HSV-based thresholding (from prof's assignment 7)
//   - saturation > 0.20 or value < 50 = foreground
//   - size-based blob classification: small = robot marker, large = obstacle
//   - proximity-based robot/opponent assignment using previous frame centres
//
// State machine: SEARCH -> CHASE -> FIRE
//
// All detection is vision-based. No simulator state (S1->) is read
// except for own heading as fallback (same as naim's approach).
//
// Pipeline: copy -> lowpass_filter -> scale -> HSV threshold
//           -> erode x2 -> dialate x2 -> label_image -> detect_objects

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

extern robot_system *S1;

// ============================================================
// FORWARD DECLARATIONS
// ============================================================

int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave);

void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value);

int detect_objects(image &a, image &rgb0, image &label, int nlabels,
                   double robot_ic[], double robot_jc[], int &n_robot_blobs,
                   double &robot_center_ic, double &robot_center_jc,
                   double opp_ic[], double opp_jc[], int &n_opp_blobs,
                   double &opp_center_ic, double &opp_center_jc,
                   double obs_ic[], double obs_jc[], int &n_obs_detected,
                   image &rgb, int frame_count);


// ============================================================
// draw_line_rgb -- Bresenham line for visual overlays
// ============================================================
void draw_line_rgb(image &img, int x0, int y0, int x1, int y1,
                   int R, int G, int B)
{
	int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
	int dy = -abs(y1-y0), sy = y0<y1 ? 1 : -1;
	int err = dx+dy, e2;
	for (;;) {
		if (x0 >= 0 && x0 < img.width && y0 >= 0 && y0 < img.height)
			draw_point_rgb(img, x0, y0, R, G, B);
		if (x0 == x1 && y0 == y1) break;
		e2 = 2*err;
		if (e2 >= dy) { err += dy; x0 += sx; }
		if (e2 <= dx) { err += dx; y0 += sy; }
	}
}


// ============================================================
// MAIN
// ============================================================

int main()
{
	const double PI = 3.14159265;

	// ---- simulation setup (must match player2) ----
	double width1 = 640, height1 = 480;
	const int N_obs = 2;
	double x_obs[N_obs] = { 320, 320 };
	double y_obs[N_obs] = { 220, 220 };
	char obstacle_file[N_obs][S_MAX] = {
		"obstacle_black.bmp", "obstacle_green.bmp"
	};
	double D = 121.0, Lx = 31.0, Ly = 0.0, Ax = 37.0, Ay = 0.0;
	double alpha_max = PI / 2.0;
	int n_robot = 2;

	int pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
	double max_speed = 120.0;

	// ---- vision tracking variables ----
	double robot_ic[10], robot_jc[10];
	int    n_robot_blobs;
	double robot_center_ic, robot_center_jc;

	double opp_ic[10], opp_jc[10];
	int    n_opp_blobs;
	double opp_center_ic, opp_center_jc;

	double obs_ic[10], obs_jc[10];
	int    n_obs_detected;

	// ---- laser state ----
	int fired = 0;

	// ---- state machine: 0=SEARCH  1=CHASE  2=FIRE ----
	int state = 0;
	static const char *snames[] = {"SEARCH", "CHASE", "FIRE"};

	// ---- tuning constants ----
	const double FIRE_DIST   = 350.0;
	const double STOP_DIST   = 160.0;
	const double BACKUP_DIST = 120.0;
	const double ALIGN_TOL   = 0.10;    // rad: heading error to fire
	const double OBS_AVOID   = 120.0;   // obstacle avoidance triggerB
	const double OBS_EMERG  =  40.0;   // emergency backup zone
	const int    WALL_MARGIN = 50;
	const int    LOST_THRESH = 25;

	int width = 640, height = 480;

	cout << "\n=== PLAYER 1 - OFFENCE (v2) ===";
	cout << "\nVision-only detection (no simulator state)";
	cout << "\nState machine: SEARCH -> CHASE -> FIRE";
	cout << "\nPress space to begin.";
	pause();

	activate_vision();
	activate_simulation(width1, height1,
		x_obs, y_obs, N_obs,
		"robot_A.bmp", "robot_B.bmp", "background.bmp",
		obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);

	set_simulation_mode(1);
	set_robot_position(500, 240, PI);
	set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

	// ---- allocate images ----
	image rgb, a, b, rgb0, label;

	rgb.type   = RGB_IMAGE;    rgb.width  = width; rgb.height = height;
	a.type     = GREY_IMAGE;   a.width    = width; a.height   = height;
	b.type     = GREY_IMAGE;   b.width    = width; b.height   = height;
	rgb0.type  = RGB_IMAGE;    rgb0.width = width; rgb0.height = height;
	label.type = LABEL_IMAGE;  label.width = width; label.height = height;

	allocate_image(rgb);
	allocate_image(a);
	allocate_image(b);
	allocate_image(rgb0);
	allocate_image(label);

	// seed centre estimates with initial positions
	robot_center_ic = 500;  robot_center_jc = 240;
	opp_center_ic   = 140;  opp_center_jc   = 240;

	int nlabels;
	static int frame_count = 0;
	int lost_frames = 0;

	wait_for_player();

	double tc0 = high_resolution_time(), tc;

	// ============================================================
	// MAIN CONTROL LOOP
	// ============================================================

	while (1) {

		acquire_image_sim(rgb);
		tc = high_resolution_time() - tc0;
		frame_count++;

		copy(rgb, rgb0);

		// ============================================================
		// VISION PIPELINE (HSV-based, from prof's assignment 7)
		// ============================================================
		copy(rgb, a);
		lowpass_filter(a, b); copy(b, a);
		scale(a, b);          copy(b, a);

		// HSV threshold
		{
			ibyte *prgb = rgb0.pdata;
			ibyte *pb   = b.pdata;
			int npix = width * height;
			double h, s, v;
			for (int k = 0; k < npix; k++, prgb += 3) {
				int Bk = prgb[0], Gk = prgb[1], Rk = prgb[2];
				calculate_HSV(Rk, Gk, Bk, h, s, v);
				if (s > 0.20 || v < 50) pb[k] = 255;
				else                    pb[k] = 0;
			}
		}
		copy(b, a);

		// morphological cleaning
		erode(a, b);   copy(b, a);
		erode(a, b);   copy(b, a);
		dialate(a, b); copy(b, a);
		dialate(a, b); copy(b, a);

		label_image(a, label, nlabels);

		detect_objects(a, rgb0, label, nlabels,
		               robot_ic, robot_jc, n_robot_blobs,
		               robot_center_ic, robot_center_jc,
		               opp_ic, opp_jc, n_opp_blobs,
		               opp_center_ic, opp_center_jc,
		               obs_ic, obs_jc, n_obs_detected,
		               rgb, frame_count);

		// ============================================================
		// ROBOT HEADING ESTIMATION
		// ============================================================
		// own heading from vision blob pair (resolve 180deg ambiguity
		// using direction toward opponent). fallback to sim heading.

		double rx = robot_center_ic;
		double ry = robot_center_jc;
		double sim_theta = S1->P[1]->x[1]; // fallback only

		// direction toward opponent for ambiguity resolution
		double dir_to_opp = atan2(opp_center_jc - ry, opp_center_ic - rx);

		double rtheta;
		if (n_robot_blobs >= 2) {
			double raw = atan2(robot_jc[1] - robot_jc[0],
			                   robot_ic[1] - robot_ic[0]);
			double err1 = dir_to_opp - raw;
			while (err1 >  PI) err1 -= 2*PI;
			while (err1 < -PI) err1 += 2*PI;
			double err2 = dir_to_opp - (raw + PI);
			while (err2 >  PI) err2 -= 2*PI;
			while (err2 < -PI) err2 += 2*PI;
			rtheta = (fabs(err1) <= fabs(err2)) ? raw : raw + PI;
		} else {
			rtheta = sim_theta;
		}

		// robot front position
		const double front_offset = 65.0;
		double fx = rx + front_offset * cos(rtheta);
		double fy = ry + front_offset * sin(rtheta);

		// ============================================================
		// OPPONENT HEADING ESTIMATION
		// ============================================================
		double opp_heading = 0.0;
		int heading_valid = 0;
		if (n_opp_blobs >= 2) {
			// front marker = blob closer to us
			double da = (opp_ic[0]-rx)*(opp_ic[0]-rx) + (opp_jc[0]-ry)*(opp_jc[0]-ry);
			double db = (opp_ic[1]-rx)*(opp_ic[1]-rx) + (opp_jc[1]-ry)*(opp_jc[1]-ry);
			double front_ic, front_jc, rear_ic, rear_jc;
			if (da <= db) {
				front_ic = opp_ic[0]; front_jc = opp_jc[0];
				rear_ic  = opp_ic[1]; rear_jc  = opp_jc[1];
			} else {
				front_ic = opp_ic[1]; front_jc = opp_jc[1];
				rear_ic  = opp_ic[0]; rear_jc  = opp_jc[0];
			}
			opp_heading = atan2(front_jc - rear_jc, front_ic - rear_ic);
			heading_valid = 1;

			draw_point_rgb(rgb, (int)front_ic, (int)front_jc, 255, 0, 0);
			draw_point_rgb(rgb, (int)rear_ic,  (int)rear_jc,  0, 255, 255);
		}

		// ---- detection status ----
		bool found = (n_opp_blobs >= 1);
		if (found) lost_frames = 0;
		else       lost_frames++;

		// ============================================================
		// GEOMETRY
		// ============================================================
		double nav_dx  = opp_center_ic - rx;
		double nav_dy  = opp_center_jc - ry;
		double dist    = sqrt(nav_dx*nav_dx + nav_dy*nav_dy);
		double nav_ang = atan2(nav_dy, nav_dx);

		// heading error to opponent
		double nav_err = nav_ang - rtheta;
		while (nav_err >  PI) nav_err -= 2*PI;
		while (nav_err < -PI) nav_err += 2*PI;

		// aim error (same as nav for single-blob, uses front marker for paired)
		double aim_err = nav_err;

		// ============================================================
		// OBSTACLE AVOIDANCE (vision-detected obstacles)
		// ============================================================
		// check both robot centre and front point against obstacles
		// hysteresis: once engaged, keep avoiding until well clear
		static bool obs_engaged = false;
		bool near_obs = false;
		bool obs_emerg = false;
		double obs_min_d = 1e9;
		int obs_closest_k = -1;
		double trigger = obs_engaged ? (OBS_AVOID + 50.0) : OBS_AVOID;
		for (int k = 0; k < n_obs_detected; k++) {
			// check centre
			double odx = rx - obs_ic[k];
			double ody = ry - obs_jc[k];
			double od  = sqrt(odx*odx + ody*ody);
			// check front
			double fdx = fx - obs_ic[k];
			double fdy = fy - obs_jc[k];
			double fd  = sqrt(fdx*fdx + fdy*fdy);
			// use whichever is closer
			double closest = (fd < od) ? fd : od;
			if (closest < obs_min_d) { obs_min_d = closest; obs_closest_k = k; }
			if (closest < trigger)   near_obs = true;
			if (closest < OBS_EMERG) obs_emerg = true;
		}
		obs_engaged = near_obs;

		// ---- wall avoidance with hysteresis ----
		// check both centre and front point against walls
		static int wall_frames = 0;
		{
			bool hit_wall = (rx < WALL_MARGIN || rx > width - WALL_MARGIN ||
			                 ry < WALL_MARGIN || ry > height - WALL_MARGIN ||
			                 fx < WALL_MARGIN || fx > width - WALL_MARGIN ||
			                 fy < WALL_MARGIN || fy > height - WALL_MARGIN);
			if (hit_wall) wall_frames = 40;
			else if (wall_frames > 0) wall_frames--;
		}
		bool near_wall = (wall_frames > 0);

		// ============================================================
		// LINE-OF-SIGHT CHECK (vision-detected obstacles)
		// ============================================================
		bool clear_shot = true;
		if (found) {
			for (int k = 0; k < n_obs_detected; k++) {
				double ox = obs_ic[k], oy = obs_jc[k];
				double dx = opp_center_ic - rx, dy = opp_center_jc - ry;
				double seg_len = sqrt(dx*dx + dy*dy);
				if (seg_len < 1.0) continue;
				double ux = dx/seg_len, uy = dy/seg_len;
				double t = (ox - rx)*ux + (oy - ry)*uy;
				if (t < 0 || t > seg_len) continue;
				double px = rx + t*ux, py = ry + t*uy;
				double perp = sqrt((ox-px)*(ox-px) + (oy-py)*(oy-py));
				if (perp < 70.0) { clear_shot = false; break; }
			}
		}

		// ============================================================
		// STATE TRANSITIONS
		// ============================================================
		if (lost_frames > LOST_THRESH) {
			state = 0;
		} else if (found && state == 0) {
			state = 1;
		} else if (found && state == 1 && fabs(aim_err) < ALIGN_TOL &&
		           ((clear_shot && dist < FIRE_DIST) || dist < STOP_DIST)) {
			state = 2;
		} else if (state == 2 && fabs(aim_err) > ALIGN_TOL * 2.5) {
			state = 1;
		}

		// ============================================================
		// DRIVE DECISIONS
		// ============================================================
		// priority: wall > obstacle > state machine

		if (near_wall) {
			double err_c = atan2(height/2.0 - ry, width/2.0 - rx) - rtheta;
			while (err_c >  PI) err_c -= 2*PI;
			while (err_c < -PI) err_c += 2*PI;
			if (fabs(err_c) > 0.3) {
				int sign = (err_c >= 0) ? 1 : -1;
				pw_l = 1500 + sign * 175;
				pw_r = 1500 + sign * 175;
			} else {
				int steer = (int)(120.0 * err_c);
				pw_l = 1500 - 200 + steer;
				pw_r = 1500 + 200 + steer;
			}

		} else if (near_obs && obs_closest_k >= 0) {
			// drive around obstacle using cross-product side selection
			double away_a = atan2(ry - obs_jc[obs_closest_k],
			                      rx - obs_ic[obs_closest_k]);

			if (obs_emerg) {
				// emergency: too close, drive directly away
				double err_away = away_a - rtheta;
				while (err_away >  PI) err_away -= 2*PI;
				while (err_away < -PI) err_away += 2*PI;
				if (fabs(err_away) > PI/2.0) {
					// facing obstacle, back up
					pw_l = 1500 + 200;
					pw_r = 1500 - 200;
				} else {
					int steer = (int)(120.0 * err_away);
					pw_l = 1500 - 250 + steer;
					pw_r = 1500 + 250 + steer;
				}
			} else {
				// normal: steer around obstacle toward opponent
				static int avoid_side = 0;
				static int avoid_frames = 0;
				if (avoid_frames <= 0) {
					double obs_dx = obs_ic[obs_closest_k] - rx;
					double obs_dy = obs_jc[obs_closest_k] - ry;
					double opp_dx = opp_center_ic - rx;
					double opp_dy = opp_center_jc - ry;
					double cross = obs_dx * opp_dy - obs_dy * opp_dx;
					avoid_side = (cross > 0) ? 1 : -1;
					avoid_frames = 60;
				}
				avoid_frames--;
				double side_a = away_a + avoid_side * PI / 3.0;
				double err_side = side_a - rtheta;
				while (err_side >  PI) err_side -= 2*PI;
				while (err_side < -PI) err_side += 2*PI;
				int steer = (int)(120.0 * err_side);
				pw_l = 1500 - 200 + steer;
				pw_r = 1500 + 200 + steer;
			}

		} else {
			switch (state) {

			case 0: // SEARCH: slow CCW spin
				pw_l = 1500 + 175;
				pw_r = 1500 + 175;
				break;

			case 1: // CHASE
				if (dist < BACKUP_DIST) {
					// back away
					pw_l = 1500 + 200;
					pw_r = 1500 - 200;
				} else if (fabs(nav_err) > PI/3.0) {
					// aim error too large, rotate in place first
					int sign = (nav_err >= 0) ? 1 : -1;
					pw_l = 1500 + sign * 175;
					pw_r = 1500 + sign * 175;
				} else {
					// drive toward opponent with proportional steering
					int steer = (int)(120.0 * nav_err);
					pw_l = 1500 - 200 + steer;
					pw_r = 1500 + 200 + steer;
				}
				break;

			case 2: // FIRE: fine-tune aim
				if (dist < BACKUP_DIST) {
					pw_l = 1500 + 100;
					pw_r = 1500 - 100;
				} else if (dist > STOP_DIST) {
					if (fabs(aim_err) > 0.5) {
						int sign = (aim_err >= 0) ? 1 : -1;
						pw_l = 1500 + sign * 175;
						pw_r = 1500 + sign * 175;
					} else {
						int steer = (int)(120.0 * aim_err);
						pw_l = 1500 - 100 + steer;
						pw_r = 1500 + 100 + steer;
					}
				} else {
					int sign = (aim_err >= 0) ? 1 : -1;
					pw_l = 1500 + sign * 175;
					pw_r = 1500 + sign * 175;
				}
				break;
			}
		}

		// clamp servos
		if (pw_l > 2000) pw_l = 2000;
		if (pw_l < 1000) pw_l = 1000;
		if (pw_r > 2000) pw_r = 2000;
		if (pw_r < 1000) pw_r = 1000;

		// ============================================================
		// FIRE LASER: ONE SHOT
		// ============================================================
		if (!fired && tc > 3.0 && (state == 1 || state == 2) &&
		    fabs(aim_err) < ALIGN_TOL && clear_shot && dist < FIRE_DIST)
		{
			fired = 1;
			laser = 1; pw_laser = 1500;
			pw_l = 1500; pw_r = 1500;

			// draw overlays on freeze frame
			{
				int hx = (int)(rx + 50*cos(rtheta));
				int hy = (int)(ry + 50*sin(rtheta));
				draw_line_rgb(rgb, (int)rx, (int)ry, hx, hy, 0, 255, 0);
			}
			if (found) {
				draw_line_rgb(rgb, (int)rx, (int)ry,
				              (int)opp_center_ic, (int)opp_center_jc, 255, 0, 255);
			}

			set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
			view_rgb_image(rgb, 1);

			cout << "\n\n========== LASER FIRED ==========";
			cout << "\n  Time:     " << (int)tc << "s";
			cout << "\n  Distance: " << (int)dist << " px";
			cout << "\n  Aim err:  " << (int)(aim_err*180.0/PI) << " deg";
			cout << "\n  Opp blobs: " << n_opp_blobs;
			cout << "\n=================================";
			cout << "\nPress space to exit.";
			pause();
			break;
		}

		// ---- visual overlays ----
		{
			int hx = (int)(rx + 50*cos(rtheta));
			int hy = (int)(ry + 50*sin(rtheta));
			draw_line_rgb(rgb, (int)rx, (int)ry, hx, hy, 0, 255, 0);
		}
		if (found) {
			draw_line_rgb(rgb, (int)rx, (int)ry,
			              (int)opp_center_ic, (int)opp_center_jc, 255, 0, 255);
		}
		// wall margin box
		draw_line_rgb(rgb, WALL_MARGIN, WALL_MARGIN, width-WALL_MARGIN, WALL_MARGIN, 255, 255, 0);
		draw_line_rgb(rgb, width-WALL_MARGIN, WALL_MARGIN, width-WALL_MARGIN, height-WALL_MARGIN, 255, 255, 0);
		draw_line_rgb(rgb, width-WALL_MARGIN, height-WALL_MARGIN, WALL_MARGIN, height-WALL_MARGIN, 255, 255, 0);
		draw_line_rgb(rgb, WALL_MARGIN, height-WALL_MARGIN, WALL_MARGIN, WALL_MARGIN, 255, 255, 0);

		set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
		view_rgb_image(rgb, 1);

		// ---- console output ----
		const char *drive_reason = "SM";
		if (near_wall) drive_reason = "WALL";
		else if (near_obs) drive_reason = "OBS";

		if (frame_count % 60 == 0) {
			cout << "\n[t=" << (int)tc << "s] " << snames[state]
			     << " drv=" << drive_reason
			     << "  me=(" << (int)rx << "," << (int)ry << ")"
			     << "  opp=(" << (int)opp_center_ic << "," << (int)opp_center_jc << ")"
			     << "  d=" << (int)dist
			     << " aim=" << (int)(aim_err*180.0/PI) << "deg"
			     << "  pw=" << pw_l << "/" << pw_r;
			if (obs_closest_k >= 0)
				cout << "  obs" << obs_closest_k << "_d=" << (int)obs_min_d;
			cout << " blobs: r=" << n_robot_blobs
			     << " o=" << n_opp_blobs
			     << " obs=" << n_obs_detected;
		}

		if (KEY('X')) break;
	}

	free_image(rgb);
	free_image(a);
	free_image(b);
	free_image(rgb0);
	free_image(label);
	deactivate_vision();
	deactivate_simulation();
	cout << "\ndone.\n";
	return 0;
}


// ============================================================
// detect_objects()
// ============================================================
int detect_objects(image &a, image &rgb0, image &label, int nlabels,
                   double robot_ic[], double robot_jc[], int &n_robot_blobs,
                   double &robot_center_ic, double &robot_center_jc,
                   double opp_ic[], double opp_jc[], int &n_opp_blobs,
                   double &opp_center_ic, double &opp_center_jc,
                   double obs_ic[], double obs_jc[], int &n_obs_detected,
                   image &rgb, int frame_count)
{
	const double min_area       = 200.0;
	const double size_threshold = 2500.0;

	double ic, jc, area, R_ave, G_ave, B_ave;
	double hue, sat, value;
	int    R, G, B;
	const char *blob_name;

	n_robot_blobs  = 0;
	n_opp_blobs    = 0;
	n_obs_detected = 0;

	for (int i_label = 1; i_label <= nlabels; i_label++) {

		features(a, rgb0, label, i_label, ic, jc, area, R_ave, G_ave, B_ave);

		if (area < min_area) continue;

		calculate_HSV((int)R_ave, (int)G_ave, (int)B_ave, hue, sat, value);

		if (area < size_threshold) {

			double d_robot = (ic - robot_center_ic)*(ic - robot_center_ic)
			               + (jc - robot_center_jc)*(jc - robot_center_jc);
			double d_opp   = (ic - opp_center_ic)*(ic - opp_center_ic)
			               + (jc - opp_center_jc)*(jc - opp_center_jc);

			if (d_robot <= d_opp) {
				blob_name = "robot";
				if (n_robot_blobs < 10) {
					robot_ic[n_robot_blobs] = ic;
					robot_jc[n_robot_blobs] = jc;
					n_robot_blobs++;
				}
				R = 0; G = 255; B = 0;
			} else {
				blob_name = "opponent";
				if (n_opp_blobs < 10) {
					opp_ic[n_opp_blobs] = ic;
					opp_jc[n_opp_blobs] = jc;
					n_opp_blobs++;
				}
				R = 0; G = 255; B = 255;
			}

		} else {

			if (value < 50) {
				blob_name = "obstacle_black";
			} else if ((hue >= 340 || hue <= 20) && sat > 0.4) {
				blob_name = "obstacle_red";
			} else if (hue > 20 && hue <= 45 && sat > 0.5) {
				blob_name = "obstacle_orange";
			} else if (hue > 80 && hue <= 160 && sat > 0.3) {
				blob_name = "obstacle_green";
			} else if (hue > 190 && hue <= 260 && sat > 0.3) {
				blob_name = "obstacle_blue";
			} else {
				blob_name = "unknown";
			}

			if (n_obs_detected < 10) {
				obs_ic[n_obs_detected] = ic;
				obs_jc[n_obs_detected] = jc;
				n_obs_detected++;
			}
			R = 255; G = 255; B = 255;
		}

		draw_point_rgb(rgb, (int)ic, (int)jc, R, G, B);

		if (frame_count == 1) {
			cout << "\nlabel " << i_label << " [" << blob_name << "]";
			cout << "  centroid: (" << (int)ic << ", " << (int)jc << ")";
			cout << "  HSV: (" << (int)hue << " deg, " << sat
			     << ", " << (int)value << ")";
			cout << "  area: " << (int)area;
		}
	}

	if (frame_count == 1) cout << "\nnlabels = " << nlabels;

	if (n_robot_blobs >= 1) {
		robot_center_ic = 0; robot_center_jc = 0;
		for (int k = 0; k < n_robot_blobs; k++) {
			robot_center_ic += robot_ic[k];
			robot_center_jc += robot_jc[k];
		}
		robot_center_ic /= n_robot_blobs;
		robot_center_jc /= n_robot_blobs;
		draw_point_rgb(rgb, (int)robot_center_ic, (int)robot_center_jc,
		               255, 0, 0);
	}

	if (n_opp_blobs >= 1) {
		opp_center_ic = 0; opp_center_jc = 0;
		for (int k = 0; k < n_opp_blobs; k++) {
			opp_center_ic += opp_ic[k];
			opp_center_jc += opp_jc[k];
		}
		opp_center_ic /= n_opp_blobs;
		opp_center_jc /= n_opp_blobs;
		draw_point_rgb(rgb, (int)opp_center_ic, (int)opp_center_jc,
		               255, 0, 255);
	}

	return 0;
}


// ============================================================
// features()
// ============================================================
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

	if (rgb.height != label.height || rgb.width != label.width) {
		cout << "\nerror in features: sizes of rgb, label are not the same!";
		return 1;
	}
	if (rgb.type != RGB_IMAGE || label.type != LABEL_IMAGE) {
		cout << "\nerror in features: input types are not valid!";
		return 1;
	}

	p  = rgb.pdata;
	pl = (i2byte *)label.pdata;

	width  = rgb.width;
	height = rgb.height;

	mi = mj = m = n = 0.0;

	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			if (pl[j*width+i] == label_i) {
				k  = i + width*j;
				pc = p + 3*k;
				Bv = *pc;
				Gv = *(pc+1);
				Rv = *(pc+2);
				R_sum += Rv;
				G_sum += Gv;
				B_sum += Bv;
				n++;
				rho = 1;
				m  += rho;
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


// ============================================================
// calculate_HSV()
// ============================================================
void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value)
{
	int max, min, delta;
	double H;

	max = min = R;
	if (G > max) max = G;
	if (B > max) max = B;
	if (G < min) min = G;
	if (B < min) min = B;

	delta = max - min;
	value = max;

	if (delta == 0) sat = 0.0;
	else            sat = (double)delta / value;

	if (delta == 0)      H = 0;
	else if (max == R)   H = (double)(G - B) / delta;
	else if (max == G)   H = (double)(B - R) / delta + 2;
	else                 H = (double)(R - G) / delta + 4;

	hue = 60 * H;
	if (hue < 0) hue += 360;
}

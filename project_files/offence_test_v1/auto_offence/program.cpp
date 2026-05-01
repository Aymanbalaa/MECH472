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

int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave);

void calculate_HSV(int R, int G, int B,
                   double &hue, double &sat, double &value);

int detect_objects(image &a, image &rgb0, image &label, int nlabels,
                   double robot_ic[], double robot_jc[], int &n_robot_blobs,
                   double &robot_center_ic, double &robot_center_jc,
                   double opp_ic[], double opp_jc[], int &n_opp_blobs,
                   double &opp_center_ic, double &opp_center_jc,
                   double obs_ic[], double obs_jc[], int &n_obs_detected,
                   image &rgb, int frame_count,
                   const double obs_seed_x[], const double obs_seed_y[],
                   int n_obs_seed);


// TO BE REMOVED , WAS FOR TESTING
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

int main()
{
	const double PI = 3.14159265;

	double width1 = 640, height1 = 480;
	const int N_obs = 2;

	double x_obs[N_obs] = { 320, 440.0 };
	double y_obs[N_obs] = { 240, 300.0 };
	char obstacle_file[N_obs][S_MAX] = {
		"obstacle_black.bmp", "obstacle_green.bmp"
	};

	double D = 121.0, Lx = 31.0, Ly = 0.0, Ax = 37.0, Ay = 0.0;
	double alpha_max = PI / 2.0;
	int n_robot = 2;

	int pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
	double max_speed = 200.0;

	// ---- vision tracking buffers ----
	double robot_ic[10], robot_jc[10];
	int    n_robot_blobs;
	double robot_center_ic, robot_center_jc;

	double opp_ic[10], opp_jc[10];
	int    n_opp_blobs;
	double opp_center_ic, opp_center_jc;

	double obs_ic[10], obs_jc[10];
	int    n_obs_detected;

	//laser state
	int fired = 0;

	//state machine: 0=SEARCH 1=CHASE 2=ATTACK

	int state = 0;
	static const char *snames[] = { "SEARCH", "CHASE", "ATTACK" };

	// ---- tuning ----
	const double FIRE_DIST   = 500.0;   // px: in-range to attempt fire
	const double STOP_DIST   = 160.0;   // px: stop closing in
	const double BACKUP_DIST = 120.0;   // px: too close, back away
	const double ALIGN_TOL   = 0.10;    // rad: required aim accuracy
	const double ATTACK_TOL  = 0.06;    // rad: tighter aim used inside ATTACK fine spin
	const double OBS_AVOID   = 70.0;   // px: obstacle avoidance trigger
	const double OBS_EMERG   =  40.0;   // px: emergency backup zone
	const int    WALL_MARGIN = 60;      // px: wall standoff (was 30 - too loose, robot reached x=627 before triggering)
	const int    LOST_THRESH = 25;      // frames without opp -> SEARCH
	const int    HEADING_STALE_THRESH = 12; // frames before forcing recovery spin
	const double LOS_BLOCK_R = 70.0;    // perp dist threshold for blocked shot
	const double FRONT_OFFSET = 65.0;   // px: visual front marker length

	int width = 640, height = 480;

	cout << "\nAUTO OFFENCE (player 1) ";
	cout << "\nPress space to begin.";
	pause();

	activate_vision();
	activate_simulation(width1, height1,
		x_obs, y_obs, N_obs,
		"robot_A.bmp", "robot_B.bmp", "background.bmp",
		obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);

	set_simulation_mode(1);
	set_robot_position(500, 100, PI);          // robot_A start
	set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

	// ---- allocate images ----
	image rgb, a, b, rgb0, label;
	rgb.type   = RGB_IMAGE;    rgb.width   = width; rgb.height   = height;
	a.type     = GREY_IMAGE;   a.width     = width; a.height     = height;
	b.type     = GREY_IMAGE;   b.width     = width; b.height     = height;
	rgb0.type  = RGB_IMAGE;    rgb0.width  = width; rgb0.height  = height;
	label.type = LABEL_IMAGE;  label.width = width; label.height = height;
	allocate_image(rgb);
	allocate_image(a);
	allocate_image(b);
	allocate_image(rgb0);
	allocate_image(label);

	// seed centroid estimates with the starting positions
	robot_center_ic = 500;  robot_center_jc = 100;
	opp_center_ic   = 140;  opp_center_jc   = 400;

	int nlabels;
	int frame_count = 0;
	int lost_frames = 0;

	// heading defines
	double rtheta_last = PI;       // last DETCETED heading
	int    heading_stale_frames = 0;
	bool   heading_ever_set = false;

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

		//vision pipeline, steps similar to assignement 7 
		copy(rgb, a);
		lowpass_filter(a, b); copy(b, a);
		scale(a, b);          copy(b, a);
		// HSV 
		{
			ibyte *prgb = rgb0.pdata;
			ibyte *pb   = b.pdata;
			int npix = width * height;
			double h, s, v;
			for (int k = 0; k < npix; k++, prgb += 3) {
				int Bk = prgb[0], Gk = prgb[1], Rk = prgb[2];
				calculate_HSV(Rk, Gk, Bk, h, s, v);
				pb[k] = (s > 0.20 || v < 50) ? 255 : 0;
			}
		}
		copy(b, a);

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
		               rgb, frame_count,
		               x_obs, y_obs, N_obs);

		// heading calcul 
		double rx = robot_center_ic;
		double ry = robot_center_jc;

		// direction toward opponent disambiguates the 180 deg in atan2
		double dir_to_opp = atan2(opp_center_jc - ry,
		                          opp_center_ic - rx);

		double rtheta;
		if (n_robot_blobs >= 2) {
			// uses previous dir and previous heading values to always define front or back otherwise through the two centroids there is 180 degree confusion
			double raw = atan2(robot_jc[1] - robot_jc[0],
			                   robot_ic[1] - robot_ic[0]);
			double ref = heading_ever_set ? rtheta_last : dir_to_opp;
			double err1 = ref - raw;
			while (err1 >  PI) err1 -= 2*PI;
			while (err1 < -PI) err1 += 2*PI;
			double err2 = ref - (raw + PI);
			while (err2 >  PI) err2 -= 2*PI;
			while (err2 < -PI) err2 += 2*PI;
			rtheta = (fabs(err1) <= fabs(err2)) ? raw : raw + PI;

			rtheta_last = rtheta;
			heading_stale_frames = 0;
			heading_ever_set = true;
		} else if (heading_ever_set) {
			// fallbacks
			rtheta = rtheta_last;
			heading_stale_frames++;
		} else {
			// fallback
			rtheta = dir_to_opp;
		}

		// robot front position 
		double fx = rx + FRONT_OFFSET * cos(rtheta);
		double fy = ry + FRONT_OFFSET * sin(rtheta);

		// is opponent detected?
		bool found = (n_opp_blobs >= 1);
		if (found) lost_frames = 0;
		else       lost_frames++;

		double nav_dx  = opp_center_ic - rx;
		double nav_dy  = opp_center_jc - ry;
		double dist    = sqrt(nav_dx*nav_dx + nav_dy*nav_dy);
		double nav_ang = atan2(nav_dy, nav_dx);

		double aim_err = nav_ang - rtheta;
		while (aim_err >  PI) aim_err -= 2*PI;
		while (aim_err < -PI) aim_err += 2*PI;

		// OBSTACLE AVOIDANCE 
		static bool obs_engaged = false;
		bool   near_obs    = false;
		bool   obs_emerg   = false;
		double obs_min_d   = 1e9;
		int    obs_closest_k = -1;
		double trigger = obs_engaged ? (OBS_AVOID + 50.0) : OBS_AVOID;
		for (int k = 0; k < n_obs_detected; k++) {
			double odx = rx - obs_ic[k];
			double ody = ry - obs_jc[k];
			double od  = sqrt(odx*odx + ody*ody);
			double fdx = fx - obs_ic[k];
			double fdy = fy - obs_jc[k];
			double fd  = sqrt(fdx*fdx + fdy*fdy);
			double closest = (fd < od) ? fd : od;
			if (closest < obs_min_d) { obs_min_d = closest; obs_closest_k = k; }
			if (closest < trigger)   near_obs = true;
			if (closest < OBS_EMERG) obs_emerg = true;
		}
		obs_engaged = near_obs;

		// WALL avoidance
		static int wall_frames = 0;
		{
			bool hit = (rx < WALL_MARGIN || rx > width - WALL_MARGIN ||
			            ry < WALL_MARGIN || ry > height - WALL_MARGIN ||
			            fx < WALL_MARGIN || fx > width - WALL_MARGIN ||
			            fy < WALL_MARGIN || fy > height - WALL_MARGIN);
			if (hit) wall_frames = 40;
			else if (wall_frames > 0) wall_frames--;
		}
		bool near_wall = (wall_frames > 0);

		// Line of sight , any opponent between us and robot?
		// checks from laser to each marker as well as robot center

		double cand_tx[3], cand_ty[3], cand_err[3];
		bool   cand_clear[3];
		const char *cand_name[3];
		int    n_cand = 0;

		// always include centroid as the baseline target
		cand_tx[n_cand] = opp_center_ic;
		cand_ty[n_cand] = opp_center_jc;
		cand_name[n_cand] = "C";
		n_cand++;
		// add each detected opponent marker
		for (int k = 0; k < n_opp_blobs && n_cand < 3; k++) {
			cand_tx[n_cand] = opp_ic[k];
			cand_ty[n_cand] = opp_jc[k];
			cand_name[n_cand] = (k == 0) ? "M0" : "M1";
			n_cand++;
		}

		for (int c = 0; c < n_cand; c++) {
			// aim error from current heading to this target.
			// Measured from robot CENTRE since rtheta is the centre's
			double t_ang = atan2(cand_ty[c] - ry, cand_tx[c] - rx);
			double e = t_ang - rtheta;
			while (e >  PI) e -= 2*PI;
			while (e < -PI) e += 2*PI;
			cand_err[c] = e;

			// LOS test: any obstacle close to the laser-exit-to-target
			// segment blocks this candidate.
			bool blocked = false;
			if (found) {
				double dx = cand_tx[c] - fx;
				double dy = cand_ty[c] - fy;
				double seg_len = sqrt(dx*dx + dy*dy);
				if (seg_len > 1.0) {
					double ux = dx/seg_len, uy = dy/seg_len;
					for (int k = 0; k < n_obs_detected; k++) {
						double ox = obs_ic[k], oy = obs_jc[k];
						double t = (ox - fx)*ux + (oy - fy)*uy;
						if (t < 0 || t > seg_len) continue;
						double px = fx + t*ux, py = fy + t*uy;
						double perp = sqrt((ox-px)*(ox-px) + (oy-py)*(oy-py));
						if (perp < LOS_BLOCK_R) { blocked = true; break; }
					}
				}
			}
			cand_clear[c] = !blocked;
		}

		// picks the best target out of 3 if any are good enough
		int  best_aim = 0;
		double best_abs = 1e9;
		bool any_clear = false;
		for (int c = 0; c < n_cand; c++) {
			if (!cand_clear[c]) continue;
			if (fabs(cand_err[c]) < best_abs) {
				best_abs = fabs(cand_err[c]);
				best_aim = c;
				any_clear = true;
			}
		}

		bool   clear_shot      = any_clear;
		double attack_aim_err  = any_clear ? cand_err[best_aim] : aim_err;
		const char *attack_tgt = any_clear ? cand_name[best_aim] : "-";

		// ============================================================
		// STATE TRANSITIONS
		// ============================================================
		// Core rule: the moment LOS is clear AND opponent is in range,
		// switch to ATTACK so we actively rotate to aim - we DO NOT
		// wait for aim to drift into ALIGN_TOL by accident during CHASE.
		//
		// If heading has been stale for too long, force SEARCH so we
		// rotate and reacquire vision pose before doing anything else.
		bool heading_recoverable = (heading_stale_frames < HEADING_STALE_THRESH);
		bool can_attack_now = (found && clear_shot && dist < FIRE_DIST);

		// ATTACK persistence: clear_shot can flicker frame-to-frame as
		// the opp moves and an obstacle skims the LOS line. Without
		// hysteresis we drop out of ATTACK every blocked frame and
		// throw away all the aim work. Stay in ATTACK for up to 12
		// blocked frames before giving up - long enough to ride out
		// transient blockage, short enough that real cover gets us
		// chasing again.
		static int attack_block_frames = 0;
		if (state == 2 && !can_attack_now) attack_block_frames++;
		else                               attack_block_frames = 0;
		bool can_attack_sticky = can_attack_now ||
		                         (state == 2 && attack_block_frames < 12 &&
		                          found && dist < FIRE_DIST);

		if (lost_frames > LOST_THRESH || !heading_recoverable) {
			state = 0; // SEARCH
		} else if (found && state == 0) {
			state = 1; // SEARCH -> CHASE
		} else if (can_attack_now) {
			// fast-path: from CHASE or ATTACK, the moment LOS is clear
			// and target is in range, lock in ATTACK and rotate to aim
			state = 2;
		} else if (state == 2 && !can_attack_sticky) {
			// LOS has been blocked for a stretch (or target out of range)
			// and our hysteresis window has expired. Back to CHASE.
			state = 1;
		}

		// ============================================================
		// DRIVE DECISIONS
		// priority: stuck-escape > wall > obstacle > state machine
		// ============================================================
		// Stuck escape: when heading is stale AND we are near a wall,
		// the wall handler can't be trusted (it rotates against a stale
		// rtheta). Drive toward arena centre instead, using the dot
		// product of (rtheta_last, vector_to_centre) to pick whether
		// FORWARD or BACKWARD points centre-ward. This way we never
		// drive deeper into the wall the robot is against.
		static int stuck_escape_frames = 0;
		if (stuck_escape_frames <= 0 && near_wall && heading_stale_frames > 25) {
			stuck_escape_frames = 50;
		}

		if (stuck_escape_frames > 0) {
			// Pure position-based escape: integrate over the whole
			// commit window with the SAME drive command, picked once at
			// entry. We freeze the choice in static vars so a flipping
			// stale heading can't flip the drive frame-to-frame (which
			// was what made the robot oscillate in place).
			static int   esc_pw_l = 1500, esc_pw_r = 1500;
			static int   esc_chosen = 0;
			if (!esc_chosen) {
				double dx_c = (width  / 2.0) - rx;
				double dy_c = (height / 2.0) - ry;
				double centre_dot = cos(rtheta_last) * dx_c +
				                    sin(rtheta_last) * dy_c;
				if (centre_dot > 0) {
					esc_pw_l = 1500 - 220;
					esc_pw_r = 1500 + 220;
				} else {
					esc_pw_l = 1500 + 220;
					esc_pw_r = 1500 - 220;
				}
				esc_chosen = 1;
			}
			pw_l = esc_pw_l;
			pw_r = esc_pw_r;
			stuck_escape_frames--;
			if (stuck_escape_frames == 0) esc_chosen = 0;

		} else if (near_wall) {
			// turn toward arena centre
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

		} else if (state == 2 && clear_shot && !obs_emerg) {
			// ATTACK with a confirmed clear shot - skip OBS orbit even
			// if we're physically near an obstacle. The obstacle isn't
			// in the laser path (clear_shot proved that), so there's no
			// reason to walk around it. Stop and rotate to aim instead.
			// (Emergency-close obstacle still gets handled by the next
			// branch via the obs_emerg fall-through.)
			int rot_pw;
			double abs_err = fabs(attack_aim_err);
			if (abs_err > 0.5)        rot_pw = 200;
			else if (abs_err > 0.15)  rot_pw = 130;
			else                      rot_pw = 70;
			int sign = (attack_aim_err >= 0) ? 1 : -1;
			pw_l = 1500 + sign * rot_pw;
			pw_r = 1500 + sign * rot_pw;

		} else if ((near_obs || (found && !clear_shot && dist < FIRE_DIST)) && obs_closest_k >= 0) {
			// drive AROUND the obstacle (cross-product side commit)
			// while still ultimately heading for the opponent
			double away_a = atan2(ry - obs_jc[obs_closest_k],
			                      rx - obs_ic[obs_closest_k]);

			if (obs_emerg) {
				// too close - back away from obstacle
				double err_away = away_a - rtheta;
				while (err_away >  PI) err_away -= 2*PI;
				while (err_away < -PI) err_away += 2*PI;
				if (fabs(err_away) > PI/2.0) {
					pw_l = 1500 + 200;
					pw_r = 1500 - 200;
				} else {
					int steer = (int)(120.0 * err_away);
					pw_l = 1500 - 250 + steer;
					pw_r = 1500 + 250 + steer;
				}
			} else {
				// pick a side using cross-product (commit for 60 frames
				// to avoid oscillation), then drive past the obstacle
				static int avoid_side = 0;
				static int avoid_frames = 0;
				if (avoid_frames <= 0) {
					double obs_dx = obs_ic[obs_closest_k] - rx;
					double obs_dy = obs_jc[obs_closest_k] - ry;
					double opp_dx = opp_center_ic - rx;
					double opp_dy = opp_center_jc - ry;
					double cross = obs_dx * opp_dy - obs_dy * opp_dx;
					avoid_side = (cross > 0) ? 1 : -1;
					// Wall-aware override: if the chosen tangent direction
					// would push the robot off-screen within ~120 px, flip
					// to the other side. Stops the OBS state from happily
					// driving into a wall in the name of "going around".
					double ox = obs_ic[obs_closest_k];
					double oy = obs_jc[obs_closest_k];
					double check_aw = atan2(ry - oy, rx - ox);
					for (int trial = 0; trial < 2; trial++) {
						double t_a = check_aw + avoid_side * (PI / 2.0);
						double tx  = rx + 120.0 * cos(t_a);
						double ty  = ry + 120.0 * sin(t_a);
						bool off = (tx < WALL_MARGIN || tx > width - WALL_MARGIN ||
						            ty < WALL_MARGIN || ty > height - WALL_MARGIN);
						if (!off) break;
						avoid_side = -avoid_side;  // flip and re-test
					}
					avoid_frames = 80;
				}
				avoid_frames--;
				// PI/2 = tangent to the obstacle, makes the orbit
				// path actually skim the perimeter instead of cutting
				// outward to a 60-deg-from-radial line
				double side_a = away_a + avoid_side * (PI / 2.0);
				double err_side = side_a - rtheta;
				while (err_side >  PI) err_side -= 2*PI;
				while (err_side < -PI) err_side += 2*PI;

				// continuous-velocity steering: always move forward,
				// just slow down + steer harder when err is large.
				// No more stop-spin hysteresis. fwd ranges 200..80,
				// steer is saturated at +/- fwd/2 so the inside wheel
				// stops but never reverses (smooth pivot turn).
				double abs_err = fabs(err_side);
				int fwd = (int)(200.0 - 120.0 * (abs_err / PI));
				if (fwd < 80) fwd = 80;
				int steer = (int)(280.0 * err_side);
				int max_steer = fwd / 2;
				if (steer >  max_steer) steer =  max_steer;
				if (steer < -max_steer) steer = -max_steer;
				pw_l = 1500 - fwd + steer;
				pw_r = 1500 + fwd + steer;
			}

		} else {
			switch (state) {

			case 0: // SEARCH: slow CCW spin
				pw_l = 1500 + 175;
				pw_r = 1500 + 175;
				break;

			case 1: // CHASE
				// continuous-velocity steering, same shape as OBS:
				// always forward, scale fwd down + steer up with err.
				// No binary "spin in place when err > PI/3" anymore -
				// that was the source of stop-spin-go jerkiness.
				if (dist < BACKUP_DIST) {
					pw_l = 1500 + 200;
					pw_r = 1500 - 200;
				} else {
					double abs_err = fabs(aim_err);
					int fwd = (int)(200.0 - 130.0 * (abs_err / PI));
					if (fwd < 70) fwd = 70;
					int steer = (int)(280.0 * aim_err);
					int max_steer = fwd / 2;
					if (steer >  max_steer) steer =  max_steer;
					if (steer < -max_steer) steer = -max_steer;
					pw_l = 1500 - fwd + steer;
					pw_r = 1500 + fwd + steer;
				}
				break;

			case 2: // ATTACK: rotate-in-place to lock aim, then fire
				// Pure in-place rotation - the only translation we
				// allow is a tiny back-off if we're inside BACKUP_DIST,
				// so we don't ram the opponent while aiming.
				// Speed scales with aim error: fast slew when far off,
				// fine creep when nearly aligned (avoids overshoot).
				// Aim is at the BEST exposed target (centroid OR
				// marker), not always the centroid - lets us hit a
				// peeking marker when the body is behind cover.
				if (dist < BACKUP_DIST) {
					pw_l = 1500 + 150;
					pw_r = 1500 - 150;
				} else {
					int rot_pw;
					double abs_err = fabs(attack_aim_err);
					if (abs_err > 0.5)        rot_pw = 200; // hard slew
					else if (abs_err > 0.15)  rot_pw = 130; // medium
					else                      rot_pw = 70;  // fine, avoid overshoot
					int sign = (attack_aim_err >= 0) ? 1 : -1;
					pw_l = 1500 + sign * rot_pw;
					pw_r = 1500 + sign * rot_pw;
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
		// FIRE LASER (one-shot)
		// ============================================================
		// Fires only inside ATTACK state - which itself is only entered
		// when LOS is clear AND in range. So if state==2 and aim is
		// good, we have a guaranteed clean shot.
		//
		// Stability gate: require 3 consecutive frames of paired-robot
		// detection so a one-frame heading glitch can't trigger a shot.
		// This is a much shorter window than the old 5-frame gate so
		// it stops blocking legitimate shots while still rejecting
		// single-frame artifacts.
		static int stable_detect_frames = 0;
		if (n_robot_blobs == 2 && n_opp_blobs >= 1) stable_detect_frames++;
		else                                        stable_detect_frames = 0;

		// Tighter aim tolerance inside ATTACK (we have time to lock in)
		// vs looser tolerance in CHASE (snap shot of opportunity).
		double fire_tol = (state == 2) ? ATTACK_TOL : ALIGN_TOL;

		// Startup delay reduced from 3.0 -> 1.0 s. The original 3 s
		// was a holdover from the v2 controller before stability gates
		// existed. Now we have stable_detect_frames > 3 doing the
		// "is detection valid" job, so 1 s is plenty for the simulator
		// to settle - early shots at t=1s are now legal.
		if (!fired && tc > 1.0 && state == 2 &&
		    stable_detect_frames > 3 &&
		    fabs(attack_aim_err) < fire_tol && clear_shot && dist < FIRE_DIST)
		{
			fired = 1;
			laser = 1;
			pw_laser = 1500;
			pw_l = 1500; pw_r = 1500;

			{
				int hx = (int)(rx + 50*cos(rtheta));
				int hy = (int)(ry + 50*sin(rtheta));
				draw_line_rgb(rgb, (int)rx, (int)ry, hx, hy, 0, 255, 0);
			}
			if (found) {
				draw_line_rgb(rgb, (int)rx, (int)ry,
				              (int)opp_center_ic, (int)opp_center_jc,
				              255, 0, 255);
			}

			set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
			view_rgb_image(rgb, 1);

			cout << "\n\n========== LASER FIRED ==========";
			cout << "\n  Time:     " << (int)tc << " s";
			cout << "\n  Distance: " << (int)dist << " px";
			cout << "\n  Target:   " << attack_tgt
			     << "  (" << (int)cand_tx[best_aim] << ","
			     << (int)cand_ty[best_aim] << ")";
			cout << "\n  Aim err:  " << (int)(attack_aim_err*180.0/PI) << " deg";
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
			              (int)opp_center_ic, (int)opp_center_jc,
			              255, 0, 255);
		}
		// wall margin box
		draw_line_rgb(rgb, WALL_MARGIN, WALL_MARGIN,
		              width-WALL_MARGIN, WALL_MARGIN, 255, 255, 0);
		draw_line_rgb(rgb, width-WALL_MARGIN, WALL_MARGIN,
		              width-WALL_MARGIN, height-WALL_MARGIN, 255, 255, 0);
		draw_line_rgb(rgb, width-WALL_MARGIN, height-WALL_MARGIN,
		              WALL_MARGIN, height-WALL_MARGIN, 255, 255, 0);
		draw_line_rgb(rgb, WALL_MARGIN, height-WALL_MARGIN,
		              WALL_MARGIN, WALL_MARGIN, 255, 255, 0);

		set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
		view_rgb_image(rgb, 1);

		// ---- console status ----
		const char *drive_reason = "SM";
		if (near_wall)      drive_reason = "WALL";
		else if (near_obs)  drive_reason = "OBS";

		if (frame_count % 60 == 0) {
			const char *drv_show = drive_reason;
			if (stuck_escape_frames > 0) drv_show = "ESCAPE";
			cout << "\n[t=" << (int)tc << "s] " << snames[state]
			     << " drv=" << drv_show
			     << "  me=(" << (int)rx << "," << (int)ry << ")"
			     << "  opp=(" << (int)opp_center_ic << ","
			     << (int)opp_center_jc << ")"
			     << "  d=" << (int)dist
			     << " aim=" << (int)(aim_err*180.0/PI) << "deg"
			     << "  pw=" << pw_l << "/" << pw_r
			     << "  blobs r=" << n_robot_blobs
			     << " o=" << n_opp_blobs
			     << " obs=" << n_obs_detected;
			if (heading_stale_frames > 0)
				cout << " heading_stale=" << heading_stale_frames;
			if (clear_shot) cout << " [aim=" << attack_tgt << "]";
			else            cout << " [BLOCKED]";
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
// detect_objects -- classify each labelled blob as
// self / opponent / obstacle and return their centroids
// (small blob => robot marker; large blob => obstacle)
// ============================================================
int detect_objects(image &a, image &rgb0, image &label, int nlabels,
                   double robot_ic[], double robot_jc[], int &n_robot_blobs,
                   double &robot_center_ic, double &robot_center_jc,
                   double opp_ic[], double opp_jc[], int &n_opp_blobs,
                   double &opp_center_ic, double &opp_center_jc,
                   double obs_ic[], double obs_jc[], int &n_obs_detected,
                   image &rgb, int frame_count,
                   const double obs_seed_x[], const double obs_seed_y[],
                   int n_obs_seed)
{
	const double min_area       = 200.0;
	const double size_threshold = 2500.0;

	// Obstacle position memory.
	// Seeded from main()'s x_obs/y_obs constants (which we OWN as program
	// configuration, not S1 sim state) so the count is locked at the
	// number of real obstacles. Phantom blobs (laser flash, robot+
	// obstacle merger, etc.) can never grow this list. Vision still
	// refines positions via low-pass each frame, but no new entries.
	static double kobs_x[10] = {0};
	static double kobs_y[10] = {0};
	static int    n_kobs   = 0;
	static bool   kobs_seeded = false;
	if (!kobs_seeded) {
		for (int k = 0; k < n_obs_seed && k < 10; k++) {
			kobs_x[k] = obs_seed_x[k];
			kobs_y[k] = obs_seed_y[k];
		}
		n_kobs = n_obs_seed;
		kobs_seeded = true;
	}
	const double KOBS_LP   = 0.2;    // low-pass blend factor
	const double REJECT_R  = 90.0;   // position rejection radius (px)

	// collect features for all blobs once (so two-pass logic doesn't
	// re-walk the image)
	struct B { double ic, jc, area, hue, sat, value; };
	const int MAX_LBL = 64;
	B blobs[MAX_LBL];
	int n_total = 0;
	{
		double ic, jc, area, R_ave, G_ave, B_ave;
		double hue, sat, value;
		for (int i_label = 1; i_label <= nlabels && n_total < MAX_LBL; i_label++) {
			features(a, rgb0, label, i_label, ic, jc, area, R_ave, G_ave, B_ave);
			if (area < min_area) continue;
			calculate_HSV((int)R_ave, (int)G_ave, (int)B_ave, hue, sat, value);
			B &b = blobs[n_total++];
			b.ic = ic; b.jc = jc; b.area = area;
			b.hue = hue; b.sat = sat; b.value = value;
		}
	}

	n_robot_blobs  = 0;
	n_opp_blobs    = 0;
	n_obs_detected = 0;

	// PASS 1: large blobs are obstacles. Update kobs memory and emit.
	for (int i = 0; i < n_total; i++) {
		if (blobs[i].area < size_threshold) continue;

		// Match to nearest seeded kobs entry (count is fixed at startup).
		// A large blob that doesn't match any kobs entry is ignored as
		// obstacle memory - it could be a laser flash, robot/obstacle
		// merger, or other transient artifact, NOT a new obstacle.
		int matched_k = -1;
		double best_d = 1e9;
		for (int k = 0; k < n_kobs; k++) {
			double dx = blobs[i].ic - kobs_x[k];
			double dy = blobs[i].jc - kobs_y[k];
			double d  = sqrt(dx*dx + dy*dy);
			if (d < 80.0 && d < best_d) { best_d = d; matched_k = k; }
		}
		if (matched_k >= 0) {
			kobs_x[matched_k] = (1.0 - KOBS_LP) * kobs_x[matched_k] + KOBS_LP * blobs[i].ic;
			kobs_y[matched_k] = (1.0 - KOBS_LP) * kobs_y[matched_k] + KOBS_LP * blobs[i].jc;
		}
		// note: no else-branch. We never add new kobs entries beyond
		// the seeded count. This prevents phantom obstacles forever.

		if (n_obs_detected < 10) {
			obs_ic[n_obs_detected] = blobs[i].ic;
			obs_jc[n_obs_detected] = blobs[i].jc;
			n_obs_detected++;
		}
		draw_point_rgb(rgb, (int)blobs[i].ic, (int)blobs[i].jc, 255, 255, 255);

		if (frame_count == 1) {
			const char *bn = "obstacle_unknown";
			if (blobs[i].value < 50) bn = "obstacle_black";
			else if ((blobs[i].hue >= 340 || blobs[i].hue <= 20) && blobs[i].sat > 0.4) bn = "obstacle_red";
			else if (blobs[i].hue > 20 && blobs[i].hue <= 45 && blobs[i].sat > 0.5) bn = "obstacle_orange";
			else if (blobs[i].hue > 80 && blobs[i].hue <= 160 && blobs[i].sat > 0.3) bn = "obstacle_green";
			else if (blobs[i].hue > 190 && blobs[i].hue <= 260 && blobs[i].sat > 0.3) bn = "obstacle_blue";
			cout << "\n[" << bn << "]"
			     << "  centroid: (" << (int)blobs[i].ic << ", " << (int)blobs[i].jc << ")"
			     << "  HSV: (" << (int)blobs[i].hue << " deg, " << blobs[i].sat
			     << ", " << (int)blobs[i].value << ")"
			     << "  area: " << (int)blobs[i].area;
		}
	}

	// PASS 1.5: any kobs not detected this frame - keep emitting them
	// as remembered obstacles so wall/LOS/avoidance handlers stay
	// consistent even when the obstacle is partially occluded
	for (int k = 0; k < n_kobs; k++) {
		bool seen = false;
		for (int q = 0; q < n_obs_detected; q++) {
			double dx = obs_ic[q] - kobs_x[k];
			double dy = obs_jc[q] - kobs_y[k];
			if (sqrt(dx*dx + dy*dy) < 80.0) { seen = true; break; }
		}
		if (!seen && n_obs_detected < 10) {
			obs_ic[n_obs_detected] = kobs_x[k];
			obs_jc[n_obs_detected] = kobs_y[k];
			n_obs_detected++;
			// orange overlay = remembered obstacle (not seen this frame)
			draw_point_rgb(rgb, (int)kobs_x[k], (int)kobs_y[k], 255, 128, 0);
		}
	}

	// PASS 2: small blobs - robot/opp markers, with obstacle rejection
	for (int i = 0; i < n_total; i++) {
		if (blobs[i].area >= size_threshold) continue;

		// Robot markers in the test frame all have sat >= 0.51 and
		// value >= 118. If a small blob looks bright + saturated, it's
		// almost certainly a robot marker - DO NOT reject it by
		// position even if it's near a known obstacle. This avoids
		// killing the robot's own front/back markers when the robot
		// is driving close to an obstacle.
		bool looks_like_robot_marker = (blobs[i].sat   >= 0.45 &&
		                                blobs[i].value >= 100);

		bool is_obs_artifact = false;

		// HSV signature reject: definitive obstacle colour
		// - black obstacle: value < 50 (robot markers always > 100)
		// - green obstacle: hue 130-180, sat > 0.55, value > 160.
		//   Robot_A's cyan back marker has hue 154 / sat 0.53 - sat cut
		//   keeps it on the right side of the line.
		if (blobs[i].value < 50) is_obs_artifact = true;
		if (blobs[i].hue >= 130 && blobs[i].hue <= 180 &&
		    blobs[i].sat >  0.55 && blobs[i].value > 160) {
			is_obs_artifact = true;
		}

		// Position rejection - ONLY if HSV doesn't look like a robot
		// marker. Without this guard, REJECT_R=90 was killing the
		// robot's own front marker whenever the robot got within ~73 px
		// of an obstacle, and the robot would go fully blind to itself.
		if (!is_obs_artifact && !looks_like_robot_marker) {
			for (int k = 0; k < n_kobs; k++) {
				double dx = blobs[i].ic - kobs_x[k];
				double dy = blobs[i].jc - kobs_y[k];
				if (sqrt(dx*dx + dy*dy) < REJECT_R) {
					is_obs_artifact = true;
					break;
				}
			}
		}

		if (is_obs_artifact) {
			// gray overlay = rejected as obstacle artifact
			draw_point_rgb(rgb, (int)blobs[i].ic, (int)blobs[i].jc, 100, 100, 100);
			continue;
		}

		double d_robot = (blobs[i].ic - robot_center_ic)*(blobs[i].ic - robot_center_ic)
		               + (blobs[i].jc - robot_center_jc)*(blobs[i].jc - robot_center_jc);
		double d_opp   = (blobs[i].ic - opp_center_ic)*(blobs[i].ic - opp_center_ic)
		               + (blobs[i].jc - opp_center_jc)*(blobs[i].jc - opp_center_jc);

		int R, G, B;
		const char *bn;
		if (d_robot <= d_opp) {
			bn = "robot";
			if (n_robot_blobs < 10) {
				robot_ic[n_robot_blobs] = blobs[i].ic;
				robot_jc[n_robot_blobs] = blobs[i].jc;
				n_robot_blobs++;
			}
			R = 0; G = 255; B = 0;
		} else {
			bn = "opponent";
			if (n_opp_blobs < 10) {
				opp_ic[n_opp_blobs] = blobs[i].ic;
				opp_jc[n_opp_blobs] = blobs[i].jc;
				n_opp_blobs++;
			}
			R = 0; G = 255; B = 255;
		}
		draw_point_rgb(rgb, (int)blobs[i].ic, (int)blobs[i].jc, R, G, B);

		if (frame_count == 1) {
			cout << "\n[" << bn << "]"
			     << "  centroid: (" << (int)blobs[i].ic << ", " << (int)blobs[i].jc << ")"
			     << "  HSV: (" << (int)blobs[i].hue << " deg, " << blobs[i].sat
			     << ", " << (int)blobs[i].value << ")"
			     << "  area: " << (int)blobs[i].area;
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

// features - prof's 
int features(image &a, image &rgb, image &label, int label_i, double &ic, double &jc, double &area,
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
// calculate_HSV - prof's HSV computation (assignment 7)
void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value)
{
	int max_v, min_v, delta;
	double H;

	max_v = min_v = R;
	if (G > max_v) max_v = G;
	if (B > max_v) max_v = B;
	if (G < min_v) min_v = G;
	if (B < min_v) min_v = B;

	delta = max_v - min_v;
	value = max_v;

	if (delta == 0) sat = 0.0;
	else            sat = (double)delta / value;

	if (delta == 0)      H = 0;
	else if (max_v == R) H = (double)(G - B) / delta;
	else if (max_v == G) H = (double)(B - R) / delta + 2;
	else                 H = (double)(R - G) / delta + 4;

	hue = 60 * H;
	if (hue < 0) hue += 360;
}


// MECH 472/663 - Player 2 DEFENCE (v3)
// Evade the opponent robot by hiding behind obstacles.
// mode = 2 (two player, player #2 controls robot_B)
//
// Vision: HSV-based thresholding (from prof's assignment 7)
//   - saturation > 0.20 or value < 50 = foreground
//   - size-based blob classification: small = robot marker, large = obstacle
//   - proximity-based robot/opponent assignment using previous frame centres
//
// Navigation: potential field (attractive + obstacle repulsion + boundary repulsion)
//   - Ka = 4.0 (attraction to hiding spot)
//   - Kr = 15000 (1/r^2 obstacle repulsion)
//   - Kb = 15000 (1/r^2 boundary repulsion)
//
// Strategy: compute hiding spot behind obstacle, navigate there, stop when hidden
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
// MAIN
// ============================================================

int main()
{
	const double PI = 3.14159265;

	// ---- simulation setup (must match player1) ----
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

	// persistent obstacle memory: keep last known positions
	// so we can still navigate when vision temporarily loses an obstacle
	double known_obs_ic[10], known_obs_jc[10];
	int    n_known_obs = 0;

	int width = 640, height = 480;

	cout << "\n=== PLAYER 2 - DEFENCE (v3) ===";
	cout << "\nVision-only detection (no simulator state)";
	cout << "\nHSV detection + potential field navigation";
	cout << "\nPress space to begin.";
	pause();

	activate_vision();
	activate_simulation(width1, height1,
		x_obs, y_obs, N_obs,
		"robot_A.bmp", "robot_B.bmp", "background.bmp",
		obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);

	set_simulation_mode(2);
	set_robot_position(140, 240, 0.0);
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

	// seed centre estimates
	robot_center_ic = 140;  robot_center_jc = 240;
	opp_center_ic   = 500;  opp_center_jc   = 240;

	int nlabels;
	static int frame_count = 0;

	join_player();

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

		// update persistent obstacle memory when vision detects them
		if (n_obs_detected > 0) {
			n_known_obs = n_obs_detected;
			for (int k = 0; k < n_obs_detected && k < 10; k++) {
				known_obs_ic[k] = obs_ic[k];
				known_obs_jc[k] = obs_jc[k];
			}
		}

		// use known obstacles for navigation (persists when vision
		// temporarily loses an obstacle behind a robot)
		int n_nav_obs = n_known_obs;
		double *nav_obs_ic = known_obs_ic;
		double *nav_obs_jc = known_obs_jc;

		// ============================================================
		// ROBOT HEADING ESTIMATION
		// ============================================================
		double rx = robot_center_ic;
		double ry = robot_center_jc;
		double sim_theta = S1->P[2]->x[1]; // fallback only

		// use direction toward hide spot for ambiguity resolution
		// (computed after hide spot, but we need heading first --
		//  use direction away from opponent as proxy)
		double away_ang = atan2(ry - opp_center_jc, rx - opp_center_ic);

		double rtheta;
		if (n_robot_blobs >= 2) {
			double raw = atan2(robot_jc[1] - robot_jc[0],
			                   robot_ic[1] - robot_ic[0]);
			double err1 = away_ang - raw;
			while (err1 >  PI) err1 -= 2*PI;
			while (err1 < -PI) err1 += 2*PI;
			double err2 = away_ang - (raw + PI);
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
		// HIDING SPOT SELECTION (using vision-detected obstacles)
		// ============================================================
		double hide_ic = fx, hide_jc = fy;
		double dx, dy, dist;

		const double hide_dist = 110.0;

		if (n_nav_obs > 0) {
			double hx_all[10], hy_all[10];
			for (int k = 0; k < n_nav_obs; k++) {
				double ox   = nav_obs_ic[k];
				double oy   = nav_obs_jc[k];
				double vx   = ox - opp_center_ic;
				double vy   = oy - opp_center_jc;
				double vlen = sqrt(vx*vx + vy*vy) + 1e-6;
				hx_all[k] = ox + hide_dist * vx / vlen;
				hy_all[k] = oy + hide_dist * vy / vlen;
				if (hx_all[k] <  10) hx_all[k] =  10;
				if (hx_all[k] > 630) hx_all[k] = 630;
				if (hy_all[k] <  10) hy_all[k] =  10;
				if (hy_all[k] > 470) hy_all[k] = 470;
			}

			double best_score = -1e9;
			for (int k = 0; k < n_nav_obs; k++) {
				double d_opp = (hx_all[k] - opp_center_ic)*(hx_all[k] - opp_center_ic)
				             + (hy_all[k] - opp_center_jc)*(hy_all[k] - opp_center_jc);
				double d_rob = (hx_all[k] - fx)*(hx_all[k] - fx)
				             + (hy_all[k] - fy)*(hy_all[k] - fy);
				double score = d_opp - 0.3 * d_rob;
				if (score > best_score) {
					best_score = score;
					hide_ic = hx_all[k];
					hide_jc = hy_all[k];
				}
			}
		}

		// ============================================================
		// LINE-OF-SIGHT CHECK (using vision-detected obstacles)
		// ============================================================
		bool in_los = true;
		const double obs_radius = 45.0;

		for (int k = 0; k < n_nav_obs; k++) {
			double ax = opp_center_ic, ay = opp_center_jc;
			double bx = fx,            by = fy;
			double px = nav_obs_ic[k], py = nav_obs_jc[k];

			double sdx = bx - ax, sdy = by - ay;
			double len2 = sdx*sdx + sdy*sdy + 1e-6;
			double t = ((px - ax)*sdx + (py - ay)*sdy) / len2;
			if (t < 0.0) t = 0.0;
			if (t > 1.0) t = 1.0;
			double cx = ax + t*sdx, cy = ay + t*sdy;

			double d = sqrt((px - cx)*(px - cx) + (py - cy)*(py - cy));
			if (d < obs_radius) { in_los = false; break; }
		}

		// proximity guard
		if (!in_los) {
			bool near_any = false;
			for (int k = 0; k < n_nav_obs; k++) {
				double ddx = fx - nav_obs_ic[k];
				double ddy = fy - nav_obs_jc[k];
				if (sqrt(ddx*ddx + ddy*ddy) < 150.0) { near_any = true; break; }
			}
			if (!near_any) in_los = true;
		}

		// ============================================================
		// POTENTIAL FIELD (using vision-detected obstacles)
		// ============================================================
		double Fx = 0, Fy = 0;

		double Ka = 4.0;
		dx   = hide_ic - fx;
		dy   = hide_jc - fy;
		dist = sqrt(dx*dx + dy*dy) + 1e-6;
		Fx  += Ka * dx / dist;
		Fy  += Ka * dy / dist;

		double Kr = 15000.0;
		for (int k = 0; k < n_nav_obs; k++) {
			dx = fx - nav_obs_ic[k];
			dy = fy - nav_obs_jc[k];
			double dist2 = dx*dx + dy*dy + 1e-6;
			Fx += Kr * dx / (dist2 * sqrt(dist2));
			Fy += Kr * dy / (dist2 * sqrt(dist2));
		}

		double Kb = 15000.0;
		Fx += Kb / (fx*fx + 1);
		Fx -= Kb / ((640.0 - fx)*(640.0 - fx) + 1);
		Fy += Kb / (fy*fy + 1);
		Fy -= Kb / ((480.0 - fy)*(480.0 - fy) + 1);

		double desired_heading = atan2(Fy, Fx);

		// ============================================================
		// REFINE HEADING using desired_heading for ambiguity resolution
		// ============================================================
		// now that we have the potential field direction, re-resolve
		// the 180-degree ambiguity more accurately
		if (n_robot_blobs >= 2) {
			double raw = atan2(robot_jc[1] - robot_jc[0],
			                   robot_ic[1] - robot_ic[0]);
			double err1 = desired_heading - raw;
			while (err1 >  PI) err1 -= 2*PI;
			while (err1 < -PI) err1 += 2*PI;
			double err2 = desired_heading - (raw + PI);
			while (err2 >  PI) err2 -= 2*PI;
			while (err2 < -PI) err2 += 2*PI;
			rtheta = (fabs(err1) <= fabs(err2)) ? raw : raw + PI;
		}

		double heading_error = desired_heading - rtheta;
		while (heading_error >  PI) heading_error -= 2*PI;
		while (heading_error < -PI) heading_error += 2*PI;

		// ============================================================
		// WAYPOINT NAVIGATOR (two-phase: rotate then drive)
		// ============================================================
		static int wp_state = 0;

		const double angle_thresh = 0.20;
		const double arrive_dist  = 30.0;

		double dist_to_wp = sqrt((hide_ic - fx)*(hide_ic - fx) +
		                         (hide_jc - fy)*(hide_jc - fy));

		if (!in_los) {
			pw_r = 1500; pw_l = 1500;
			wp_state = 0;

		} else if (dist_to_wp < arrive_dist) {
			pw_r = 1500; pw_l = 1500;
			wp_state = 0;

		} else if (wp_state == 0) {
			if (fabs(heading_error) < angle_thresh) {
				wp_state = 1;
				pw_r = 1500; pw_l = 1500;
			} else {
				int turn = (int)(300.0 * heading_error);
				if (turn >  450) turn =  450;
				if (turn < -450) turn = -450;
				pw_r = 1500 + turn;
				pw_l = 1500 + turn;
			}

		} else {
			if (fabs(heading_error) > angle_thresh * 2.5) {
				wp_state = 0;
				pw_r = 1500; pw_l = 1500;
			} else {
				int fwd = (int)(3.5 * dist_to_wp);
				if (fwd > 480) fwd = 480;
				if (fwd <  80) fwd =  80;
				int turn = (int)(180.0 * heading_error);
				if (turn >  200) turn =  200;
				if (turn < -200) turn = -200;
				pw_r = 1500 + fwd + turn;
				pw_l = 1500 - fwd + turn;
			}
		}

		if (pw_r > 2000) pw_r = 2000;
		if (pw_r < 1000) pw_r = 1000;
		if (pw_l > 2000) pw_l = 2000;
		if (pw_l < 1000) pw_l = 1000;

		laser = 0;

		set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

		// ---- console output ----
		if (frame_count % 60 == 0) {
			cout << "\n[t=" << (int)tc << "s]"
			     << " robot=(" << (int)rx << "," << (int)ry << ")"
			     << " opp=(" << (int)opp_center_ic << "," << (int)opp_center_jc << ")"
			     << " hide=(" << (int)hide_ic << "," << (int)hide_jc << ")"
			     << " wp_d=" << (int)dist_to_wp
			     << " herr=" << (int)(heading_error * 180.0 / PI) << "deg"
			     << " pw=" << pw_l << "/" << pw_r;
			if (!in_los) cout << " [HIDDEN]";
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

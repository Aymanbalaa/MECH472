// OFFENSE ROBOT MARKERS OF COLORS RED AND GREEN
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

const double PI    = 3.14159265;
const int    IMG_W = 640;
const int    IMG_H = 480;

// functions reused from class examples
int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave);
void calculate_HSV(int R, int G, int B,
                   double &hue, double &sat, double &value);

static double wrap(double a) {
	while (a >  PI) a -= 2*PI;
	while (a < -PI) a += 2*PI;
	return a;
}

// helped cleanup the drive , caused by some other functions returning higher values than expected
static int clamp_pw(int pw) {
	if (pw < 1000) return 1000;
	if (pw > 2000) return 2000;
	return pw;
}

// diff drive kinematics while using the clamp method and adjusting with error to try and eliminate jerking movements
// initially this was implemented without error but was then readjusted for smooth
static void drive_smooth(int &pw_l, int &pw_r, double err) {
	double abs_err = fabs(err);
	int fwd = (int)(200.0 - 120.0 * (abs_err / PI));
	if (fwd < 80) fwd = 80;
	int steer = (int)(280.0 * err);
	int max_s = fwd / 2;
	if (steer >  max_s) steer =  max_s;
	if (steer < -max_s) steer = -max_s;
	pw_l = clamp_pw(1500 - fwd + steer);
	pw_r = clamp_pw(1500 + fwd + steer);
}

// spinning in place for sharp turns : depends on what is needed from the robot
static void spin(int &pw_l, int &pw_r, double err) {
	int rot;
	double abs_err = fabs(err);
	if      (abs_err > 0.5)  rot = 200;
	else if (abs_err > 0.15) rot = 130;
	else                     rot = 70;
	int sign = (err >= 0) ? 1 : -1;
	pw_l = clamp_pw(1500 + sign * rot);
	pw_r = clamp_pw(1500 + sign * rot);
}

void run_vision(image &rgb, image &rgb0, image &a, image &b, image &label,
                double robot_ic[], double robot_jc[], int &n_robot_blobs,
                double opp_ic[],   double opp_jc[],   int &n_opp_blobs,
                double obs_ic[],   double obs_jc[],   int &n_obs_blobs,
                double &rcx, double &rcy, double &ocx, double &ocy,
                const double obs_seed_x[], const double obs_seed_y[], int n_seed,
                int frame_count);


int main()
{
	// sim setup , those should be similar to the values the defense is setting as well
	double width1 = IMG_W, height1 = IMG_H;
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

	const int    LOST_THRESH  = 25;     // frames lost -> SEARCH
	const double LOS_BLOCK_R  = 70.0;   // px: perp-dist threshold to block LOS
	const double FRONT_OFFSET = 65.0;   // px: laser exit offset from centre

   //paramerts can be tuned , mostly for margin and avoidance , explained one by one below
	const double FIRE_DIST    = 200.0;  // how close do i have to be to shoot?
	const double BACKUP_DIST  = 120.0;  // how much do i need to backup in emergency/blocked
	const double FIRE_TOL     = 0.06;   // how accurate do i need to be when shooting?
	const double OBS_AVOID    = 150.0;  // by how much should i avoid obstacles
	const double OBS_EMERG    = 40.0;   // when am i in emergency prox to obstacle?
	const int    WALL_MARGIN  = 15;     // how careful should i be to the wall?

	cout << "\nPLAYER 2 - OFFENSE - BLUE AND ORANGE ROBOT";
	pause();

	activate_vision();
	activate_simulation(width1, height1, x_obs, y_obs, N_obs, "robot_B.bmp", "robot_A.bmp", "background.bmp", obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);
	set_simulation_mode(2);
	set_robot_position(140, 240, 0.0);
	set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

	image rgb, a, b, rgb0, label;
	rgb.type   = RGB_IMAGE;   rgb.width   = IMG_W; rgb.height   = IMG_H;
	a.type     = GREY_IMAGE;  a.width     = IMG_W; a.height     = IMG_H;
	b.type     = GREY_IMAGE;  b.width     = IMG_W; b.height     = IMG_H;
	rgb0.type  = RGB_IMAGE;   rgb0.width  = IMG_W; rgb0.height  = IMG_H;
	label.type = LABEL_IMAGE; label.width = IMG_W; label.height = IMG_H;
	allocate_image(rgb);  allocate_image(a);   allocate_image(b);
	allocate_image(rgb0); allocate_image(label);

	double robot_ic[10], robot_jc[10];  int n_robot_blobs = 0;
	double opp_ic[10],   opp_jc[10];    int n_opp_blobs   = 0;
	double obs_ic[10],   obs_jc[10];    int n_obs_blobs   = 0;
	//initial coordinates/estimates of robot , can still run with wrong values but this will smoothen initial detection
	// r for robot , o for oponent 
	double rcx = 140, rcy = 240;
	double ocx = 500, ocy = 100;

	//defining some vars to be used below
	int    state = 0;       
	int    fired = 0;
	int    lost_frames = 0;
	int    frame_count = 0;
	double rtheta_last = 0.0;
	bool   heading_set = false;

	static const char *snames[] = { "SEARCH", "CHASE", "ATTACK" };

	join_player();
	double tc0 = high_resolution_time(), tc;

	while (1) {

		acquire_image_sim(rgb);
		tc = high_resolution_time() - tc0;
		frame_count++;

		run_vision(rgb, rgb0, a, b, label,
		           robot_ic, robot_jc, n_robot_blobs,
		           opp_ic,   opp_jc,   n_opp_blobs,
		           obs_ic,   obs_jc,   n_obs_blobs,
		           rcx, rcy, ocx, ocy,
		           x_obs, y_obs, N_obs, frame_count);

		// if markers detection got lost , reuse the last known coordinates 
		//this help if robot tempo got out of arena 
		double rx = rcx, ry = rcy;
		double dir_to_opp = atan2(ocy - ry, ocx - rx);
		double rtheta;
		if (n_robot_blobs >= 2) { // continue since full robot detected
			double raw = atan2(robot_jc[1] - robot_jc[0],
			                   robot_ic[1] - robot_ic[0]);
			double ref = heading_set ? rtheta_last : dir_to_opp;
			double e1 = wrap(ref - raw);
			double e2 = wrap(ref - (raw + PI));
			rtheta = (fabs(e1) <= fabs(e2)) ? raw : raw + PI;
			rtheta_last = rtheta;
			heading_set = true;
		} else {
			rtheta = heading_set ? rtheta_last : dir_to_opp;
		}

		double fx = rx + FRONT_OFFSET * cos(rtheta);
		double fy = ry + FRONT_OFFSET * sin(rtheta);

		bool found = (n_opp_blobs >= 1);
		if (found) lost_frames = 0;
		else       lost_frames++;

		double dist    = sqrt((ocx-rx)*(ocx-rx) + (ocy-ry)*(ocy-ry)); //distance from us to opponent
		double aim_err = wrap(atan2(ocy-ry, ocx-rx) - rtheta); // how far is my current aim from opponent

		// obstacle proximity
		bool   near_obs = false, obs_emerg = false;
		double obs_min  = 1e9;
		int    obs_k    = -1;
		double obs_dist[10];
		for (int k = 0; k < n_obs_blobs; k++) {
			double od = sqrt((rx-obs_ic[k])*(rx-obs_ic[k]) +
			                 (ry-obs_jc[k])*(ry-obs_jc[k]));
			double fd = sqrt((fx-obs_ic[k])*(fx-obs_ic[k]) +
			                 (fy-obs_jc[k])*(fy-obs_jc[k]));
			double c  = (fd < od) ? fd : od;
			obs_dist[k] = c;
			if (c < obs_min) { obs_min = c; obs_k = k; }
			if (c < OBS_AVOID) near_obs = true;
			if (c < OBS_EMERG) obs_emerg = true;
		}

		// obs_k hysteresis - lock onto nearest, only flip if 30 px closer
		static int obs_k_locked = -1;
		if (obs_emerg || obs_min > OBS_AVOID + 20.0) {
			obs_k_locked = obs_k;
		} else if (obs_k_locked >= 0 && obs_k_locked < n_obs_blobs) {
			if (obs_dist[obs_k] + 30.0 < obs_dist[obs_k_locked])
				obs_k_locked = obs_k;
		} else {
			obs_k_locked = obs_k;
		}
		if (obs_k_locked >= 0 && obs_k_locked < n_obs_blobs && !obs_emerg) {
			obs_k = obs_k_locked;
		}

		bool wall_hit = (rx < WALL_MARGIN || rx > IMG_W - WALL_MARGIN ||
		                 ry < WALL_MARGIN || ry > IMG_H - WALL_MARGIN ||
		                 fx < WALL_MARGIN || fx > IMG_W - WALL_MARGIN ||
		                 fy < WALL_MARGIN || fy > IMG_H - WALL_MARGIN);

		// LINE OF SIGHT CHECK BUT ACROSS MULTIPLE CANDIDATES
		// initially this was developed just for center of robot but was then redesigned to add checks to the individual markers
		// robot was able to hit targets who are partially hidden if one of the markers is exposed
		double cand_tx[3], cand_ty[3], cand_err[3];
		bool   cand_clear[3];
		const char *cand_name[3];
		int n_cand = 0;
		cand_tx[n_cand] = ocx; cand_ty[n_cand] = ocy;
		cand_name[n_cand++] = "C"; // C is for center
		for (int k = 0; k < n_opp_blobs && n_cand < 3; k++) {
			cand_tx[n_cand] = opp_ic[k];
			cand_ty[n_cand] = opp_jc[k];
			cand_name[n_cand++] = (k == 0) ? "M0" : "M1"; // M0 and M1 are for the markers
		}
		for (int c = 0; c < n_cand; c++) {
			cand_err[c] = wrap(atan2(cand_ty[c]-ry, cand_tx[c]-rx) - rtheta); // checks aim error for each candidate 
			bool blocked = false;
			if (found) {
				double dx = cand_tx[c] - fx, dy = cand_ty[c] - fy; // do we even have open LOS
				double seg = sqrt(dx*dx + dy*dy);
				if (seg > 1.0) {
					double ux = dx/seg, uy = dy/seg; 
					for (int k = 0; k < n_obs_blobs; k++) {
						double t = (obs_ic[k]-fx)*ux + (obs_jc[k]-fy)*uy; 
						if (t < 0 || t > seg) continue; //
						double px = fx + t*ux, py = fy + t*uy;
						double perp = sqrt((obs_ic[k]-px)*(obs_ic[k]-px) + 
						                   (obs_jc[k]-py)*(obs_jc[k]-py));
						if (perp < LOS_BLOCK_R) { blocked = true; break; } // checks if any obstacle is blocking LOS to M0 M1 or C
					}
				}
			}
			cand_clear[c] = !blocked;
		}
		int best = 0; double best_abs = 1e9; bool any_clear = false;
		for (int c = 0; c < n_cand; c++) {
			if (!cand_clear[c]) continue;
			if (fabs(cand_err[c]) < best_abs) {
				best_abs = fabs(cand_err[c]); best = c; any_clear = true;
			}
		}
		bool   clear_shot  = any_clear;
		double attack_err  = any_clear ? cand_err[best] : aim_err;
		const char *aim_tgt = any_clear ? cand_name[best] : "-";

		bool can_attack = (found && clear_shot && dist < FIRE_DIST); // opponent found , clear shot and inside firing range?

		// the following are state transitions and actions for the offense robot , explained in report and exam
		if (lost_frames > LOST_THRESH)        state = 0; // search
		else if (found && state == 0)         state = 1; // chase
		else if (can_attack)                  state = 2; // attack
		else if (state == 2 && !can_attack)   state = 1; // chase

		if (wall_hit) {
			double err_c = wrap(atan2(IMG_H/2.0 - ry, IMG_W/2.0 - rx) - rtheta); //turn towards center , best guess
			if (fabs(err_c) > 0.3) spin(pw_l, pw_r, err_c);
			else                   drive_smooth(pw_l, pw_r, err_c);

		} else if (state == 2 && clear_shot && !obs_emerg) {
			if (dist < BACKUP_DIST) { pw_l = 1500 + 150; pw_r = 1500 - 150; } // back up if too close
			else                     spin(pw_l, pw_r, attack_err);

		} else if ((near_obs || (found && !clear_shot && dist < FIRE_DIST)) &&
		           obs_k >= 0)
		{
			double away = atan2(ry - obs_jc[obs_k], rx - obs_ic[obs_k]); // change dir to avoid obstacl
			if (obs_emerg) {
				double err_a = wrap(away - rtheta);
				if (fabs(err_a) > PI/2.0) { pw_l = 1500 + 200; pw_r = 1500 - 200; }
				else                       drive_smooth(pw_l, pw_r, err_a);
			} else {
				static int side = 0, hold = 0;
				if (hold <= 0) {
					double obs_dx = obs_ic[obs_k] - rx;
					double obs_dy = obs_jc[obs_k] - ry;
					double opp_dx = ocx - rx, opp_dy = ocy - ry;
					// sign flipped: image y-down inverts cross-product CCW
					side = (obs_dx*opp_dy - obs_dy*opp_dx > 0) ? -1 : 1;
					hold = 80;
				}
				hold--;
				double tang = away + side * (PI / 2.0);
				drive_smooth(pw_l, pw_r, wrap(tang - rtheta));
			}

		} else {
			switch (state) {
			case 0: pw_l = 1500 + 175; pw_r = 1500 + 175; break;  // search
			case 1: // chase
				if (dist < BACKUP_DIST) { pw_l = 1500 + 200; pw_r = 1500 - 200; }
				else                     drive_smooth(pw_l, pw_r, aim_err);
				break;
			case 2:
				if (dist < BACKUP_DIST) { pw_l = 1500 + 150; pw_r = 1500 - 150; }
				else                     spin(pw_l, pw_r, attack_err);
				break;
			}
		}

		pw_l = clamp_pw(pw_l);
		pw_r = clamp_pw(pw_r);

		// fire (one shot)
		if (!fired && tc > 1.0 && state == 2 &&
		    fabs(attack_err) < FIRE_TOL && clear_shot && dist < FIRE_DIST &&
		    n_robot_blobs == 2 && n_opp_blobs >= 1)
		{
			fired = 1; laser = 1; pw_laser = 1500;
			pw_l = 1500; pw_r = 1500;
			set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
			view_rgb_image(rgb, 1);
			cout << "\n OFFENSE FIRED THE LASER";
			cout << "\nPress space to exit";
			pause();
			break; // stop program to hold angle after laser is fired
		}

		set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
		view_rgb_image(rgb, 1);

		if (KEY('X')) break;
	}

	free_image(rgb);  free_image(a); free_image(b);
	free_image(rgb0); free_image(label);
	deactivate_vision();
	deactivate_simulation();
	cout << "\ndone\n";
	return 0;
}

// run_vision -- HSV pipeline + blob classification
void run_vision(image &rgb, image &rgb0, image &a, image &b, image &label,
                double robot_ic[], double robot_jc[], int &n_robot_blobs,
                double opp_ic[],   double opp_jc[],   int &n_opp_blobs,
                double obs_ic[],   double obs_jc[],   int &n_obs_blobs,
                double &rcx, double &rcy, double &ocx, double &ocy,
                const double obs_seed_x[], const double obs_seed_y[], int n_seed,
                int frame_count)
{
	const double MIN_AREA = 200.0;
	const double SIZE_THR = 2500.0;
	const double REJECT_R = 90.0;
	const double KOBS_LP  = 0.2;

	// pipeline used in class example
	copy(rgb, rgb0);
	copy(rgb, a);
	lowpass_filter(a, b); copy(b, a);
	scale(a, b);          copy(b, a);
	{
		ibyte *prgb = rgb0.pdata;
		ibyte *pb   = b.pdata;
		int npix = rgb0.width * rgb0.height;
		double h, s, v;
		for (int k = 0; k < npix; k++, prgb += 3) {
			calculate_HSV(prgb[2], prgb[1], prgb[0], h, s, v);
			pb[k] = (s > 0.20 || v < 50) ? 255 : 0;
		}
	}
	copy(b, a);
	erode(a, b);   copy(b, a);
	erode(a, b);   copy(b, a);
	dialate(a, b); copy(b, a);
	dialate(a, b); copy(b, a);
	int nlabels;
	label_image(a, label, nlabels);

	static double kobs_x[10], kobs_y[10];
	static bool   kobs_seeded = false;
	if (!kobs_seeded) {
		for (int k = 0; k < n_seed; k++) {
			kobs_x[k] = obs_seed_x[k]; kobs_y[k] = obs_seed_y[k];
		}
		kobs_seeded = true;
	}

	//note!!!!
	// collect features for all blobs once
	// similar logic to defense detect_objects function but with slight diff implementation
	// rules and classifications are performed the same way the defense robot will do them 
	// when tested with diff obstacles and starting positions , all blobs were correctly detected
	struct B { double ic, jc, area, hue, sat, value; };
	const int MAX_B = 64;
	B blobs[MAX_B];
	int n_total = 0;
	{
		double ic, jc, area, R_a, G_a, B_a, h, s, v;
		for (int i = 1; i <= nlabels && n_total < MAX_B; i++) {
			features(a, rgb0, label, i, ic, jc, area, R_a, G_a, B_a);
			if (area < MIN_AREA) continue;
			calculate_HSV((int)R_a, (int)G_a, (int)B_a, h, s, v);
			B &bb = blobs[n_total++];
			bb.ic = ic; bb.jc = jc; bb.area = area;
			bb.hue = h; bb.sat = s; bb.value = v;
		}
	}

	n_robot_blobs = n_opp_blobs = n_obs_blobs = 0;

	// consider the largest blobs as obstacles
	for (int i = 0; i < n_total; i++) {
		if (blobs[i].area < SIZE_THR) continue;
		int matched = -1; double best_d = 1e9;
		for (int k = 0; k < n_seed; k++) {
			double dx = blobs[i].ic - kobs_x[k];
			double dy = blobs[i].jc - kobs_y[k];
			double d  = sqrt(dx*dx + dy*dy);
			if (d < 80.0 && d < best_d) { best_d = d; matched = k; }
		}
		if (matched >= 0) {
			kobs_x[matched] = (1-KOBS_LP)*kobs_x[matched] + KOBS_LP*blobs[i].ic;
			kobs_y[matched] = (1-KOBS_LP)*kobs_y[matched] + KOBS_LP*blobs[i].jc;
		}
		if (n_obs_blobs < 10) {
			obs_ic[n_obs_blobs] = blobs[i].ic;
			obs_jc[n_obs_blobs] = blobs[i].jc;
			n_obs_blobs++;
		}
	}

	// the smaller blobs are robot markers
	for (int i = 0; i < n_total; i++) {
		if (blobs[i].area >= SIZE_THR) continue;

		bool reject = false;
		if (blobs[i].value < 50) reject = true;
		if (blobs[i].hue >= 130 && blobs[i].hue <= 180 &&
		    blobs[i].sat  >  0.55 && blobs[i].value > 160) reject = true;

		bool looks_marker = (blobs[i].sat >= 0.45 && blobs[i].value >= 100);

		if (!reject && !looks_marker) {
			for (int k = 0; k < n_seed; k++) {
				double dx = blobs[i].ic - kobs_x[k];
				double dy = blobs[i].jc - kobs_y[k];
				if (sqrt(dx*dx + dy*dy) < REJECT_R) { reject = true; break; }
			}
		}
		if (reject) continue;

		double d_r = (blobs[i].ic - rcx) * (blobs[i].ic - rcx) + // distance from robot center , if closer to robot center than opponent center , then it is a robot marker	
		             (blobs[i].jc-rcy)*(blobs[i].jc-rcy);
		double d_o = (blobs[i].ic-ocx)*(blobs[i].ic-ocx) +
		             (blobs[i].jc-ocy)*(blobs[i].jc-ocy);
		if (d_r <= d_o) {
			if (n_robot_blobs < 10) {
				robot_ic[n_robot_blobs] = blobs[i].ic;
				robot_jc[n_robot_blobs] = blobs[i].jc;
				n_robot_blobs++;
			}
		} else {
			if (n_opp_blobs < 10) {
				opp_ic[n_opp_blobs] = blobs[i].ic;
				opp_jc[n_opp_blobs] = blobs[i].jc;
				n_opp_blobs++;
			}
		}
	}

	// calculate robot and opp centers if markers are detected
	if (n_robot_blobs >= 1) {
		double sx = 0, sy = 0;
		for (int k = 0; k < n_robot_blobs; k++) { sx += robot_ic[k]; sy += robot_jc[k]; }
		rcx = sx / n_robot_blobs;  rcy = sy / n_robot_blobs;
	}
	if (n_opp_blobs >= 1) {
		double sx = 0, sy = 0;
		for (int k = 0; k < n_opp_blobs; k++) { sx += opp_ic[k]; sy += opp_jc[k]; }
		ocx = sx / n_opp_blobs;  ocy = sy / n_opp_blobs;
	}
}

// class examples , were slightly modified if not directly reused
int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave)
{
	ibyte  *p  = rgb.pdata, *pc;
	i2byte *pl = (i2byte *)label.pdata;
	int width = rgb.width, height = rgb.height;
	double mi = 0, mj = 0, m = 0, n = 0;
	double R_sum = 0, G_sum = 0, B_sum = 0;
	const double EPS = 1e-7;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			if (pl[j*width+i] == label_i) {
				pc = p + 3*(i + width*j);
				B_sum += pc[0]; G_sum += pc[1]; R_sum += pc[2];
				n++; m++;
				mi += i; mj += j;
			}
		}
	}
	ic    = mi / (m + EPS);
	jc    = mj / (m + EPS);
	R_ave = R_sum / (n + EPS);
	G_ave = G_sum / (n + EPS);
	B_ave = B_sum / (n + EPS);
	area  = n;
	return 0;
}

void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value)
{
	int max_v = R, min_v = R;
	if (G > max_v) max_v = G;  if (B > max_v) max_v = B;
	if (G < min_v) min_v = G;  if (B < min_v) min_v = B;
	int delta = max_v - min_v;
	value = max_v;
	sat   = (delta == 0) ? 0.0 : (double)delta / value;
	double H = 0;
	if (delta == 0)      H = 0;
	else if (max_v == R) H = (double)(G - B) / delta;
	else if (max_v == G) H = (double)(B - R) / delta + 2;
	else                 H = (double)(R - G) / delta + 4;
	hue = 60 * H;
	if (hue < 0) hue += 360;
}

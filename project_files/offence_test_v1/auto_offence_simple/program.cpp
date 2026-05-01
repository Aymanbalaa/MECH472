// ATTEMPT FOR SIMPLE OFFENSE 
// MIGHT NOT BE AS RELIABLE 


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

// prof's vision functions (definitions at bottom)
int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave);
void calculate_HSV(int R, int G, int B,
                   double &hue, double &sat, double &value);


// ------------------------------------------------------------
// helpers
// ------------------------------------------------------------
static double wrap(double a) {
	while (a >  PI) a -= 2*PI;
	while (a < -PI) a += 2*PI;
	return a;
}

static int clamp_pw(int pw) {
	if (pw < 1000) return 1000;
	if (pw > 2000) return 2000;
	return pw;
}

// continuous-velocity steer: forward speed scales down with |err|,
// steer saturates at fwd/2 so inside wheel can stop but never reverse.
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

// in-place rotation, scaled by err magnitude
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


// ------------------------------------------------------------
// vision state passed back from run_vision()
// ------------------------------------------------------------
struct VisionState {
	double robot_ic[10], robot_jc[10];  int n_robot;
	double opp_ic[10],   opp_jc[10];    int n_opp;
	double obs_ic[10],   obs_jc[10];    int n_obs;
	double rcx, rcy;   // robot centroid (low-pass tracked)
	double ocx, ocy;   // opp   centroid (low-pass tracked)
};

void run_vision(image &rgb, image &rgb0, image &a, image &b, image &label,
                VisionState &vs,
                const double obs_seed_x[], const double obs_seed_y[], int n_seed,
                int frame_count);


// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main()
{
	// sim setup (must match manual_defence)
	double width1 = IMG_W, height1 = IMG_H;
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

	// tuning
	// Geometry: obstacle radius ~32 px (from area=3191), robot half-body
	// ~60 px (D/2 + a bit). Body edge touches obstacle at ~92 px center-
	// to-center. OBS_AVOID needs to be > 92 to react before contact.
	const double FIRE_DIST    = 500.0;  // px: in-range to attempt fire
	const double BACKUP_DIST  = 120.0;  // px: too close, back away
	const double FIRE_TOL     = 0.06;   // rad: aim accuracy required to fire
	const double OBS_AVOID    = 110.0;  // px: obstacle avoid trigger (was 70 - body crashed)
	const double OBS_EMERG    = 80.0;   // px: emergency back-away zone (was 40 - already inside)
	const int    WALL_MARGIN  = 60;     // px: wall standoff
	const int    LOST_THRESH  = 25;     // frames lost -> SEARCH
	const double LOS_BLOCK_R  = 70.0;   // px: perp-dist threshold to block LOS
	const double FRONT_OFFSET = 65.0;   // px: laser exit offset from centre

	cout << "\nAUTO OFFENCE SIMPLE - press space to begin.";
	pause();

	activate_vision();
	activate_simulation(width1, height1, x_obs, y_obs, N_obs,
		"robot_A.bmp", "robot_B.bmp", "background.bmp",
		obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);
	set_simulation_mode(1);
	set_robot_position(500, 100, PI);
	set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

	// images
	image rgb, a, b, rgb0, label;
	rgb.type   = RGB_IMAGE;   rgb.width   = IMG_W; rgb.height   = IMG_H;
	a.type     = GREY_IMAGE;  a.width     = IMG_W; a.height     = IMG_H;
	b.type     = GREY_IMAGE;  b.width     = IMG_W; b.height     = IMG_H;
	rgb0.type  = RGB_IMAGE;   rgb0.width  = IMG_W; rgb0.height  = IMG_H;
	label.type = LABEL_IMAGE; label.width = IMG_W; label.height = IMG_H;
	allocate_image(rgb);  allocate_image(a);   allocate_image(b);
	allocate_image(rgb0); allocate_image(label);

	VisionState vs = {};
	vs.rcx = 500; vs.rcy = 100;   // seed: robot_A start
	vs.ocx = 140; vs.ocy = 400;   // seed: robot_B start

	int    state = 0;       // 0=SEARCH 1=CHASE 2=ATTACK
	int    fired = 0;
	int    lost_frames = 0;
	int    frame_count = 0;
	int    stable_detect = 0;
	int    attack_block  = 0;

	double rtheta_last = PI;
	bool   heading_set = false;

	static const char *snames[] = { "SEARCH", "CHASE", "ATTACK" };

	wait_for_player();
	double tc0 = high_resolution_time(), tc;

	while (1) {

		acquire_image_sim(rgb);
		tc = high_resolution_time() - tc0;
		frame_count++;

		run_vision(rgb, rgb0, a, b, label, vs, x_obs, y_obs, N_obs, frame_count);

		// -- heading: paired markers, fall back to last known --
		double rx = vs.rcx, ry = vs.rcy;
		double dir_to_opp = atan2(vs.ocy - ry, vs.ocx - rx);
		double rtheta;
		if (vs.n_robot >= 2) {
			double raw = atan2(vs.robot_jc[1] - vs.robot_jc[0],
			                   vs.robot_ic[1] - vs.robot_ic[0]);
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

		bool found = (vs.n_opp >= 1);
		if (found) lost_frames = 0;
		else       lost_frames++;

		double dist    = sqrt((vs.ocx-rx)*(vs.ocx-rx) + (vs.ocy-ry)*(vs.ocy-ry));
		double aim_err = wrap(atan2(vs.ocy-ry, vs.ocx-rx) - rtheta);

		// -- obstacle proximity (closest of centre OR front) --
		bool   near_obs = false, obs_emerg = false;
		double obs_min  = 1e9;
		int    obs_k    = -1;
		for (int k = 0; k < vs.n_obs; k++) {
			double od = sqrt((rx-vs.obs_ic[k])*(rx-vs.obs_ic[k]) +
			                 (ry-vs.obs_jc[k])*(ry-vs.obs_jc[k]));
			double fd = sqrt((fx-vs.obs_ic[k])*(fx-vs.obs_ic[k]) +
			                 (fy-vs.obs_jc[k])*(fy-vs.obs_jc[k]));
			double c  = (fd < od) ? fd : od;
			if (c < obs_min) { obs_min = c; obs_k = k; }
			if (c < OBS_AVOID) near_obs = true;
			if (c < OBS_EMERG) obs_emerg = true;
		}

		// -- wall hysteresis (40-frame hold once tripped) --
		static int wall_hold = 0;
		bool wall_hit = (rx < WALL_MARGIN || rx > IMG_W - WALL_MARGIN ||
		                 ry < WALL_MARGIN || ry > IMG_H - WALL_MARGIN ||
		                 fx < WALL_MARGIN || fx > IMG_W - WALL_MARGIN ||
		                 fy < WALL_MARGIN || fy > IMG_H - WALL_MARGIN);
		if (wall_hit) wall_hold = 40;
		else if (wall_hold > 0) wall_hold--;
		bool near_wall = (wall_hold > 0);

		// -- multi-target LOS (per competition rules: any marker hit) --
		double cand_tx[3], cand_ty[3], cand_err[3];
		bool   cand_clear[3];
		const char *cand_name[3];
		int n_cand = 0;
		cand_tx[n_cand] = vs.ocx; cand_ty[n_cand] = vs.ocy;
		cand_name[n_cand++] = "C";
		for (int k = 0; k < vs.n_opp && n_cand < 3; k++) {
			cand_tx[n_cand] = vs.opp_ic[k];
			cand_ty[n_cand] = vs.opp_jc[k];
			cand_name[n_cand++] = (k == 0) ? "M0" : "M1";
		}
		for (int c = 0; c < n_cand; c++) {
			cand_err[c] = wrap(atan2(cand_ty[c]-ry, cand_tx[c]-rx) - rtheta);
			bool blocked = false;
			if (found) {
				double dx = cand_tx[c] - fx, dy = cand_ty[c] - fy;
				double seg = sqrt(dx*dx + dy*dy);
				if (seg > 1.0) {
					double ux = dx/seg, uy = dy/seg;
					for (int k = 0; k < vs.n_obs; k++) {
						double t = (vs.obs_ic[k]-fx)*ux + (vs.obs_jc[k]-fy)*uy;
						if (t < 0 || t > seg) continue;
						double px = fx + t*ux, py = fy + t*uy;
						double perp = sqrt((vs.obs_ic[k]-px)*(vs.obs_ic[k]-px) +
						                   (vs.obs_jc[k]-py)*(vs.obs_jc[k]-py));
						if (perp < LOS_BLOCK_R) { blocked = true; break; }
					}
				}
			}
			cand_clear[c] = !blocked;
		}
		// pick clear candidate with smallest aim error
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

		// -- state transitions with ATTACK persistence --
		bool can_attack = (found && clear_shot && dist < FIRE_DIST);
		if (state == 2 && !can_attack) attack_block++;
		else                            attack_block = 0;
		bool attack_sticky = can_attack ||
		                     (state == 2 && attack_block < 12 &&
		                      found && dist < FIRE_DIST);

		if (lost_frames > LOST_THRESH)               state = 0;
		else if (found && state == 0)                state = 1;
		else if (can_attack)                         state = 2;
		else if (state == 2 && !attack_sticky)       state = 1;

		// -- drive: priority wall > attack-rotate > obstacle > state --
		if (near_wall) {
			double err_c = wrap(atan2(IMG_H/2.0 - ry, IMG_W/2.0 - rx) - rtheta);
			if (fabs(err_c) > 0.3) spin(pw_l, pw_r, err_c);
			else                   drive_smooth(pw_l, pw_r, err_c);

		} else if (state == 2 && clear_shot && !obs_emerg) {
			// have a clear shot - hold position and rotate to aim
			if (dist < BACKUP_DIST) { pw_l = 1500 + 150; pw_r = 1500 - 150; }
			else                     spin(pw_l, pw_r, attack_err);

		} else if ((near_obs || (found && !clear_shot && dist < FIRE_DIST)) &&
		           obs_k >= 0)
		{
			double away = atan2(ry - vs.obs_jc[obs_k], rx - vs.obs_ic[obs_k]);
			if (obs_emerg) {
				double err_a = wrap(away - rtheta);
				if (fabs(err_a) > PI/2.0) { pw_l = 1500 + 200; pw_r = 1500 - 200; }
				else                       drive_smooth(pw_l, pw_r, err_a);
			} else {
				// tangent orbit: cross-product side select, 80-frame commit
				static int side = 0, hold = 0;
				if (hold <= 0) {
					double obs_dx = vs.obs_ic[obs_k] - rx;
					double obs_dy = vs.obs_jc[obs_k] - ry;
					double opp_dx = vs.ocx - rx, opp_dy = vs.ocy - ry;
					side = (obs_dx*opp_dy - obs_dy*opp_dx > 0) ? 1 : -1;
					hold = 80;
				}
				hold--;
				double tang = away + side * (PI / 2.0);
				drive_smooth(pw_l, pw_r, wrap(tang - rtheta));
			}

		} else {
			switch (state) {
			case 0: pw_l = 1500 + 175; pw_r = 1500 + 175; break;  // SEARCH
			case 1: // CHASE
				if (dist < BACKUP_DIST) { pw_l = 1500 + 200; pw_r = 1500 - 200; }
				else                     drive_smooth(pw_l, pw_r, aim_err);
				break;
			case 2: // ATTACK persisted with !clear_shot, no obstacle/wall
				if (dist < BACKUP_DIST) { pw_l = 1500 + 150; pw_r = 1500 - 150; }
				else                     spin(pw_l, pw_r, attack_err);
				break;
			}
		}

		pw_l = clamp_pw(pw_l);
		pw_r = clamp_pw(pw_r);

		// -- fire (one-shot, gated on stability) --
		if (vs.n_robot == 2 && vs.n_opp >= 1) stable_detect++;
		else                                   stable_detect = 0;
		if (!fired && tc > 1.0 && state == 2 && stable_detect > 3 &&
		    fabs(attack_err) < FIRE_TOL && clear_shot && dist < FIRE_DIST)
		{
			fired = 1; laser = 1; pw_laser = 1500;
			pw_l = 1500; pw_r = 1500;
			set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
			view_rgb_image(rgb, 1);
			cout << "\n\n=== LASER FIRED ===";
			cout << "\n  t=" << (int)tc << "s  d=" << (int)dist
			     << "  target=" << aim_tgt
			     << "  aim=" << (int)(attack_err*180.0/PI) << "deg";
			cout << "\nPress space to exit.";
			pause();
			break;
		}

		set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
		view_rgb_image(rgb, 1);

		if (frame_count % 60 == 0) {
			cout << "\n[t=" << (int)tc << "s] " << snames[state]
			     << "  me=(" << (int)rx << "," << (int)ry << ")"
			     << "  opp=(" << (int)vs.ocx << "," << (int)vs.ocy << ")"
			     << "  d=" << (int)dist
			     << "  aim=" << (int)(aim_err*180.0/PI) << "deg"
			     << "  blobs r=" << vs.n_robot << " o=" << vs.n_opp
			     << " obs=" << vs.n_obs;
			if (clear_shot) cout << "  [aim=" << aim_tgt << "]";
			else            cout << "  [BLOCKED]";
		}

		if (KEY('X')) break;
	}

	free_image(rgb);  free_image(a); free_image(b);
	free_image(rgb0); free_image(label);
	deactivate_vision();
	deactivate_simulation();
	cout << "\ndone.\n";
	return 0;
}


// ============================================================
// run_vision -- HSV pipeline + blob classification
// ============================================================
void run_vision(image &rgb, image &rgb0, image &a, image &b, image &label,
                VisionState &vs,
                const double obs_seed_x[], const double obs_seed_y[], int n_seed,
                int frame_count)
{
	const double MIN_AREA = 200.0;
	const double SIZE_THR = 2500.0;
	const double REJECT_R = 90.0;
	const double KOBS_LP  = 0.2;

	// pipeline (prof's assignment 7)
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

	// obstacle position memory, seeded once from main()'s constants.
	// Frozen at n_seed entries -- phantom large blobs can't grow it.
	static double kobs_x[10], kobs_y[10];
	static bool   kobs_seeded = false;
	if (!kobs_seeded) {
		for (int k = 0; k < n_seed; k++) {
			kobs_x[k] = obs_seed_x[k]; kobs_y[k] = obs_seed_y[k];
		}
		kobs_seeded = true;
	}

	// collect features for all blobs once
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

	vs.n_robot = vs.n_opp = vs.n_obs = 0;

	// PASS 1: large blobs -> obstacles. Match to seeded kobs and update.
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
		if (vs.n_obs < 10) {
			vs.obs_ic[vs.n_obs] = blobs[i].ic;
			vs.obs_jc[vs.n_obs] = blobs[i].jc;
			vs.n_obs++;
		}
		draw_point_rgb(rgb, (int)blobs[i].ic, (int)blobs[i].jc, 255, 255, 255);
	}

	// PASS 2: small blobs -> markers. Reject obstacle slices first.
	for (int i = 0; i < n_total; i++) {
		if (blobs[i].area >= SIZE_THR) continue;

		// HSV signature reject:
		//   black obstacle: value < 50 (markers always > 100)
		//   green obstacle: hue 130-180, sat > 0.55, value > 160
		//                   (cyan back marker has hue 154 / sat 0.53)
		bool reject = false;
		if (blobs[i].value < 50) reject = true;
		if (blobs[i].hue >= 130 && blobs[i].hue <= 180 &&
		    blobs[i].sat  >  0.55 && blobs[i].value > 160) reject = true;

		// Marker-color guard for position rejection.
		// Robot markers all have sat >= 0.50 and value >= 110. A small
		// blob with that signature is almost certainly a real marker,
		// even if it's near an obstacle - position-rejecting it would
		// make the robot blind to its own front marker every time it
		// gets close to an obstacle (which is exactly when we need it).
		bool looks_marker = (blobs[i].sat >= 0.45 && blobs[i].value >= 100);

		// position reject for slices near a known obstacle, but only
		// if the blob doesn't look like a real marker
		if (!reject && !looks_marker) {
			for (int k = 0; k < n_seed; k++) {
				double dx = blobs[i].ic - kobs_x[k];
				double dy = blobs[i].jc - kobs_y[k];
				if (sqrt(dx*dx + dy*dy) < REJECT_R) { reject = true; break; }
			}
		}
		if (reject) continue;

		// assign by proximity to last known centroid
		double d_r = (blobs[i].ic-vs.rcx)*(blobs[i].ic-vs.rcx) +
		             (blobs[i].jc-vs.rcy)*(blobs[i].jc-vs.rcy);
		double d_o = (blobs[i].ic-vs.ocx)*(blobs[i].ic-vs.ocx) +
		             (blobs[i].jc-vs.ocy)*(blobs[i].jc-vs.ocy);
		if (d_r <= d_o) {
			if (vs.n_robot < 10) {
				vs.robot_ic[vs.n_robot] = blobs[i].ic;
				vs.robot_jc[vs.n_robot] = blobs[i].jc;
				vs.n_robot++;
			}
			draw_point_rgb(rgb, (int)blobs[i].ic, (int)blobs[i].jc, 0, 255, 0);
		} else {
			if (vs.n_opp < 10) {
				vs.opp_ic[vs.n_opp] = blobs[i].ic;
				vs.opp_jc[vs.n_opp] = blobs[i].jc;
				vs.n_opp++;
			}
			draw_point_rgb(rgb, (int)blobs[i].ic, (int)blobs[i].jc, 0, 255, 255);
		}
	}

	// update centroids (averages of currently visible markers)
	if (vs.n_robot >= 1) {
		double sx = 0, sy = 0;
		for (int k = 0; k < vs.n_robot; k++) { sx += vs.robot_ic[k]; sy += vs.robot_jc[k]; }
		vs.rcx = sx / vs.n_robot;  vs.rcy = sy / vs.n_robot;
		draw_point_rgb(rgb, (int)vs.rcx, (int)vs.rcy, 255, 0, 0);
	}
	if (vs.n_opp >= 1) {
		double sx = 0, sy = 0;
		for (int k = 0; k < vs.n_opp; k++) { sx += vs.opp_ic[k]; sy += vs.opp_jc[k]; }
		vs.ocx = sx / vs.n_opp;  vs.ocy = sy / vs.n_opp;
		draw_point_rgb(rgb, (int)vs.ocx, (int)vs.ocy, 255, 0, 255);
	}

	// frame-1 dump for sanity
	if (frame_count == 1) {
		cout << "\n--- frame 1 blobs (nlabels=" << nlabels << ") ---";
		for (int i = 0; i < n_total; i++) {
			cout << "\n  " << (blobs[i].area >= SIZE_THR ? "OBS" : "MK ")
			     << " (" << (int)blobs[i].ic << "," << (int)blobs[i].jc << ")"
			     << "  hue=" << (int)blobs[i].hue
			     << "  sat=" << blobs[i].sat
			     << "  val=" << (int)blobs[i].value
			     << "  area=" << (int)blobs[i].area;
		}
	}
}


// ============================================================
// prof's features() (vision_example_6.1) and calculate_HSV() (assignment 7)
// ============================================================
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

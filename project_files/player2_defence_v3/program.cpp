
// MECH 472 - player 2 defence
// hide behind an obstacle from the attacker, vision only

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

// forward decls
int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave);

int detect_objects(image &a, image &rgb0, image &label, int nlabels,
                   double robot_ic[], double robot_jc[], int &n_robot_blobs,
                   double &robot_center_ic, double &robot_center_jc,
                   double opp_ic[], double opp_jc[], int &n_opp_blobs,
                   double &opp_center_ic, double &opp_center_jc,
                   double obs_ic[], double obs_jc[], int &n_obs_detected,
                   image &rgb, int frame);

void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value);

// any obstacle within 65 px of segment (ax,ay)->(bx,by)?
bool blocked(double ax, double ay, double bx, double by,
             double ox[], double oy[], int n)
{
	double dx = bx-ax, dy = by-ay;
	double L = sqrt(dx*dx + dy*dy) + 1e-7;
	for (int k = 0; k < n; k++) {
		double rx_ = ox[k]-ax, ry_ = oy[k]-ay;
		double t  = (rx_*dx + ry_*dy) / L;
		if (t < 0 || t > L) continue;
		double px = rx_ - t*dx/L, py = ry_ - t*dy/L;
		if (px*px + py*py < 65*65) return true;
	}
	return false;
}


int main()
{
	const double PI = 3.14159265;

	// sim setup
	double width1 = 640, height1 = 480;
	const int N_obs = 2;
	double x_obs[N_obs] = { 200, 450 };
	double y_obs[N_obs] = { 180, 320 };
	char obstacle_file[N_obs][S_MAX] = {
		"obstacle_black.bmp", "obstacle_black.bmp"
	};
	double D = 121.0, Lx = 31.0, Ly = 0.0, Ax = 37.0, Ay = 0.0;
	double alpha_max = PI / 2.0;
	int n_robot = 2;

	int pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
	double max_speed = 120.0;

	// blob storage (per frame)
	double robot_ic[10], robot_jc[10];
	int    n_robot_blobs;
	double robot_center_ic, robot_center_jc;

	double opp_ic[10], opp_jc[10];
	int    n_opp_blobs;
	double opp_center_ic, opp_center_jc;

	double obs_ic[10], obs_jc[10];
	int    n_obs_detected;

	// remembered obstacle positions
	double kobs_x[10], kobs_y[10];
	int    n_kobs = 0;

	int width = 640, height = 480;

	cout << "\n=== player 2 defence ===";
	cout << "\npress space to begin";
	pause();

	activate_vision();
	activate_simulation(width1, height1,
		x_obs, y_obs, N_obs,
		"robot_A.bmp", "robot_B.bmp", "background.bmp",
		obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);

	set_simulation_mode(2);
	set_robot_position(140, 400, 0.0);
	set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

	// images
	image rgb, a, b, rgb0, label;

	rgb.type   = RGB_IMAGE;   rgb.width   = width; rgb.height  = height;
	a.type     = GREY_IMAGE;  a.width     = width; a.height    = height;
	b.type     = GREY_IMAGE;  b.width     = width; b.height    = height;
	rgb0.type  = RGB_IMAGE;   rgb0.width  = width; rgb0.height = height;
	label.type = LABEL_IMAGE; label.width = width; label.height = height;

	allocate_image(rgb);
	allocate_image(a);
	allocate_image(b);
	allocate_image(rgb0);
	allocate_image(label);

	robot_center_ic = 140; robot_center_jc = 240;
	opp_center_ic   = 500; opp_center_jc   = 240;

	int nlabels;
	int frame = 0;

	join_player();

	double tc0 = high_resolution_time(), tc;

	// ---- main loop ----
	while (1) {

		acquire_image_sim(rgb);
		tc = high_resolution_time() - tc0;
		frame++;

		copy(rgb, rgb0);

		// HSV foreground: sat > 0.20 (5*(mx-mn) > mx) or value < 50
		{
			ibyte *prgb = rgb0.pdata;
			ibyte *pa   = a.pdata;
			int npix = width * height;
			for (int k = 0; k < npix; k++, prgb += 3) {
				int Bk = prgb[0], Gk = prgb[1], Rk = prgb[2];
				int mx = Rk, mn = Rk;
				if (Gk > mx) mx = Gk; else if (Gk < mn) mn = Gk;
				if (Bk > mx) mx = Bk; else if (Bk < mn) mn = Bk;
				if (5*(mx-mn) > mx || mx < 50) pa[k] = 255; else pa[k] = 0;
			}
		}

		// clean: erode x2, dialate x2
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
		               rgb, frame);

		// keep closest pair if extra robot blobs
		if (n_robot_blobs > 2) {
			int bi = 0, bj = 1;
			double md = 1e9;
			for (int i = 0; i < n_robot_blobs; i++)
				for (int j = i+1; j < n_robot_blobs; j++) {
					double dx = robot_ic[i]-robot_ic[j], dy = robot_jc[i]-robot_jc[j];
					double d  = dx*dx + dy*dy;
					if (d < md) { md = d; bi = i; bj = j; }
				}
			robot_ic[0] = robot_ic[bi]; robot_jc[0] = robot_jc[bi];
			robot_ic[1] = robot_ic[bj]; robot_jc[1] = robot_jc[bj];
			robot_center_ic = (robot_ic[0]+robot_ic[1]) / 2.0;
			robot_center_jc = (robot_jc[0]+robot_jc[1]) / 2.0;
			n_robot_blobs = 2;
		}

		// opp needs both markers; park stale centre after 30 lost frames
		static int opp_lost = 0;
		bool opp_visible = (n_opp_blobs >= 2);
		if (opp_visible) opp_lost = 0;
		else if (++opp_lost == 30) {
			opp_center_ic = 640.0 - robot_center_ic;
			opp_center_jc = 480.0 - robot_center_jc;
		}

		// remember obstacles: match nearest within 60 px, low pass 0.20
		// skip when seeing fewer than known (one is being stood on)
		if (n_obs_detected > 0 && n_obs_detected >= n_kobs) {
			for (int j = 0; j < n_obs_detected; j++) {
				int bk = -1; double bd2 = 60*60;
				for (int k = 0; k < n_kobs; k++) {
					double dx = obs_ic[j]-kobs_x[k], dy = obs_jc[j]-kobs_y[k];
					double d2 = dx*dx + dy*dy;
					if (d2 < bd2) { bd2 = d2; bk = k; }
				}
				if (bk >= 0) {
					kobs_x[bk] = 0.8*kobs_x[bk] + 0.2*obs_ic[j];
					kobs_y[bk] = 0.8*kobs_y[bk] + 0.2*obs_jc[j];
				} else if (n_kobs < N_obs) {
					kobs_x[n_kobs] = obs_ic[j];
					kobs_y[n_kobs] = obs_jc[j];
					n_kobs++;
				}
			}
		}

		double rx = robot_center_ic, ry = robot_center_jc;
		static double rt_lp = 0.0;
		static int    rt_init = 0;
		double rtheta;

		// LOS: clear if no obstacle blocks robot-opp line
		bool in_los = !(n_kobs > 0 && opp_visible &&
		                blocked(opp_center_ic, opp_center_jc, rx, ry, kobs_x, kobs_y, n_kobs));

		// hide spot: 110 px behind each obstacle (away from opp)
		// score = d_opp - 0.3*d_rob; +1e8 if shielded by an obstacle
		double hide_ic = rx, hide_jc = ry, best = -1e9;
		for (int k = 0; k < n_kobs; k++) {
			double vx = kobs_x[k]-opp_center_ic, vy = kobs_y[k]-opp_center_jc;
			double vl = sqrt(vx*vx + vy*vy) + 1e-6;
			double hx = kobs_x[k] + 110.0*vx/vl;
			double hy = kobs_y[k] + 110.0*vy/vl;
			if (hx <  40) hx =  40;  if (hx > 600) hx = 600;
			if (hy <  40) hy =  40;  if (hy > 440) hy = 440;
			double dxo = hx-opp_center_ic, dyo = hy-opp_center_jc;
			double dxr = hx-rx,            dyr = hy-ry;
			double score = (dxo*dxo + dyo*dyo) - 0.3*(dxr*dxr + dyr*dyr);
			if (blocked(opp_center_ic, opp_center_jc, hx, hy, kobs_x, kobs_y, n_kobs)) score += 1e8;
			else                                                                       score -= 1e8;
			if (score > best) { best = score; hide_ic = hx; hide_jc = hy; }
		}

		// potential field
		double Fx = 0, Fy = 0;
		double dx = hide_ic-rx, dy = hide_jc-ry;
		double dist = sqrt(dx*dx + dy*dy) + 1e-6;
		Fx += 4.0*dx/dist; Fy += 4.0*dy/dist;          // attraction Ka=4

		for (int k = 0; k < n_kobs; k++) {
			dx = rx-kobs_x[k]; dy = ry-kobs_y[k];
			double d2 = dx*dx + dy*dy + 1e-6;
			Fx += 40000.0*dx/(d2*sqrt(d2));            // 1/r^2 repulsion Kr=40000
			Fy += 40000.0*dy/(d2*sqrt(d2));
		}

		// boundary repulsion Kb=15000
		Fx += 15000.0 / (rx*rx + 1);
		Fx -= 15000.0 / ((640.0-rx)*(640.0-rx) + 1);
		Fy += 15000.0 / (ry*ry + 1);
		Fy -= 15000.0 / ((480.0-ry)*(480.0-ry) + 1);

		double theta_d = atan2(Fy, Fx);

		// heading: pick correct end of marker pair using theta_d, low pass 0.30
		if (n_robot_blobs >= 2) {
			double raw = atan2(robot_jc[1]-robot_jc[0], robot_ic[1]-robot_ic[0]);
			double e1 = theta_d - raw;
			while (e1 >  PI) e1 -= 2*PI;
			while (e1 < -PI) e1 += 2*PI;
			double e2 = theta_d - (raw + PI);
			while (e2 >  PI) e2 -= 2*PI;
			while (e2 < -PI) e2 += 2*PI;
			if (fabs(e1) <= fabs(e2)) rtheta = raw;
			else                      rtheta = raw + PI;
			if (!rt_init) { rt_lp = rtheta; rt_init = 1; }
			double dth = rtheta - rt_lp;
			while (dth >  PI) dth -= 2*PI;
			while (dth < -PI) dth += 2*PI;
			rt_lp += 0.30 * dth;
			rtheta = rt_lp;
		} else {
			// markers lost, hold last heading
			if (rt_init) rtheta = rt_lp;
			else         rtheta = 0.0;
		}

		double th_err = theta_d - rtheta;
		while (th_err >  PI) th_err -= 2*PI;
		while (th_err < -PI) th_err += 2*PI;

		// nav: phase 0 = rotate in place, phase 1 = drive forward
		static int wp = 0;

		double d_wp = sqrt((hide_ic-rx)*(hide_ic-rx) + (hide_jc-ry)*(hide_jc-ry));
		bool near_wall = (rx < 40 || rx > 600 || ry < 40 || ry > 440);

		if (near_wall) {
			// turn back to centre to keep markers on screen
			double err_c = atan2(240.0-ry, 320.0-rx) - rtheta;
			while (err_c >  PI) err_c -= 2*PI;
			while (err_c < -PI) err_c += 2*PI;
			int sign;
			if (err_c >= 0) sign = 1; else sign = -1;
			if (fabs(err_c) > 0.3) {
				pw_l = 1500 + sign*250;
				pw_r = 1500 + sign*250;
			} else {
				int steer = (int)(120.0 * err_c);
				pw_l = 1500 - 250 + steer;
				pw_r = 1500 + 250 + steer;
			}
			wp = 0;

		} else if (!opp_visible) {
			// stop, wait for opp to come back
			pw_r = 1500; pw_l = 1500;
			wp = 0;

		} else if (wp == 0) {
			// rotate to face hide spot
			if (fabs(th_err) < 0.20) {
				wp = 1;
			} else {
				int turn = (int)(350.0 * th_err);
				if (turn >  450) turn =  450;
				if (turn < -450) turn = -450;
				pw_r = 1500 + turn;
				pw_l = 1500 + turn;
			}

		} else {
			// drive forward, re-align if heading drifts
			if (d_wp < 30.0) {
				pw_r = 1500; pw_l = 1500;
			} else if (fabs(th_err) > 0.50) {
				wp = 0;
			} else {
				int fwd = (int)(3.5 * d_wp);
				if (in_los) {
					if (fwd > 420) fwd = 420;
					if (fwd < 280) fwd = 280;
				} else {
					if (fwd > 380) fwd = 380;
					if (fwd < 120) fwd = 120;
				}
				int turn = (int)(180.0 * th_err);
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

		// status print every 60 frames
		if (frame % 60 == 0) {
			cout << "\n[t=" << (int)tc << "s]"
			     << " r=(" << (int)rx << "," << (int)ry << ")"
			     << " opp=(" << (int)opp_center_ic << "," << (int)opp_center_jc << ")"
			     << " hide=(" << (int)hide_ic << "," << (int)hide_jc << ")"
			     << " d_wp=" << (int)d_wp
			     << " th_err=" << (int)(th_err*180.0/PI) << "deg"
			     << " los=" << in_los
			     << " pw=" << pw_l << "/" << pw_r
			     << " r=" << n_robot_blobs
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


// ---- detect_objects ----
int detect_objects(image &a, image &rgb0, image &label, int nlabels,
                   double robot_ic[], double robot_jc[], int &n_robot_blobs,
                   double &robot_center_ic, double &robot_center_jc,
                   double opp_ic[], double opp_jc[], int &n_opp_blobs,
                   double &opp_center_ic, double &opp_center_jc,
                   double obs_ic[], double obs_jc[], int &n_obs_detected,
                   image &rgb, int frame)
{
	const double min_area       = 200.0;
	const double size_threshold = 2500.0;

	double ic, jc, area, R_ave, G_ave, B_ave;
	double hue, sat, value;
	int    R, G, B;
	const char *name;

	n_robot_blobs  = 0;
	n_opp_blobs    = 0;
	n_obs_detected = 0;

	for (int lbl = 1; lbl <= nlabels; lbl++) {

		features(a, rgb0, label, lbl, ic, jc, area, R_ave, G_ave, B_ave);
		if (area < min_area) continue;

		calculate_HSV((int)R_ave, (int)G_ave, (int)B_ave, hue, sat, value);

		if (area < size_threshold) {
			// small blob = robot marker, assigned by proximity to last centre
			double d_robot = (ic-robot_center_ic)*(ic-robot_center_ic)
			               + (jc-robot_center_jc)*(jc-robot_center_jc);
			double d_opp   = (ic-opp_center_ic)*(ic-opp_center_ic)
			               + (jc-opp_center_jc)*(jc-opp_center_jc);
			if (d_robot <= d_opp) {
				name = "robot";
				if (n_robot_blobs < 10) { robot_ic[n_robot_blobs] = ic; robot_jc[n_robot_blobs] = jc; n_robot_blobs++; }
				R = 0; G = 255; B = 0;
			} else {
				name = "opponent";
				if (n_opp_blobs < 10) { opp_ic[n_opp_blobs] = ic; opp_jc[n_opp_blobs] = jc; n_opp_blobs++; }
				R = 0; G = 255; B = 255;
			}
		} else {
			// large blob = obstacle, classify by colour
			if      (value < 50)                              name = "obstacle_black";
			else if ((hue >= 340 || hue <= 20) && sat > 0.4)  name = "obstacle_red";
			else if (hue > 20  && hue <= 45  && sat > 0.5)    name = "obstacle_orange";
			else if (hue > 80  && hue <= 160 && sat > 0.3)    name = "obstacle_green";
			else if (hue > 190 && hue <= 260 && sat > 0.3)    name = "obstacle_blue";
			else                                              name = "unknown";
			if (n_obs_detected < 10) { obs_ic[n_obs_detected] = ic; obs_jc[n_obs_detected] = jc; n_obs_detected++; }
			R = 255; G = 255; B = 255;
		}

		draw_point_rgb(rgb, (int)ic, (int)jc, R, G, B);

		if (frame == 1) {
			cout << "\nlabel " << lbl << " [" << name << "]";
			cout << "  c: (" << (int)ic << ", " << (int)jc << ")";
			cout << "  HSV: (" << (int)hue << ", " << sat << ", " << (int)value << ")";
			cout << "  area: " << (int)area;
		}
	}

	if (frame == 1) cout << "\nnlabels = " << nlabels;

	// robot centre from detected blobs
	if (n_robot_blobs >= 1) {
		robot_center_ic = 0; robot_center_jc = 0;
		for (int k = 0; k < n_robot_blobs; k++) {
			robot_center_ic += robot_ic[k];
			robot_center_jc += robot_jc[k];
		}
		robot_center_ic /= n_robot_blobs;
		robot_center_jc /= n_robot_blobs;
		draw_point_rgb(rgb, (int)robot_center_ic, (int)robot_center_jc, 255, 0, 0);
	}

	// opp centre only when both markers seen
	if (n_opp_blobs >= 2) {
		opp_center_ic = 0; opp_center_jc = 0;
		for (int k = 0; k < n_opp_blobs; k++) {
			opp_center_ic += opp_ic[k];
			opp_center_jc += opp_jc[k];
		}
		opp_center_ic /= n_opp_blobs;
		opp_center_jc /= n_opp_blobs;
		draw_point_rgb(rgb, (int)opp_center_ic, (int)opp_center_jc, 255, 0, 255);
	}

	return 0;
}


// ---- features ----
int features(image &a, image &rgb, image &label, int label_i,
             double &ic, double &jc, double &area,
             double &R_ave, double &G_ave, double &B_ave)
{
	ibyte  *p, *pc;
	i2byte *pl;
	i4byte  i, j, k, width, height;
	double  mi, mj, m, EPS = 1e-7, n;
	double  R_sum = 0, G_sum = 0, B_sum = 0;
	int     Rv, Gv, Bv;

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
				m++;
				mi += i;
				mj += j;
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


// ---- HSV from RGB ----
void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value)
{
	int mx = R, mn = R;
	if (G > mx) mx = G; if (B > mx) mx = B;
	if (G < mn) mn = G; if (B < mn) mn = B;
	int delta = mx - mn;

	value = mx;
	if (delta == 0) sat = 0.0; else sat = (double)delta / value;

	double H;
	if      (delta == 0) H = 0;
	else if (mx == R)    H = (double)(G - B) / delta;
	else if (mx == G)    H = (double)(B - R) / delta + 2;
	else                 H = (double)(R - G) / delta + 4;

	hue = 60 * H;
	if (hue < 0) hue += 360;
}

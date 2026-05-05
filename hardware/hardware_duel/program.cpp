// hardware_duel - run offence + defence from one camera, two BT links

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <windows.h>
#include "image_transfer.h"
#include "vision.h"
#include "timer.h"
#include "serial_com.h"

#define KEY(c) ( GetAsyncKeyState((int)(c)) & (SHORT)0x8000 )

using namespace std;

static char COM_OFFENSE[] = "COM9";
static char COM_DEFENSE[] = "COM5";

static const int    TH_STOP        = 90;
static const int    DTH_FWD        = 30;
static const int    DTH_TURN       = 30;

static const int    MIN_AREA       = 30;
static const double MIN_PAIR_SEP   = 30.0;
static const double MAX_PAIR_SEP   = 150.0;

static const double OFF_ALIGN_TOL  = 0.25;   // rad
static const double OFF_STOP_DIST  = 140.0;
static const double OFF_AVOID_DIST = 110.0;
static const double OFF_K_REP      = 3.0;

static const double DEF_ALIGN_TOL      = 0.20;
static const double DEF_REALIGN_TOL    = 0.50;
static const double DEF_STEER_DEADBAND = 0.08;
static const double DEF_STOP_DIST      = 30.0;
static const int    DEF_NEAR_WALL      = 40;
static const double DEF_HIDE_OFFSET    = 160.0;
static const double DEF_LOS_BLOCK_R    = 65.0;
static const double DEF_K_ATTRACT      = 4.0;
static const double DEF_K_REPEL        = 40000.0;
static const double DEF_K_BOUND        = 10000.0;
static const double DEF_HEADING_LP     = 0.30;

static const int    BLACK_HARD_MAX = 60;
static const int    BLACK_SOFT_MAX = 110;
static const double BLACK_SAT_MAX  = 0.35;

static const int    OBS_MIN_AREA      = 150;
static const int    OBS_BORDER_MARGIN = 40;
static const double OWN_BODY_RADIUS   = 60.0;
static const double ASPECT_MAX        = 1.5;
static const double EXTENT_MIN        = 0.55;
static const double EXTENT_MAX        = 0.95;
static const int    N_OBS             = 2;

static const int    WHITE_MIN         = 150;
static const int    ARENA_MIN_AREA    = 8000;

static const double K_BARREL          = 0.15;
static const double IMG_CX            = 320.0;
static const double IMG_CY            = 240.0;
static const double IMG_F             = 320.0;

static const double HIDE_LP_A         = 0.12;   // hide-spot LP weight


void calculate_HSV(int R, int G, int B, double &hue, double &sat, double &value)
{
	int max, min, delta; double H;
	max = min = R;
	if (G > max) max = G;  if (B > max) max = B;
	if (G < min) min = G;  if (B < min) min = B;
	delta = max - min; value = max;
	sat = (delta == 0) ? 0.0 : (double)delta / value;
	if (delta == 0)      H = 0;
	else if (max == R)   H = (double)(G - B) / delta;
	else if (max == G)   H = (double)(B - R) / delta + 2;
	else                 H = (double)(R - G) / delta + 4;
	hue = 60 * H;
	if (hue < 0) hue += 360;
}

void hsv_threshold(image &rgb, image &binary,
                   double hue_target, double hue_tol,
                   double sat_min, double sat_max, double val_min)
{
	int npix = rgb.width * rgb.height;
	ibyte *p = rgb.pdata, *q = binary.pdata;
	for (int i = 0; i < npix; i++) {
		int B = p[0], G = p[1], R = p[2];
		double h, s, v;
		calculate_HSV(R, G, B, h, s, v);
		double dh = fabs(h - hue_target);
		if (dh > 180.0) dh = 360.0 - dh;
		*q = (dh < hue_tol && s > sat_min && s < sat_max && v > val_min) ? 255 : 0;
		p += 3; q += 1;
	}
}

void black_threshold(image &rgb, image &binary,
                     int hard_max, int soft_max, double sat_max)
{
	int npix = rgb.width * rgb.height;
	ibyte *p = rgb.pdata, *q = binary.pdata;
	for (int i = 0; i < npix; i++) {
		int B = p[0], G = p[1], R = p[2];
		int max_v = R;
		if (G > max_v) max_v = G;
		if (B > max_v) max_v = B;
		bool is_black;
		if (max_v < hard_max) {
			is_black = true;
		} else if (max_v < soft_max) {
			int min_v = R;
			if (G < min_v) min_v = G;
			if (B < min_v) min_v = B;
			double sat = (max_v == 0) ? 0.0 : (double)(max_v - min_v) / max_v;
			is_black = (sat < sat_max);
		} else {
			is_black = false;
		}
		*q = is_black ? 255 : 0;
		p += 3; q += 1;
	}
}

void mask_in(image &binary, image &arena_mask)
{
	int npix = binary.width * binary.height;
	ibyte *b = binary.pdata, *m = arena_mask.pdata;
	for (int i = 0; i < npix; i++) if (m[i] == 0) b[i] = 0;
}

int largest_label(image &binary, image &label, int nlabels,
                  int &ic, int &jc, int &area)
{
	int counts[256];
	if (nlabels <= 0) { ic = jc = -1; area = 0; return 0; }
	if (nlabels > 255) nlabels = 255;
	for (int k = 0; k <= nlabels; k++) counts[k] = 0;
	i2byte *lp = (i2byte *)label.pdata;
	for (int idx = 0; idx < label.width * label.height; idx++) {
		int v = lp[idx];
		if (v > 0 && v <= nlabels) counts[v]++;
	}
	int best_k = 0; area = 0;
	for (int k = 1; k <= nlabels; k++) {
		if (counts[k] > area) { area = counts[k]; best_k = k; }
	}
	if (best_k == 0) { ic = jc = -1; return 0; }
	double dic, djc;
	centroid(binary, label, best_k, dic, djc);
	ic = (int)dic; jc = (int)djc;
	return best_k;
}

int nearest_label(image &binary, image &label, int nlabels,
                  double near_x, double near_y, double max_dist, int min_area,
                  int &ic, int &jc, int &area)
{
	if (nlabels <= 0) { ic = jc = -1; area = 0; return 0; }
	if (nlabels > 255) nlabels = 255;
	int counts[256];
	for (int k = 0; k <= nlabels; k++) counts[k] = 0;
	i2byte *lp = (i2byte *)label.pdata;
	for (int idx = 0; idx < label.width * label.height; idx++) {
		int v = lp[idx];
		if (v > 0 && v <= nlabels) counts[v]++;
	}
	int best_k = 0, best_a = 0;
	double best_d = max_dist;
	for (int k = 1; k <= nlabels; k++) {
		if (counts[k] < min_area) continue;
		double dic, djc;
		centroid(binary, label, k, dic, djc);
		double dx = dic - near_x, dy = djc - near_y;
		double d = sqrt(dx*dx + dy*dy);
		if (d < best_d) { best_d = d; best_k = k; best_a = counts[k]; }
	}
	if (best_k == 0) { ic = jc = -1; area = 0; return 0; }
	double dic, djc;
	centroid(binary, label, best_k, dic, djc);
	ic = (int)dic; jc = (int)djc;
	area = best_a;
	return best_k;
}

void barrel_correct(double ic, double jc, double &ic_out, double &jc_out)
{
	double xn = (ic - IMG_CX) / IMG_F;
	double yn = (jc - IMG_CY) / IMG_F;
	double r2 = xn*xn + yn*yn;
	double s  = 1.0 + K_BARREL * r2;
	ic_out = IMG_CX + xn * s * IMG_F;
	jc_out = IMG_CY + yn * s * IMG_F;
}

double angle_diff(double a, double b)
{
	double d = a - b;
	while (d >  M_PI) d -= 2*M_PI;
	while (d < -M_PI) d += 2*M_PI;
	return d;
}

bool segment_blocked(double ax, double ay, double bx, double by,
                     double ox[], double oy[], int n)
{
	double dx = bx - ax, dy = by - ay;
	double L  = sqrt(dx*dx + dy*dy) + 1e-7;
	for (int k = 0; k < n; k++) {
		double rx = ox[k] - ax, ry = oy[k] - ay;
		double t  = (rx*dx + ry*dy) / L;
		if (t < 0 || t > L) continue;
		double px = rx - t*dx/L, py = ry - t*dy/L;
		if (px*px + py*py < DEF_LOS_BLOCK_R*DEF_LOS_BLOCK_R) return true;
	}
	return false;
}


int main()
{
	int width  = 640;
	int height = 480;
	int cam_number = 1;

	image rgb, grey, binary, label, arena_mask;
	int nlabels;

	HANDLE h_off, h_def;
	const int NMAX = 64;
	char buf_off[NMAX], buf_def[NMAX];
	int n;
	unsigned char start_char = 255;

	struct color_target {
		double hue_target, hue_tol, sat_min, sat_max, val_min;
		int    ic, jc, area;
	};
	color_target ct[4] = {
		{ 193.0, 15.0, 0.50, 1.01,  30.0,   0, 0, 0 },  // off-F blue
		{  13.0,  8.0, 0.75, 1.01, 100.0,   0, 0, 0 },  // off-B orange
		{ 125.0, 40.0, 0.25, 1.01,  10.0,   0, 0, 0 },  // def-F green
		{   0.0,  8.0, 0.65, 1.01,  50.0,   0, 0, 0 },  // def-B purple
	};

	struct obstacle { double ic, jc; int area; };
	obstacle obs[N_OBS];
	int      n_obs = 0;
	obstacle locked_obs[N_OBS];
	int      locked_n   = 0;
	bool     obs_locked = false;

	double def_rt_lp = 0.0;
	bool   def_rt_init = false;
	int    def_wp = 0;

	double hide_x_lp = 0.0, hide_y_lp = 0.0;
	bool   hide_lp_init = false;

	double def_x_last = 320.0, def_y_last = 240.0;
	bool   def_have_last = false;
	int    def_lost_frames = 0;

	double off_x_last = 320.0, off_y_last = 240.0;
	bool   off_have_last = false;
	int    off_lost_frames = 0;

	if (open_serial(COM_OFFENSE, h_off, 0)) return 1;
	if (open_serial(COM_DEFENSE, h_def, 0)) { close_serial(h_off); return 1; }

	while (!KEY(VK_SPACE)) Sleep(1);
	buf_off[0] = 's';
	buf_def[0] = 's';
	if (serial_send(buf_off, 1, h_off)) { close_serial(h_off); close_serial(h_def); return 1; }
	if (serial_send(buf_def, 1, h_def)) { close_serial(h_off); close_serial(h_def); return 1; }
	Sleep(100);

	activate_vision();
	activate_camera(cam_number, height, width);

	rgb.type        = RGB_IMAGE;   rgb.width        = width; rgb.height        = height; allocate_image(rgb);
	grey.type       = GREY_IMAGE;  grey.width       = width; grey.height       = height; allocate_image(grey);
	binary.type     = GREY_IMAGE;  binary.width     = width; binary.height     = height; allocate_image(binary);
	label.type      = LABEL_IMAGE; label.width      = width; label.height      = height; allocate_image(label);
	arena_mask.type = GREY_IMAGE;  arena_mask.width = width; arena_mask.height = height; allocate_image(arena_mask);

	bool arena_active = false;

	int off_th_left = TH_STOP, off_th_right = TH_STOP;
	int def_th_left = TH_STOP, def_th_right = TH_STOP;

	while (1) {
		acquire_image(rgb, cam_number);

		// arena mask
		{
			int npix = rgb.width * rgb.height;
			ibyte *p = rgb.pdata, *q = binary.pdata;
			for (int i = 0; i < npix; i++) {
				int B = p[0], G = p[1], R = p[2];
				q[i] = (R > WHITE_MIN && G > WHITE_MIN && B > WHITE_MIN) ? 255 : 0;
				p += 3;
			}
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			int alabels;
			label_image(binary, label, alabels);

			int counts[256];
			int nl = (alabels > 255) ? 255 : alabels;
			for (int k = 0; k <= nl; k++) counts[k] = 0;
			i2byte *lp = (i2byte *)label.pdata;
			for (int idx = 0; idx < npix; idx++) {
				int v = lp[idx];
				if (v > 0 && v <= nl) counts[v]++;
			}
			int best_k = 0, best_a = 0;
			for (int k = 1; k <= nl; k++) {
				if (counts[k] > best_a) { best_a = counts[k]; best_k = k; }
			}
			arena_active = (best_a >= ARENA_MIN_AREA);

			ibyte *am = arena_mask.pdata;
			if (arena_active) {
				for (int idx = 0; idx < npix; idx++) am[idx] = (lp[idx] == best_k) ? 255 : 0;
			} else {
				for (int idx = 0; idx < npix; idx++) am[idx] = 255;
			}
		}

		// off-F (blue) - largest
		{
			color_target &t = ct[0];
			hsv_threshold(rgb, binary, t.hue_target, t.hue_tol, t.sat_min, t.sat_max, t.val_min);
			mask_in(binary, arena_mask);
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			label_image(binary, label, nlabels);
			largest_label(binary, label, nlabels, t.ic, t.jc, t.area);
		}
		bool off_F_seen = (ct[0].area > MIN_AREA);

		// off-B (orange) - anchor on off-F
		{
			color_target &t = ct[1];
			hsv_threshold(rgb, binary, t.hue_target, t.hue_tol, t.sat_min, t.sat_max, t.val_min);
			mask_in(binary, arena_mask);
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			label_image(binary, label, nlabels);
			if (off_F_seen) {
				nearest_label(binary, label, nlabels,
				              ct[0].ic, ct[0].jc, MAX_PAIR_SEP, MIN_AREA,
				              t.ic, t.jc, t.area);
			} else {
				largest_label(binary, label, nlabels, t.ic, t.jc, t.area);
			}
		}
		bool off_B_seen = (ct[1].area > MIN_AREA);

		// def-F (green) - largest
		{
			color_target &t = ct[2];
			hsv_threshold(rgb, binary, t.hue_target, t.hue_tol, t.sat_min, t.sat_max, t.val_min);
			mask_in(binary, arena_mask);
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			label_image(binary, label, nlabels);
			largest_label(binary, label, nlabels, t.ic, t.jc, t.area);
		}
		bool def_F_seen = (ct[2].area > MIN_AREA);

		// def-B (purple) - anchor on def-F
		{
			color_target &t = ct[3];
			hsv_threshold(rgb, binary, t.hue_target, t.hue_tol, t.sat_min, t.sat_max, t.val_min);
			mask_in(binary, arena_mask);
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			label_image(binary, label, nlabels);
			if (def_F_seen) {
				nearest_label(binary, label, nlabels,
				              ct[2].ic, ct[2].jc, MAX_PAIR_SEP, MIN_AREA,
				              t.ic, t.jc, t.area);
			} else {
				largest_label(binary, label, nlabels, t.ic, t.jc, t.area);
			}
		}
		bool def_B_seen = (ct[3].area > MIN_AREA);

		// obstacles - shape-based, locked once both detected
		if (obs_locked) {
			n_obs = locked_n;
			for (int k = 0; k < locked_n; k++) obs[k] = locked_obs[k];
		} else {
			black_threshold(rgb, binary, BLACK_HARD_MAX, BLACK_SOFT_MAX, BLACK_SAT_MAX);
			mask_in(binary, arena_mask);
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			label_image(binary, label, nlabels);
			n_obs = 0;

			int nl = (nlabels > 255) ? 255 : nlabels;
			int counts[256];
			int xmn[256], xmx[256], ymn[256], ymx[256];
			double sumi[256], sumj[256];
			for (int k = 0; k <= nl; k++) {
				counts[k] = 0;
				xmn[k] = width;  xmx[k] = -1;
				ymn[k] = height; ymx[k] = -1;
				sumi[k] = 0.0;   sumj[k] = 0.0;
			}
			i2byte *lp = (i2byte *)label.pdata;
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					int v = lp[j*width + i];
					if (v <= 0 || v > nl) continue;
					counts[v]++;
					sumi[v] += i;
					sumj[v] += j;
					if (i < xmn[v]) xmn[v] = i;
					if (i > xmx[v]) xmx[v] = i;
					if (j < ymn[v]) ymn[v] = j;
					if (j > ymx[v]) ymx[v] = j;
				}
			}

			struct cand { int area; double ic, jc; };
			cand cands[64]; int n_cands = 0;
			for (int k = 1; k <= nl && n_cands < 64; k++) {
				int a = counts[k];
				if (a < OBS_MIN_AREA) continue;
				int bw = xmx[k] - xmn[k] + 1;
				int bh = ymx[k] - ymn[k] + 1;
				if (bw <= 0 || bh <= 0) continue;
				double aspect = (bw > bh) ? (double)bw/bh : (double)bh/bw;
				double extent = (double)a / (double)(bw * bh);
				if (aspect > ASPECT_MAX)              continue;
				if (extent < EXTENT_MIN || extent > EXTENT_MAX) continue;

				double ic = sumi[k] / a;
				double jc = sumj[k] / a;
				if (ic < OBS_BORDER_MARGIN || ic > width  - OBS_BORDER_MARGIN ||
				    jc < OBS_BORDER_MARGIN || jc > height - OBS_BORDER_MARGIN) continue;

				bool own = false;
				for (int c = 0; c < 4 && !own; c++) {
					if (ct[c].area <= MIN_AREA) continue;
					double dx = ic - ct[c].ic, dy = jc - ct[c].jc;
					if (sqrt(dx*dx + dy*dy) < OWN_BODY_RADIUS) own = true;
				}
				if (own) continue;

				cands[n_cands].area = a;
				cands[n_cands].ic   = ic;
				cands[n_cands].jc   = jc;
				n_cands++;
			}

			for (int slot = 0; slot < N_OBS; slot++) {
				int best = -1;
				for (int i = 0; i < n_cands; i++) {
					if (cands[i].area < 0) continue;
					if (best < 0 || cands[i].area > cands[best].area) best = i;
				}
				if (best < 0) break;
				barrel_correct(cands[best].ic, cands[best].jc,
				               obs[n_obs].ic, obs[n_obs].jc);
				obs[n_obs].area = cands[best].area;
				n_obs++;
				cands[best].area = -1;
			}

			if (n_obs == N_OBS) {
				for (int k = 0; k < n_obs; k++) locked_obs[k] = obs[k];
				locked_n = n_obs;
				obs_locked = true;
			}
		}

		// barrel-corrected centroids
		double oF_x, oF_y, oB_x, oB_y, dF_x, dF_y, dB_x, dB_y;
		barrel_correct(ct[0].ic, ct[0].jc, oF_x, oF_y);
		barrel_correct(ct[1].ic, ct[1].jc, oB_x, oB_y);
		barrel_correct(ct[2].ic, ct[2].jc, dF_x, dF_y);
		barrel_correct(ct[3].ic, ct[3].jc, dB_x, dB_y);

		// offense pose
		double off_x = off_x_last, off_y = off_y_last;
		double off_heading = 0.0;
		bool   off_pos = false, off_heading_valid = false;
		if (off_F_seen && off_B_seen) {
			double dx = oF_x - oB_x, dy = oF_y - oB_y;
			double sep = sqrt(dx*dx + dy*dy);
			if (sep >= MIN_PAIR_SEP && sep <= MAX_PAIR_SEP) {
				off_x = 0.5 * (oF_x + oB_x);
				off_y = 0.5 * (oF_y + oB_y);
				off_heading = atan2(dy, dx);
				off_pos = true; off_heading_valid = true;
				off_x_last = off_x; off_y_last = off_y;
				off_have_last = true; off_lost_frames = 0;
			}
		}
		if (!off_pos && off_F_seen) { off_x = oF_x; off_y = oF_y; off_pos = true; }
		else if (!off_pos && off_B_seen) { off_x = oB_x; off_y = oB_y; off_pos = true; }
		if (!off_heading_valid) off_lost_frames++;

		// defense pose
		double def_x = def_x_last, def_y = def_y_last;
		bool   def_pos = false;
		bool   def_raw_heading_ok = false;
		double def_raw_heading = 0.0;
		if (def_F_seen && def_B_seen) {
			double dx = dF_x - dB_x, dy = dF_y - dB_y;
			double sep = sqrt(dx*dx + dy*dy);
			if (sep >= MIN_PAIR_SEP && sep <= MAX_PAIR_SEP) {
				def_x = 0.5 * (dF_x + dB_x);
				def_y = 0.5 * (dF_y + dB_y);
				def_pos = true;
				def_raw_heading = atan2(dy, dx);
				def_raw_heading_ok = true;
				def_x_last = def_x; def_y_last = def_y;
				def_have_last = true; def_lost_frames = 0;
			}
		}
		if (!def_pos && def_F_seen) { def_x = dF_x; def_y = dF_y; def_pos = true; }
		else if (!def_pos && def_B_seen) { def_x = dB_x; def_y = dB_y; def_pos = true; }
		if (!def_raw_heading_ok) def_lost_frames++;

		double obs_x[N_OBS], obs_y[N_OBS];
		for (int k = 0; k < n_obs; k++) { obs_x[k] = obs[k].ic; obs_y[k] = obs[k].jc; }

		// OFFENSE: chase defense via potential field
		double off_desired = 0.0, off_head_err = 0.0, off_dist = 0.0;
		bool   def_visible = (def_F_seen && def_B_seen) ||
		                     (def_have_last && def_lost_frames < 30);
		if (off_pos && def_visible) {
			double dx = def_x - off_x;
			double dy = def_y - off_y;
			off_dist = sqrt(dx*dx + dy*dy);
			double ax = dx / (off_dist + 1e-6);
			double ay = dy / (off_dist + 1e-6);
			double rx = 0.0, ry = 0.0;
			for (int k = 0; k < n_obs; k++) {
				double ox = off_x - obs[k].ic;
				double oy = off_y - obs[k].jc;
				double od = sqrt(ox*ox + oy*oy);
				if (od < OFF_AVOID_DIST && od > 1.0) {
					double w = (OFF_AVOID_DIST - od) / OFF_AVOID_DIST;
					rx += OFF_K_REP * w * (ox / od);
					ry += OFF_K_REP * w * (oy / od);
				}
			}
			off_desired = atan2(ay + ry, ax + rx);
			if (off_heading_valid) off_head_err = angle_diff(off_desired, off_heading);
		}

		off_th_left = TH_STOP; off_th_right = TH_STOP;
		if (!off_heading_valid || !def_visible) {
			// LOST
		} else if (off_dist < OFF_STOP_DIST) {
			// AT_TGT
		} else if (fabs(off_head_err) > OFF_ALIGN_TOL) {
			if (off_head_err > 0) {
				off_th_left  = TH_STOP - DTH_TURN;
				off_th_right = TH_STOP + DTH_TURN;
			} else {
				off_th_left  = TH_STOP + DTH_TURN;
				off_th_right = TH_STOP - DTH_TURN;
			}
		} else {
			off_th_left  = TH_STOP + DTH_FWD;
			off_th_right = TH_STOP + DTH_FWD;
		}

		// DEFENSE: hide behind obstacle from offense
		bool off_visible = off_heading_valid ||
		                   (off_have_last && off_lost_frames < 30);

		double hide_x = def_x, hide_y = def_y;
		double best_score = -1e18;
		for (int k = 0; k < n_obs; k++) {
			double vx = obs[k].ic - off_x, vy = obs[k].jc - off_y;
			double vl = sqrt(vx*vx + vy*vy) + 1e-6;
			double hx = obs[k].ic + DEF_HIDE_OFFSET * vx / vl;
			double hy = obs[k].jc + DEF_HIDE_OFFSET * vy / vl;
			if (hx <  40) hx =  40;
			if (hx > 600) hx = 600;
			if (hy <  40) hy =  40;
			if (hy > 440) hy = 440;
			double dxo = hx - off_x, dyo = hy - off_y;
			double dxr = hx - def_x, dyr = hy - def_y;
			double score = (dxo*dxo + dyo*dyo) - 0.3 * (dxr*dxr + dyr*dyr);
			if (segment_blocked(off_x, off_y, hx, hy, obs_x, obs_y, n_obs)) score += 1e8;
			else                                                            score -= 1e8;
			if (score > best_score) { best_score = score; hide_x = hx; hide_y = hy; }
		}

		// LP the hide spot - raw can teleport when picker swaps obstacles
		if (!hide_lp_init) {
			hide_x_lp = hide_x; hide_y_lp = hide_y; hide_lp_init = true;
		} else {
			hide_x_lp = (1.0 - HIDE_LP_A) * hide_x_lp + HIDE_LP_A * hide_x;
			hide_y_lp = (1.0 - HIDE_LP_A) * hide_y_lp + HIDE_LP_A * hide_y;
		}
		hide_x = hide_x_lp;
		hide_y = hide_y_lp;

		// potential field
		double Fx = 0.0, Fy = 0.0;
		double dxh = hide_x - def_x, dyh = hide_y - def_y;
		double dh  = sqrt(dxh*dxh + dyh*dyh) + 1e-6;
		Fx += DEF_K_ATTRACT * dxh / dh;
		Fy += DEF_K_ATTRACT * dyh / dh;
		for (int k = 0; k < n_obs; k++) {
			double dxo = def_x - obs[k].ic, dyo = def_y - obs[k].jc;
			double d2  = dxo*dxo + dyo*dyo + 1e-6;
			double d3  = d2 * sqrt(d2);
			Fx += DEF_K_REPEL * dxo / d3;
			Fy += DEF_K_REPEL * dyo / d3;
		}
		Fx += DEF_K_BOUND / (def_x*def_x + 1.0);
		Fx -= DEF_K_BOUND / ((640.0 - def_x)*(640.0 - def_x) + 1.0);
		Fy += DEF_K_BOUND / (def_y*def_y + 1.0);
		Fy -= DEF_K_BOUND / ((480.0 - def_y)*(480.0 - def_y) + 1.0);
		double theta_d = atan2(Fy, Fx);

		// heading: pick correct end + low-pass
		double rtheta;
		if (def_raw_heading_ok) {
			double e1 = angle_diff(theta_d, def_raw_heading);
			double e2 = angle_diff(theta_d, def_raw_heading + M_PI);
			double rt = (fabs(e1) <= fabs(e2)) ? def_raw_heading : (def_raw_heading + M_PI);
			if (!def_rt_init) { def_rt_lp = rt; def_rt_init = true; }
			double dth = angle_diff(rt, def_rt_lp);
			def_rt_lp += DEF_HEADING_LP * dth;
			while (def_rt_lp >  M_PI) def_rt_lp -= 2*M_PI;
			while (def_rt_lp < -M_PI) def_rt_lp += 2*M_PI;
			rtheta = def_rt_lp;
		} else if (def_rt_init) {
			rtheta = def_rt_lp;
		} else {
			rtheta = 0.0;
		}

		double th_err = angle_diff(theta_d, rtheta);
		double d_wp   = sqrt(dxh*dxh + dyh*dyh);
		bool   def_near_wall = (def_x < DEF_NEAR_WALL || def_x > 640.0 - DEF_NEAR_WALL ||
		                        def_y < DEF_NEAR_WALL || def_y > 480.0 - DEF_NEAR_WALL);

		def_th_left = TH_STOP; def_th_right = TH_STOP;
		if (!def_pos || !def_raw_heading_ok) {
			// LOST
		} else if (def_near_wall) {
			double err_c = angle_diff(atan2(240.0 - def_y, 320.0 - def_x), rtheta);
			if (fabs(err_c) > 0.3) {
				int sign = (err_c >= 0) ? 1 : -1;
				def_th_left  = TH_STOP - sign * DTH_TURN;
				def_th_right = TH_STOP + sign * DTH_TURN;
			} else {
				int steer = (int)(20.0 * err_c);
				def_th_left  = TH_STOP + DTH_FWD - steer;
				def_th_right = TH_STOP + DTH_FWD + steer;
			}
			def_wp = 0;
		} else if (!off_visible) {
			def_wp = 0;
		} else if (def_wp == 0) {
			if (fabs(th_err) < DEF_ALIGN_TOL) {
				def_wp = 1;
			} else {
				int sign = (th_err >= 0) ? 1 : -1;
				def_th_left  = TH_STOP - sign * DTH_TURN;
				def_th_right = TH_STOP + sign * DTH_TURN;
			}
		} else {
			if (d_wp < DEF_STOP_DIST) {
				// arrived
			} else if (fabs(th_err) > DEF_REALIGN_TOL) {
				def_wp = 0;
			} else {
				int fwd = (int)(0.4 * d_wp);
				if (fwd < 25) fwd = 25;
				if (fwd > 50) fwd = 50;
				int turn = 0;
				if (fabs(th_err) > DEF_STEER_DEADBAND) {
					turn = (int)(25.0 * th_err);
					if (turn >  25) turn =  25;
					if (turn < -25) turn = -25;
				}
				def_th_left  = TH_STOP + fwd - turn;
				def_th_right = TH_STOP + fwd + turn;
			}
		}

		if (KEY(VK_SPACE)) {
			off_th_left = off_th_right = TH_STOP;
			def_th_left = def_th_right = TH_STOP;
		}
		if (KEY('X')) goto cleanup;

		if (off_th_left  < 0)   off_th_left  = 0;
		if (off_th_left  > 180) off_th_left  = 180;
		if (off_th_right < 0)   off_th_right = 0;
		if (off_th_right > 180) off_th_right = 180;
		if (def_th_left  < 0)   def_th_left  = 0;
		if (def_th_left  > 180) def_th_left  = 180;
		if (def_th_right < 0)   def_th_right = 0;
		if (def_th_right > 180) def_th_right = 180;

		view_rgb_image(rgb);

		buf_off[0] = start_char;
		buf_off[1] = (unsigned char)off_th_left;
		buf_off[2] = (unsigned char)off_th_right;
		buf_off[3] = (unsigned char)TH_STOP;
		// defense motors are wired left/right reversed - swap at the byte boundary
		buf_def[0] = start_char;
		buf_def[1] = (unsigned char)def_th_right;
		buf_def[2] = (unsigned char)def_th_left;
		buf_def[3] = (unsigned char)TH_STOP;
		n = 4;
		serial_send(buf_off, n, h_off);
		serial_send(buf_def, n, h_def);

		Sleep(30);
	}

cleanup:
	buf_off[0] = start_char; buf_off[1] = TH_STOP; buf_off[2] = TH_STOP; buf_off[3] = TH_STOP;
	buf_def[0] = start_char; buf_def[1] = TH_STOP; buf_def[2] = TH_STOP; buf_def[3] = TH_STOP;
	serial_send(buf_off, 4, h_off);
	serial_send(buf_def, 4, h_def);
	Sleep(50);
	close_serial(h_off);
	close_serial(h_def);

	free_image(rgb);
	free_image(grey);
	free_image(binary);
	free_image(label);
	free_image(arena_mask);
	deactivate_vision();
	return 0;
}

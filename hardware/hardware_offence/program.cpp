// hardware_offence - drive to opponent while avoiding black obstacles

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

static char COM_PORT[] = "COM9";

static const int    TH_STOP        = 90;
static const int    DTH_FWD        = 30;
static const int    DTH_TURN       = 30;

static const int    MIN_AREA       = 30;
static const double MIN_PAIR_SEP   = 30.0;
static const double MAX_PAIR_SEP   = 150.0;

static const double ALIGN_TOL      = 0.25;   // rad
static const double STOP_DIST      = 120.0;  // px
static const double AVOID_DIST     = 130.0;  // px
static const double K_REP          = 3.0;

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


int main()
{
	int width  = 640;
	int height = 480;
	int cam_number = 1;

	image rgb, grey, binary, label, arena_mask;
	int nlabels;

	HANDLE h1;
	const int NMAX = 64;
	char buffer_out[NMAX];
	int n;
	int th_left, th_right, th_aux;
	unsigned char start_char = 255;

	struct color_target {
		double hue_target, hue_tol, sat_min, sat_max, val_min;
		int    ic, jc, area;
	};
	color_target ct[3] = {
		{ 193.0, 15.0, 0.50, 1.01,  30.0,   0, 0, 0 },  // us-F  blue
		{  13.0,  8.0, 0.75, 1.01, 100.0,   0, 0, 0 },  // us-B  orange
		{ 138.0, 22.0, 0.30, 1.01,  10.0,   0, 0, 0 },  // tgt   green
	};

	double us_heading       = 0.0;
	bool   us_heading_valid = false;

	struct obstacle { double ic, jc; int area; };
	obstacle obs[N_OBS];
	int n_obs = 0;
	obstacle locked_obs[N_OBS];
	int      locked_n = 0;
	bool     obs_locked = false;

	if (open_serial(COM_PORT, h1, 0)) return 1;

	while (!KEY(VK_SPACE)) Sleep(1);
	buffer_out[0] = 's';
	if (serial_send(buffer_out, 1, h1)) { close_serial(h1); return 1; }
	Sleep(100);

	activate_vision();
	activate_camera(cam_number, height, width);

	rgb.type        = RGB_IMAGE;   rgb.width        = width; rgb.height        = height; allocate_image(rgb);
	grey.type       = GREY_IMAGE;  grey.width       = width; grey.height       = height; allocate_image(grey);
	binary.type     = GREY_IMAGE;  binary.width     = width; binary.height     = height; allocate_image(binary);
	label.type      = LABEL_IMAGE; label.width      = width; label.height      = height; allocate_image(label);
	arena_mask.type = GREY_IMAGE;  arena_mask.width = width; arena_mask.height = height; allocate_image(arena_mask);

	th_aux = TH_STOP;
	bool arena_active = false;

	while (1) {
		acquire_image(rgb, cam_number);

		// arena mask: largest white connected component, fall back to all-in
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

		// us-F (blue) - largest in arena
		{
			color_target &t = ct[0];
			hsv_threshold(rgb, binary, t.hue_target, t.hue_tol, t.sat_min, t.sat_max, t.val_min);
			mask_in(binary, arena_mask);
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			label_image(binary, label, nlabels);
			largest_label(binary, label, nlabels, t.ic, t.jc, t.area);
		}
		bool us_F_seen = (ct[0].area > MIN_AREA);

		// us-B (orange) - anchor on us-F
		{
			color_target &t = ct[1];
			hsv_threshold(rgb, binary, t.hue_target, t.hue_tol, t.sat_min, t.sat_max, t.val_min);
			mask_in(binary, arena_mask);
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			label_image(binary, label, nlabels);
			if (us_F_seen) {
				nearest_label(binary, label, nlabels,
				              ct[0].ic, ct[0].jc, MAX_PAIR_SEP, MIN_AREA,
				              t.ic, t.jc, t.area);
			} else {
				largest_label(binary, label, nlabels, t.ic, t.jc, t.area);
			}
		}
		bool us_B_seen = (ct[1].area > MIN_AREA);

		// target (green) - largest
		{
			color_target &t = ct[2];
			hsv_threshold(rgb, binary, t.hue_target, t.hue_tol, t.sat_min, t.sat_max, t.val_min);
			mask_in(binary, arena_mask);
			erode(binary, grey);    erode(grey, binary);
			dialate(binary, grey);  dialate(grey, binary);
			label_image(binary, label, nlabels);
			largest_label(binary, label, nlabels, t.ic, t.jc, t.area);
		}
		bool tgt_seen = (ct[2].area > MIN_AREA);

		// obstacles - shape-based, detect once then lock
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
				if (us_F_seen) {
					double dx = ic - ct[0].ic, dy = jc - ct[0].jc;
					if (sqrt(dx*dx + dy*dy) < OWN_BODY_RADIUS) own = true;
				}
				if (!own && us_B_seen) {
					double dx = ic - ct[1].ic, dy = jc - ct[1].jc;
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

		// heading + position from F-B pair
		double F_x, F_y, B_x, B_y, T_x, T_y;
		barrel_correct(ct[0].ic, ct[0].jc, F_x, F_y);
		barrel_correct(ct[1].ic, ct[1].jc, B_x, B_y);
		barrel_correct(ct[2].ic, ct[2].jc, T_x, T_y);

		us_heading_valid = false;
		double us_x = 0, us_y = 0; bool us_pos = false;
		if (us_F_seen && us_B_seen) {
			double dx = F_x - B_x;
			double dy = F_y - B_y;
			double sep = sqrt(dx*dx + dy*dy);
			if (sep >= MIN_PAIR_SEP && sep <= MAX_PAIR_SEP) {
				us_heading = atan2(dy, dx);
				us_heading_valid = true;
				us_x = 0.5*(F_x + B_x);
				us_y = 0.5*(F_y + B_y);
				us_pos = true;
			}
		}
		if (!us_pos && us_F_seen) { us_x = F_x; us_y = F_y; us_pos = true; }
		else if (!us_pos && us_B_seen) { us_x = B_x; us_y = B_y; us_pos = true; }

		// potential field steering
		double desired = 0.0, head_err = 0.0, dist_to_tgt = 0.0;
		if (us_pos && tgt_seen) {
			double dx = T_x - us_x;
			double dy = T_y - us_y;
			dist_to_tgt = sqrt(dx*dx + dy*dy);
			double ax = dx / dist_to_tgt;
			double ay = dy / dist_to_tgt;
			double rx = 0.0, ry = 0.0;
			for (int k = 0; k < n_obs; k++) {
				double ox = us_x - obs[k].ic;
				double oy = us_y - obs[k].jc;
				double od = sqrt(ox*ox + oy*oy);
				if (od < AVOID_DIST && od > 1.0) {
					double w = (AVOID_DIST - od) / AVOID_DIST;
					rx += K_REP * w * (ox / od);
					ry += K_REP * w * (oy / od);
				}
			}
			desired = atan2(ay + ry, ax + rx);
			if (us_heading_valid) head_err = angle_diff(desired, us_heading);
		}

		// controller
		th_left = TH_STOP; th_right = TH_STOP;
		if (!us_heading_valid || !tgt_seen) {
			// LOST - hold stop
		} else if (dist_to_tgt < STOP_DIST) {
			// AT_TGT - hold stop
		} else if (fabs(head_err) > ALIGN_TOL) {
			if (head_err > 0) {
				th_left  = TH_STOP - DTH_TURN;
				th_right = TH_STOP + DTH_TURN;
			} else {
				th_left  = TH_STOP + DTH_TURN;
				th_right = TH_STOP - DTH_TURN;
			}
		} else {
			th_left  = TH_STOP + DTH_FWD;
			th_right = TH_STOP + DTH_FWD;
		}

		if (KEY(VK_SPACE)) { th_left = th_right = TH_STOP; }
		if (KEY('X')) goto cleanup;

		if (th_left  < 0)   th_left  = 0;
		if (th_left  > 180) th_left  = 180;
		if (th_right < 0)   th_right = 0;
		if (th_right > 180) th_right = 180;

		view_rgb_image(rgb);

		buffer_out[0] = start_char;
		buffer_out[1] = (unsigned char)th_left;
		buffer_out[2] = (unsigned char)th_right;
		buffer_out[3] = (unsigned char)th_aux;
		n = 4;
		if (serial_send(buffer_out, n, h1)) {
			while (1) {
				if (KEY('C')) break;
				if (KEY('X')) goto cleanup;
				Sleep(10);
			}
		}

		Sleep(30);
	}

cleanup:
	buffer_out[0] = start_char;
	buffer_out[1] = (unsigned char)TH_STOP;
	buffer_out[2] = (unsigned char)TH_STOP;
	buffer_out[3] = (unsigned char)TH_STOP;
	serial_send(buffer_out, 4, h1);
	Sleep(50);
	close_serial(h1);

	free_image(rgb);
	free_image(grey);
	free_image(binary);
	free_image(label);
	free_image(arena_mask);
	deactivate_vision();
	return 0;
}

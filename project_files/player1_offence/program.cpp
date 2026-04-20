
// MECH 472/663 - Player 1 OFFENCE
// Chase the opponent robot and hit it with the laser.
// mode = 1 (two player, player #1 controls robot_A)
//
// Vision pipeline based on prof's examples (vision_example_6, simulator_example2):
//   copy -> lowpass_filter -> scale -> threshold -> invert -> erode -> dialate
//   -> label_image -> centroid2
//
// State machine: SEARCH -> CHASE -> FIRE
//
// Rules:
// - no laser sweeping (can't change laser angle while firing)
// - one shot per turn
// - can't leave screen or hit obstacles

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

const int IMG_W = 640;
const int IMG_H = 480;

// vision pipeline images
image img_rgb;
image img_rgb0;
image img_grey;
image img_bin;
image img_tmp;
image img_lbl;


// -----------------------------------------------------------------------
// centroid2 -- centroid + average RGB of a labelled blob
// (from prof's vision_example_6 / update centroid)
// -----------------------------------------------------------------------
int centroid2(image &rgb, image &label, int i_label,
              double &ic, double &jc,
              int &R_ave, int &G_ave, int &B_ave, double &n_pixels)
{
    ibyte  *p, *pc;
    i2byte *pl;
    int i, j, k, width, height;
    double mi, mj, m, n, R_sum, G_sum, B_sum;
    const double EPS = 1e-7;

    p      = rgb.pdata;
    pl     = (i2byte *)label.pdata;
    width  = rgb.width;
    height = rgb.height;
    mi = mj = m = n = R_sum = G_sum = B_sum = 0.0;

    for (j = 0; j < height; j++) {
        for (i = 0; i < width; i++) {
            if (pl[j * width + i] == i_label) {
                k  = i + width * j;
                pc = p + 3 * k;
                int B = *pc, G = *(pc+1), R = *(pc+2);
                R_sum += R; G_sum += G; B_sum += B;
                n++; m++;
                mi += i; mj += j;
            }
        }
    }
    ic    = mi / (m + EPS);
    jc    = mj / (m + EPS);
    R_ave = (int)(R_sum / (n + EPS));
    G_ave = (int)(G_sum / (n + EPS));
    B_ave = (int)(B_sum / (n + EPS));
    n_pixels = n;
    return 0;
}


// -----------------------------------------------------------------------
// is_obstacle_colour -- reject blobs that match known obstacle colours
// -----------------------------------------------------------------------
int is_obstacle_colour(int R, int G, int B)
{
    if (R < 35 && G < 35 && B < 35) return 1;             // black
    if (G > 80 && G > R + 30 && G > B + 25) return 1;     // green
    if (B > 80 && B > R + 30 && B > G + 25) return 1;     // blue
    if (R > 140 && G > 55 && G < 170 && B < 70) return 1; // orange obs
    if (R > 140 && G < 90 && B < 90) return 1;             // red obs
    return 0;
}


// -----------------------------------------------------------------------
// angle_diff -- signed angular difference wrapped to [-pi, pi]
// (from prof's control approach in simulator_example2)
// -----------------------------------------------------------------------
double angle_diff(double a, double b)
{
    const double PI = 3.14159265;
    double d = b - a;
    while (d >  PI) d -= 2.0 * PI;
    while (d < -PI) d += 2.0 * PI;
    return d;
}


// -----------------------------------------------------------------------
// mask_obstacles -- paint over known obstacle footprints with gray
// so the blob detector ignores them
// -----------------------------------------------------------------------
void mask_obstacles(image &img)
{
    const int MASK_R = 62;
    ibyte *p = img.pdata;
    const int W = img.width, H = img.height;
    for (int k = 0; k < S1->N_obs; k++) {
        int cx = (int)S1->x_obs[k];
        int cy = (int)S1->y_obs[k];
        for (int dy = -MASK_R; dy <= MASK_R; dy++) {
            for (int dx = -MASK_R; dx <= MASK_R; dx++) {
                if (dx*dx + dy*dy > MASK_R*MASK_R) continue;
                int px = cx+dx, py = cy+dy;
                if (px < 0 || px >= W || py < 0 || py >= H) continue;
                ibyte *pix = p + 3*(py*W + px);
                pix[0] = 200; pix[1] = 200; pix[2] = 200;
            }
        }
    }
}


// -----------------------------------------------------------------------
// find_opponent -- full vision pipeline to detect opponent
//
// Each robot has two markers (blobs). This function:
//   1. Runs the prof's vision pipeline to find blobs
//   2. Filters out obstacles by position and colour
//   3. Pairs nearby blobs as the two markers of one robot
//   4. Identifies self (closest to known position) vs opponent (farthest)
//   5. Determines opponent heading from the pair geometry
//
// Returns 1 if opponent found, 0 otherwise
// -----------------------------------------------------------------------
const double PAIR_MIN   = 45.0;
const double PAIR_MAX   = 140.0;
const double OBS_EXCL_R = 30.0;

int find_opponent(double &target_ic, double &target_jc,
                  double &nav_ic,    double &nav_jc,
                  double &opp_heading, int &heading_valid,
                  double my_x, double my_y)
{
    heading_valid = 0;

    // prof's vision pipeline: greyscale -> filter -> threshold -> morphology -> label
    copy(img_rgb, img_grey);
    lowpass_filter(img_grey, img_tmp);
    copy(img_tmp, img_grey);
    scale(img_grey, img_grey);
    threshold(img_grey, img_bin, 70);
    invert(img_bin, img_bin);
    erode(img_bin, img_tmp);   copy(img_tmp, img_bin);
    dialate(img_bin, img_tmp); copy(img_tmp, img_bin);

    int nlabels;
    label_image(img_bin, img_lbl, nlabels);
    if (nlabels == 0) return 0;

    // collect robot-candidate blobs (reject obstacles)
    const int MAX_BLOBS = 40;
    double b_ic[MAX_BLOBS], b_jc[MAX_BLOBS], b_area[MAX_BLOBS];
    int    b_R[MAX_BLOBS], b_G[MAX_BLOBS], b_B[MAX_BLOBS];
    int    n_blobs = 0;

    for (int lbl = 1; lbl <= nlabels && n_blobs < MAX_BLOBS; lbl++) {
        double ic, jc, area;
        int R, G, B;
        centroid2(img_rgb0, img_lbl, lbl, ic, jc, R, G, B, area);
        if (area < 250) continue;

        // layer 1: position-based exclusion near known obstacles
        bool pos_excl = false;
        for (int k = 0; k < S1->N_obs; k++) {
            double dx = ic - S1->x_obs[k];
            double dy = jc - S1->y_obs[k];
            if (sqrt(dx*dx + dy*dy) < OBS_EXCL_R) { pos_excl = true; break; }
        }
        if (pos_excl) {
            draw_point_rgb(img_rgb, (int)ic, (int)jc, 255, 128, 0);
            continue;
        }

        // layer 2: colour-based exclusion
        if (is_obstacle_colour(R, G, B)) {
            draw_point_rgb(img_rgb, (int)ic, (int)jc, 255, 255, 0);
            continue;
        }

        b_ic[n_blobs] = ic; b_jc[n_blobs] = jc; b_area[n_blobs] = area;
        b_R[n_blobs] = R; b_G[n_blobs] = G; b_B[n_blobs] = B;
        n_blobs++;
    }
    if (n_blobs == 0) return 0;

    // pair blobs within PAIR_MIN..PAIR_MAX as the two markers of one robot
    const int MAX_PAIRS = 10;
    int pair_a[MAX_PAIRS], pair_b[MAX_PAIRS];
    int paired[MAX_BLOBS];
    int n_pairs = 0;
    for (int i = 0; i < n_blobs; i++) paired[i] = 0;

    for (int i = 0; i < n_blobs && n_pairs < MAX_PAIRS; i++) {
        if (paired[i]) continue;
        double best_d = 1e9; int best_j = -1;
        for (int j = i+1; j < n_blobs; j++) {
            if (paired[j]) continue;
            double dx = b_ic[i]-b_ic[j], dy = b_jc[i]-b_jc[j];
            double d = sqrt(dx*dx + dy*dy);
            if (d >= PAIR_MIN && d <= PAIR_MAX && d < best_d) { best_d = d; best_j = j; }
        }
        if (best_j >= 0) {
            pair_a[n_pairs] = i; pair_b[n_pairs] = best_j;
            paired[i] = paired[best_j] = 1;
            n_pairs++;
        }
    }

    // build candidate list: pairs first, then unpaired singletons
    const int MAX_CANDS = 20;
    double cand_mic[MAX_CANDS], cand_mjc[MAX_CANDS];
    double cand_aic[MAX_CANDS], cand_ajc[MAX_CANDS];
    double cand_bic[MAX_CANDS], cand_bjc[MAX_CANDS];
    double cand_d[MAX_CANDS];
    int    cand_pair[MAX_CANDS];
    int n_cands = 0;

    for (int k = 0; k < n_pairs && n_cands < MAX_CANDS; k++) {
        int a = pair_a[k], b = pair_b[k];
        double mic = (b_ic[a]+b_ic[b])/2.0, mjc = (b_jc[a]+b_jc[b])/2.0;
        double dx = mic-my_x, dy = mjc-my_y;
        cand_mic[n_cands] = mic;   cand_mjc[n_cands] = mjc;
        cand_aic[n_cands] = b_ic[a]; cand_ajc[n_cands] = b_jc[a];
        cand_bic[n_cands] = b_ic[b]; cand_bjc[n_cands] = b_jc[b];
        cand_d[n_cands] = sqrt(dx*dx+dy*dy);
        cand_pair[n_cands] = 1;
        n_cands++;
    }
    for (int i = 0; i < n_blobs && n_cands < MAX_CANDS; i++) {
        if (paired[i]) continue;
        double dx = b_ic[i]-my_x, dy = b_jc[i]-my_y;
        cand_mic[n_cands] = b_ic[i]; cand_mjc[n_cands] = b_jc[i];
        cand_aic[n_cands] = b_ic[i]; cand_ajc[n_cands] = b_jc[i];
        cand_bic[n_cands] = b_ic[i]; cand_bjc[n_cands] = b_jc[i];
        cand_d[n_cands] = sqrt(dx*dx+dy*dy);
        cand_pair[n_cands] = 0;
        n_cands++;
    }
    if (n_cands == 0) return 0;

    // my robot = closest candidate to my known position
    int self_idx = 0;
    for (int k = 1; k < n_cands; k++)
        if (cand_d[k] < cand_d[self_idx]) self_idx = k;

    draw_point_rgb(img_rgb, (int)cand_mic[self_idx], (int)cand_mjc[self_idx], 0, 255, 0);

    if (n_cands == 1) {
        if (cand_d[0] < 60.0) return 0;  // only found myself
        nav_ic = target_ic = cand_mic[0];
        nav_jc = target_jc = cand_mjc[0];
        draw_point_rgb(img_rgb, (int)nav_ic, (int)nav_jc, 255, 0, 0);
        return 1;
    }

    // opponent = farthest candidate from my position
    double best_d2 = -1; int opp_idx = -1;
    for (int k = 0; k < n_cands; k++) {
        if (k == self_idx) continue;
        if (cand_d[k] > best_d2) { best_d2 = cand_d[k]; opp_idx = k; }
    }
    if (opp_idx < 0) return 0;

    nav_ic = cand_mic[opp_idx];
    nav_jc = cand_mjc[opp_idx];

    if (cand_pair[opp_idx]) {
        double a_ic = cand_aic[opp_idx], a_jc = cand_ajc[opp_idx];
        double bic  = cand_bic[opp_idx], bjc  = cand_bjc[opp_idx];

        // front marker = blob closer to me
        double da = sqrt((a_ic-my_x)*(a_ic-my_x)+(a_jc-my_y)*(a_jc-my_y));
        double db = sqrt((bic-my_x)*(bic-my_x)+(bjc-my_y)*(bjc-my_y));
        double front_ic, front_jc, rear_ic, rear_jc;
        if (da <= db) { front_ic=a_ic; front_jc=a_jc; rear_ic=bic; rear_jc=bjc; }
        else          { front_ic=bic;  front_jc=bjc;  rear_ic=a_ic; rear_jc=a_jc; }

        target_ic     = front_ic;  target_jc     = front_jc;
        opp_heading   = atan2(front_jc-rear_jc, front_ic-rear_ic);
        heading_valid = 1;

        draw_point_rgb(img_rgb,(int)front_ic,(int)front_jc, 255,  0,  0);
        draw_point_rgb(img_rgb,(int)rear_ic, (int)rear_jc,    0,255,255);
        draw_point_rgb(img_rgb,(int)nav_ic,  (int)nav_jc,   255,255,255);
    } else {
        target_ic = nav_ic; target_jc = nav_jc;
        draw_point_rgb(img_rgb,(int)nav_ic,(int)nav_jc, 255,0,0);
    }
    return 1;
}


// -----------------------------------------------------------------------
// draw_line_rgb -- Bresenham line for visual overlays
// -----------------------------------------------------------------------
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


// -----------------------------------------------------------------------
// drive helpers (based on prof's servo model: left servo flipped)
//   forward:  pw_l = 1500-V,  pw_r = 1500+V
//   CCW spin: pw_l = pw_r = 1500+X
//   backward: pw_l = 1500+V,  pw_r = 1500-V
// -----------------------------------------------------------------------
int clamp_pw(int pw)
{
    if (pw < 1000) return 1000;
    if (pw > 2000) return 2000;
    return pw;
}

void drive_forward(int &pw_l, int &pw_r, double heading_err,
                   int v_fwd, double steer_gain)
{
    int steer = (int)(steer_gain * heading_err);
    pw_l = clamp_pw(1500 - v_fwd + steer);
    pw_r = clamp_pw(1500 + v_fwd + steer);
}

void rotate_inplace(int &pw_l, int &pw_r, double heading_err, int rot_pw = 175)
{
    int sign = (heading_err >= 0) ? 1 : -1;
    pw_l = clamp_pw(1500 + sign * rot_pw);
    pw_r = clamp_pw(1500 + sign * rot_pw);
}

void drive_backward(int &pw_l, int &pw_r, int v_bwd = 200)
{
    pw_l = clamp_pw(1500 + v_bwd);
    pw_r = clamp_pw(1500 - v_bwd);
}


// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main()
{
    const double PI = 3.14159265;

    // --- simulation setup (must match player2) ---
    double width1 = IMG_W, height1 = IMG_H;
    const int N_obs = 2;
    double x_obs[N_obs] = { 120, 320 }; //270.5 135 
    double y_obs[N_obs] = { 120, 320 };
    char obstacle_file[N_obs][S_MAX] = {
        "obstacle_black.bmp", "obstacle_green.bmp"
    };
    double D = 121.0, Lx = 31.0, Ly = 0.0, Ax = 37.0, Ay = 0.0;
    double alpha_max = PI / 2.0;
    int n_robot = 2;

    int pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
    double max_speed = 120.0;

    // --- vision state ---
    double opp_nav_ic = IMG_W/2.0, opp_nav_jc = IMG_H/2.0;
    double opp_tgt_ic = IMG_W/2.0, opp_tgt_jc = IMG_H/2.0;
    double opp_heading = 0.0;
    int found = 0, heading_valid = 0, lost_frames = 0;

    // --- laser state (one shot per turn) ---
    int fired = 0;

    // --- state machine: 0=SEARCH  1=CHASE  2=FIRE ---
    int state = 0;
    static const char *snames[] = {"SEARCH", "CHASE", "FIRE"};

    // --- tuning constants ---
    const int    V_FWD       = 200;     // forward pw offset
    const double STEER_GAIN  = 120.0;   // proportional steering
    const int    ROT_PW      = 175;     // rotation pw offset
    const double STOP_DIST   = 160.0;   // stop approaching when this close
    const double BACKUP_DIST = 120.0;   // back away when this close
    const double FIRE_DIST   = 350.0;   // max firing range
    const double ALIGN_TOL   = 0.10;    // rad: heading error to fire
    const double OBS_AVOID   = 150.0;   // obstacle avoidance trigger distance
    const int    WALL_MARGIN = 20;      // wall avoidance margin (pixels)
    const int    LOST_THRESH = 25;      // frames before switching to SEARCH

    cout << "\n=== PLAYER 1 - OFFENCE ===";
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

    // allocate vision pipeline images
    img_rgb.type  = RGB_IMAGE;   img_rgb.width  = IMG_W; img_rgb.height  = IMG_H;
    img_rgb0.type = RGB_IMAGE;   img_rgb0.width = IMG_W; img_rgb0.height = IMG_H;
    img_grey.type = GREY_IMAGE;  img_grey.width = IMG_W; img_grey.height = IMG_H;
    img_bin.type  = GREY_IMAGE;  img_bin.width  = IMG_W; img_bin.height  = IMG_H;
    img_tmp.type  = GREY_IMAGE;  img_tmp.width  = IMG_W; img_tmp.height  = IMG_H;
    img_lbl.type  = LABEL_IMAGE; img_lbl.width  = IMG_W; img_lbl.height  = IMG_H;
    allocate_image(img_rgb);  allocate_image(img_rgb0);
    allocate_image(img_grey); allocate_image(img_bin);
    allocate_image(img_tmp);  allocate_image(img_lbl);

    wait_for_player();

    double tc0 = high_resolution_time(), tc;

    while (1) {

        acquire_image_sim(img_rgb);
        mask_obstacles(img_rgb);
        copy(img_rgb, img_rgb0);
        tc = high_resolution_time() - tc0;

        // --- my robot state from simulator (prof's approach) ---
        double my_x  = S1->P[1]->x[2];
        double my_y  = S1->P[1]->x[3];
        double theta = S1->P[1]->x[1];
        if (my_x < 0) my_x = 0; if (my_x > IMG_W) my_x = IMG_W;
        if (my_y < 0) my_y = 0; if (my_y > IMG_H) my_y = IMG_H;

        // --- vision ---
        double tgt_ic, tgt_jc, nav_ic, nav_jc, h;
        int hv;
        found = find_opponent(tgt_ic, tgt_jc, nav_ic, nav_jc, h, hv, my_x, my_y);
        if (found) {
            opp_tgt_ic = tgt_ic; opp_tgt_jc = tgt_jc;
            opp_nav_ic = nav_ic; opp_nav_jc = nav_jc;
            if (hv) { opp_heading = h; heading_valid = 1; }
            lost_frames = 0;
        } else {
            lost_frames++;
            heading_valid = 0;
        }

        // --- geometry ---
        double nav_dx  = opp_nav_ic - my_x;
        double nav_dy  = opp_nav_jc - my_y;
        double dist    = sqrt(nav_dx*nav_dx + nav_dy*nav_dy);
        double nav_err = angle_diff(theta, atan2(nav_dy, nav_dx));

        double aim_dx  = opp_tgt_ic - my_x;
        double aim_dy  = opp_tgt_jc - my_y;
        double aim_err = angle_diff(theta, atan2(aim_dy, aim_dx));

        // --- obstacle check ---
        bool near_obs = false;
        double obs_min_d = 1e9;
        int obs_closest_k = -1;
        for (int k = 0; k < S1->N_obs; k++) {
            double odx = my_x - S1->x_obs[k];
            double ody = my_y - S1->y_obs[k];
            double od  = sqrt(odx*odx + ody*ody);
            if (od < obs_min_d) { obs_min_d = od; obs_closest_k = k; }
            if (od < OBS_AVOID) near_obs = true;
        }

        // --- wall avoidance with hysteresis ---
        static int wall_frames = 0;
        {
            bool hit_wall = (my_x < WALL_MARGIN || my_x > IMG_W - WALL_MARGIN ||
                             my_y < WALL_MARGIN || my_y > IMG_H - WALL_MARGIN);
            if (hit_wall) wall_frames = 25;
            else if (wall_frames > 0) wall_frames--;
        }
        bool near_wall = (wall_frames > 0);

        // --- line-of-sight check: is path to opponent clear of obstacles? ---
        bool clear_shot = true;
        if (found) {
            for (int k = 0; k < S1->N_obs; k++) {
                double ox = S1->x_obs[k], oy = S1->y_obs[k];
                double dx = opp_tgt_ic - my_x, dy = opp_tgt_jc - my_y;
                double seg_len = sqrt(dx*dx + dy*dy);
                if (seg_len < 1.0) continue;
                double ux = dx/seg_len, uy = dy/seg_len;
                double t = (ox - my_x)*ux + (oy - my_y)*uy;
                if (t < 0 || t > seg_len) continue;
                double px = my_x + t*ux, py = my_y + t*uy;
                double perp = sqrt((ox-px)*(ox-px) + (oy-py)*(oy-py));
                if (perp < 70.0) { clear_shot = false; break; }
            }
        }

        // --- state transitions ---
        if (lost_frames > LOST_THRESH) {
            state = 0;
        } else if (found && state == 0) {
            state = 1;  // SEARCH -> CHASE
        } else if (found && state == 1 && fabs(aim_err) < ALIGN_TOL &&
                   ((clear_shot && dist < FIRE_DIST) || dist < STOP_DIST)) {
            state = 2;  // CHASE -> FIRE
        } else if (state == 2 && fabs(aim_err) > ALIGN_TOL * 2.5) {
            state = 1;  // lost aim -> back to CHASE
        }

        // --- drive decisions ---
        // priority: wall > obstacle > state machine
        if (near_wall) {
            double err_c = angle_diff(theta, atan2(IMG_H/2.0 - my_y, IMG_W/2.0 - my_x));
            if (fabs(err_c) > 0.3) rotate_inplace(pw_l, pw_r, err_c, ROT_PW);
            else                   drive_forward(pw_l, pw_r, err_c, V_FWD, STEER_GAIN);

        } else if (near_obs) {
            // drive around obstacle, not away from it
            static int avoid_side = 0;
            static int avoid_frames = 0;
            if (avoid_frames <= 0) {
                // pick side: cross product tells us which side of obstacle the opponent is on
                double obs_dx = S1->x_obs[obs_closest_k] - my_x;
                double obs_dy = S1->y_obs[obs_closest_k] - my_y;
                double opp_dx = opp_nav_ic - my_x;
                double opp_dy = opp_nav_jc - my_y;
                double cross = obs_dx * opp_dy - obs_dy * opp_dx;
                avoid_side = (cross > 0) ? 1 : -1;
                avoid_frames = 60;
            }
            avoid_frames--;
            double away_a = atan2(my_y - S1->y_obs[obs_closest_k],
                                  my_x - S1->x_obs[obs_closest_k]);
            double side_a = away_a + avoid_side * PI / 3.0;
            double err_side = angle_diff(theta, side_a);
            drive_forward(pw_l, pw_r, err_side, V_FWD, STEER_GAIN);

        } else {
            switch (state) {

            case 0: // SEARCH: slow CCW spin until opponent found
                pw_l = 1500 + ROT_PW;
                pw_r = 1500 + ROT_PW;
                break;

            case 1: // CHASE
                if (dist < BACKUP_DIST) {
                    drive_backward(pw_l, pw_r, V_FWD);
                } else if (clear_shot && dist < FIRE_DIST) {
                    rotate_inplace(pw_l, pw_r, aim_err, ROT_PW);
                } else {
                    drive_forward(pw_l, pw_r, nav_err, V_FWD, STEER_GAIN);
                }
                break;

            case 2: // FIRE: hold aim on front marker, gentle approach
                if (dist < BACKUP_DIST) {
                    if (fabs(aim_err) < PI/2.0) drive_backward(pw_l, pw_r, V_FWD/2);
                    else { pw_l = clamp_pw(1500 - V_FWD/2); pw_r = clamp_pw(1500 + V_FWD/2); }
                } else if (dist > STOP_DIST) {
                    if (fabs(aim_err) > 0.5)
                        rotate_inplace(pw_l, pw_r, aim_err, ROT_PW);
                    else
                        drive_forward(pw_l, pw_r, aim_err, V_FWD/2, STEER_GAIN);
                } else {
                    rotate_inplace(pw_l, pw_r, aim_err, ROT_PW);
                }
                break;
            }
        }

        // --- fire laser: ONE SHOT then stop ---
        if (!fired && tc > 3.0 && (state == 1 || state == 2) &&
            fabs(aim_err) < ALIGN_TOL && clear_shot && dist < FIRE_DIST)
        {
            fired = 1;
            laser = 1; pw_laser = 1500;
            pw_l = 1500; pw_r = 1500;

            // draw overlays on freeze frame
            {
                int hx = (int)(my_x + 50*cos(theta));
                int hy = (int)(my_y + 50*sin(theta));
                draw_line_rgb(img_rgb, (int)my_x, (int)my_y, hx, hy, 0, 255, 0);
            }
            if (found) {
                draw_line_rgb(img_rgb, (int)my_x, (int)my_y,
                              (int)opp_nav_ic, (int)opp_nav_jc, 255, 0, 255);
            }

            set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
            view_rgb_image(img_rgb, 1);

            cout << "\n\n========== LASER FIRED ==========";
            cout << "\n  Time:     " << (int)tc << "s";
            cout << "\n  Distance: " << (int)dist << " px";
            cout << "\n  Aim err:  " << (int)(aim_err*180.0/PI) << " deg";
            cout << "\n  Pair:     " << (heading_valid ? "YES" : "NO");
            cout << "\n=================================";
            cout << "\nPress space to exit.";
            pause();
            break;
        }

        // --- visual overlays (normal frames) ---
        {
            int hx = (int)(my_x + 50*cos(theta));
            int hy = (int)(my_y + 50*sin(theta));
            draw_line_rgb(img_rgb, (int)my_x, (int)my_y, hx, hy, 0, 255, 0);
        }
        if (found) {
            draw_line_rgb(img_rgb, (int)my_x, (int)my_y,
                          (int)opp_nav_ic, (int)opp_nav_jc, 255, 0, 255);
        }
        draw_line_rgb(img_rgb, WALL_MARGIN, WALL_MARGIN, IMG_W-WALL_MARGIN, WALL_MARGIN, 255, 255, 0);
        draw_line_rgb(img_rgb, IMG_W-WALL_MARGIN, WALL_MARGIN, IMG_W-WALL_MARGIN, IMG_H-WALL_MARGIN, 255, 255, 0);
        draw_line_rgb(img_rgb, IMG_W-WALL_MARGIN, IMG_H-WALL_MARGIN, WALL_MARGIN, IMG_H-WALL_MARGIN, 255, 255, 0);
        draw_line_rgb(img_rgb, WALL_MARGIN, IMG_H-WALL_MARGIN, WALL_MARGIN, WALL_MARGIN, 255, 255, 0);

        set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
        view_rgb_image(img_rgb, 1);

        // --- console output ---
        const char *drive_reason = "SM";
        if (near_wall) drive_reason = "WALL";
        else if (near_obs) drive_reason = "OBS";

        static int frame = 0;
        if (++frame % 60 == 0) {
            cout << "\n[t=" << (int)tc << "s] " << snames[state]
                 << " drv=" << drive_reason
                 << "  me=(" << (int)my_x << "," << (int)my_y
                 << " th=" << (int)(theta*180.0/PI) << ")"
                 << "  opp=(" << (int)opp_nav_ic << "," << (int)opp_nav_jc << ")"
                 << "  d=" << (int)dist
                 << " aim=" << (int)(aim_err*180.0/PI) << "deg"
                 << "  pw=" << pw_l << "/" << pw_r;
            if (obs_closest_k >= 0)
                cout << "  obs" << obs_closest_k << "_d=" << (int)obs_min_d;
        }

        if (KEY('X')) break;
    }

    free_image(img_rgb);  free_image(img_rgb0);
    free_image(img_grey); free_image(img_bin);
    free_image(img_tmp);  free_image(img_lbl);
    deactivate_vision();
    deactivate_simulation();
    cout << "\ndone.\n";
    return 0;
}

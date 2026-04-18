
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

// -----------------------------------------------------------------------
// PLAYER CONFIGURATION
//   0 = single-player test (stationary opponent, no sync needed)
//   1 = two-player, this is Player 1 (waits for Player 2, shows image)
//   2 = two-player, this is Player 2 (joins Player 1, no image display)
// -----------------------------------------------------------------------
const int PLAYER_NUM = 2;

const int IMG_W = 640;
const int IMG_H = 480;

image img_rgb;
image img_rgb0;
image img_grey;
image img_bin;
image img_tmp;
image img_lbl;

// -----------------------------------------------------------------------
// centroid2 -- centroid + average RGB of a labelled blob
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
// is_obstacle_colour
// Used as a SECONDARY filter after position-based exclusion (see below).
// Catches clearly coloured obstacles that may appear away from their
// simulator-registered centre (e.g. after dialate expands the blob).
// -----------------------------------------------------------------------
int is_obstacle_colour(int R, int G, int B)
{
    // Black / very dark (well below robot grey ~42)
    if (R < 35 && G < 35 && B < 35) return 1;

    // Green
    if (G > 80 && G > R + 30 && G > B + 25) return 1;

    // Blue
    if (B > 80 && B > R + 30 && B > G + 25) return 1;

    // Orange (high R, mid G, low B)
    if (R > 140 && G > 55 && G < 170 && B < 70) return 1;

    // Red
    if (R > 140 && G < 90 && B < 90) return 1;

    return 0;
}

// -----------------------------------------------------------------------
// angle_diff -- signed angular difference wrapped to [-pi, pi]
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
// mask_obstacles
// Paint over the known stationary obstacle footprints with bright gray
// BEFORE the vision pipeline runs.  Because obstacles never move, we know
// exactly where they are.  Masking at the pixel level means the blob
// detector never sees them at all, so the opponent is fully visible even
// when standing right next to an obstacle.
// Tune MASK_R if the obstacle bitmaps are larger or smaller than expected.
// -----------------------------------------------------------------------
void mask_obstacles(image &img)
{
    const int MASK_R = 62;                // px radius — obstacle BMPs are 80-84px wide (~42px radius + margin)
    ibyte *p  = img.pdata;
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
                pix[0] = 200;  // B  }  bright gray → greyscale ≈200 >> threshold-70
                pix[1] = 200;  // G  }  → treated as background by the blob detector
                pix[2] = 200;  // R  }
            }
        }
    }
}

// -----------------------------------------------------------------------
// find_opponent
//
// Each robot has two markers:
//   Front (target) -- laser must hit this (~80 px ahead of rear)
//   Rear  (angle)  -- used to compute robot heading
//
// Obstacle exclusion uses TWO layers:
//   1. Position-based: any blob whose centroid is within OBS_EXCL_R of a
//      known obstacle position (from S1->x_obs) is rejected.  This handles
//      black obstacles robustly even when they blend with a robot.
//   2. Colour-based: fallback for any remaining coloured obstacle blobs.
//
// Returns:
//   target_ic/jc   -- front marker (aim laser here)
//   nav_ic/jc      -- midpoint of both markers (drive toward here)
//   opp_heading    -- direction rear->front; only valid when heading_valid=1
//   heading_valid  -- 1 if a marker pair was found for the opponent
// -----------------------------------------------------------------------
const double PAIR_MIN    = 45.0;   // px: minimum inter-marker distance
const double PAIR_MAX    = 140.0;  // px: maximum inter-marker distance
const double OBS_EXCL_R  = 30.0;  // px: residual safety net (masking does the real work)

int find_opponent(double &target_ic, double &target_jc,
                  double &nav_ic,    double &nav_jc,
                  double &opp_heading, int &heading_valid,
                  double my_x, double my_y)
{
    heading_valid = 0;

    // --- Vision pipeline ---
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

    // --- Collect robot-candidate blobs ---
    const int MAX_BLOBS = 40;
    double b_ic[MAX_BLOBS], b_jc[MAX_BLOBS], b_area[MAX_BLOBS];
    int    b_R [MAX_BLOBS], b_G [MAX_BLOBS], b_B  [MAX_BLOBS];
    int    n_blobs = 0;
    int    n_obs_live = S1->N_obs;

    for (int lbl = 1; lbl <= nlabels && n_blobs < MAX_BLOBS; lbl++) {
        double ic, jc, area;
        int R, G, B;
        centroid2(img_rgb0, img_lbl, lbl, ic, jc, R, G, B, area);
        if (area < 250) continue;   // skip tiny noise

        // Layer 1: position-based obstacle exclusion (primary, most reliable)
        // Rejects any blob whose centroid is close to a known obstacle location.
        bool pos_excl = false;
        for (int k = 0; k < n_obs_live; k++) {
            double dx = ic - S1->x_obs[k];
            double dy = jc - S1->y_obs[k];
            if (sqrt(dx*dx + dy*dy) < OBS_EXCL_R) { pos_excl = true; break; }
        }
        if (pos_excl) {
            draw_point_rgb(img_rgb, (int)ic, (int)jc, 255, 128, 0); // orange = pos-excluded
            continue;
        }

        // Layer 2: colour-based obstacle exclusion (catches stray coloured blobs)
        if (is_obstacle_colour(R, G, B)) {
            draw_point_rgb(img_rgb, (int)ic, (int)jc, 255, 255, 0); // yellow = colour-excluded
            continue;
        }

        b_ic[n_blobs] = ic; b_jc[n_blobs] = jc; b_area[n_blobs] = area;
        b_R [n_blobs] = R;  b_G [n_blobs] = G;  b_B  [n_blobs]  = B;
        n_blobs++;
    }
    if (n_blobs == 0) return 0;

    // --- Pair blobs within PAIR_MIN..PAIR_MAX px as the two markers of one robot ---
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
            double d  = sqrt(dx*dx + dy*dy);
            if (d >= PAIR_MIN && d <= PAIR_MAX && d < best_d) { best_d = d; best_j = j; }
        }
        if (best_j >= 0) {
            pair_a[n_pairs] = i; pair_b[n_pairs] = best_j;
            paired[i] = paired[best_j] = 1;
            n_pairs++;
        }
    }

    // --- Build robot candidate list (pairs first, then unpaired singletons) ---
    const int MAX_CANDS = 20;
    double cand_mic[MAX_CANDS], cand_mjc[MAX_CANDS];
    double cand_aic[MAX_CANDS], cand_ajc[MAX_CANDS];
    double cand_bic[MAX_CANDS], cand_bjc[MAX_CANDS];
    double cand_d  [MAX_CANDS];
    int    cand_pair[MAX_CANDS];
    int n_cands = 0;

    for (int k = 0; k < n_pairs && n_cands < MAX_CANDS; k++) {
        int a = pair_a[k], b = pair_b[k];
        double mic = (b_ic[a]+b_ic[b])/2.0, mjc = (b_jc[a]+b_jc[b])/2.0;
        double dx = mic-my_x, dy = mjc-my_y;
        cand_mic[n_cands] = mic;   cand_mjc[n_cands] = mjc;
        cand_aic[n_cands] = b_ic[a]; cand_ajc[n_cands] = b_jc[a];
        cand_bic[n_cands] = b_ic[b]; cand_bjc[n_cands] = b_jc[b];
        cand_d  [n_cands] = sqrt(dx*dx+dy*dy);
        cand_pair[n_cands] = 1;
        n_cands++;
    }
    for (int i = 0; i < n_blobs && n_cands < MAX_CANDS; i++) {
        if (paired[i]) continue;
        double dx = b_ic[i]-my_x, dy = b_jc[i]-my_y;
        cand_mic[n_cands] = b_ic[i]; cand_mjc[n_cands] = b_jc[i];
        cand_aic[n_cands] = b_ic[i]; cand_ajc[n_cands] = b_jc[i];
        cand_bic[n_cands] = b_ic[i]; cand_bjc[n_cands] = b_jc[i];
        cand_d  [n_cands] = sqrt(dx*dx+dy*dy);
        cand_pair[n_cands] = 0;
        n_cands++;
    }
    if (n_cands == 0) return 0;

    // My robot = candidate closest to my known simulator position
    int self_idx = 0;
    for (int k = 1; k < n_cands; k++)
        if (cand_d[k] < cand_d[self_idx]) self_idx = k;

    draw_point_rgb(img_rgb, (int)cand_mic[self_idx], (int)cand_mjc[self_idx], 0, 255, 0);
    if (cand_pair[self_idx]) {
        draw_point_rgb(img_rgb, (int)cand_aic[self_idx], (int)cand_ajc[self_idx], 0, 180, 0);
        draw_point_rgb(img_rgb, (int)cand_bic[self_idx], (int)cand_bjc[self_idx], 0, 180, 0);
    }

    // Only one candidate -- is it me or the opponent?
    if (n_cands == 1) {
        if (cand_d[0] < 60.0) return 0;   // it's me, no opponent visible
        nav_ic = target_ic = cand_mic[0];
        nav_jc = target_jc = cand_mjc[0];
        draw_point_rgb(img_rgb, (int)nav_ic, (int)nav_jc, 255, 0, 0);
        return 1;
    }

    // Opponent = farthest candidate from my position
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

        // Front marker = blob closer to me (robot faces me when engaging)
        double da = sqrt((a_ic-my_x)*(a_ic-my_x)+(a_jc-my_y)*(a_jc-my_y));
        double db = sqrt((bic -my_x)*(bic -my_x)+(bjc -my_y)*(bjc -my_y));
        double front_ic, front_jc, rear_ic, rear_jc;
        if (da <= db) { front_ic=a_ic; front_jc=a_jc; rear_ic=bic; rear_jc=bjc; }
        else          { front_ic=bic;  front_jc=bjc;  rear_ic=a_ic; rear_jc=a_jc; }

        target_ic     = front_ic;  target_jc     = front_jc;
        opp_heading   = atan2(front_jc-rear_jc, front_ic-rear_ic);
        heading_valid = 1;

        draw_point_rgb(img_rgb,(int)front_ic,(int)front_jc, 255,  0,  0); // red   = front/target
        draw_point_rgb(img_rgb,(int)rear_ic, (int)rear_jc,    0,255,255); // cyan  = rear
        draw_point_rgb(img_rgb,(int)nav_ic,  (int)nav_jc,   255,255,255); // white = midpoint

        static int dbg=0;
        if (dbg++ % 120 == 0)
            cout << "\n  [pair] front=(" << (int)front_ic << "," << (int)front_jc
                 << ") rear=(" << (int)rear_ic << "," << (int)rear_jc
                 << ") sep=" << (int)sqrt((front_ic-rear_ic)*(front_ic-rear_ic)+(front_jc-rear_jc)*(front_jc-rear_jc))
                 << " hdg=" << opp_heading;
    } else {
        target_ic = nav_ic; target_jc = nav_jc;
        draw_point_rgb(img_rgb,(int)nav_ic,(int)nav_jc, 255,0,0);
    }
    return 1;
}

// -----------------------------------------------------------------------
// Servo helpers  (left servo physically flipped)
//
//   vl = -(pw_l-1500)/500 * v_max      vr = +(pw_r-1500)/500 * v_max
//
//   Forward:       pw_l = 1500-X,  pw_r = 1500+X          (X>0)
//   CCW rotation:  pw_l = pw_r = 1500+X
//   CW  rotation:  pw_l = pw_r = 1500-X
//   Backward:      pw_l = 1500+X,  pw_r = 1500-X
//   Fwd + steer:   pw_l = 1500-V+steer,  pw_r = 1500+V+steer
//                  (+steer = CCW correction,  -steer = CW correction)
// -----------------------------------------------------------------------
inline int clamp_pw(int pw) {
    if (pw < 1000) return 1000;
    if (pw > 2000) return 2000;
    return pw;
}
void drive_forward(int &pw_l, int &pw_r, double heading_err,
                   int v_fwd, double gain)
{
    int steer = (int)(heading_err * gain);
    // Clamp steer to half the forward speed — prevents tight circles.
    int max_steer = v_fwd / 2;
    if (steer >  max_steer) steer =  max_steer;
    if (steer < -max_steer) steer = -max_steer;
    pw_l = clamp_pw(1500 - v_fwd + steer);
    pw_r = clamp_pw(1500 + v_fwd + steer);
}
void rotate_inplace(int &pw_l, int &pw_r, double heading_err, int rot_pw = 180)
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

    // Simulation setup
    double width1 = IMG_W, height1 = IMG_H;
    const int N_obs = 2;
    double x_obs[N_obs] = { 270.5, 135.0 };
    double y_obs[N_obs] = { 270.5, 135.0 };
    char obstacle_file[N_obs][S_MAX] = {
        "obstacle_black.bmp", "obstacle_green.bmp"
    };
    double D=121.0, Lx=31.0, Ly=0.0, Ax=37.0, Ay=0.0, alpha_max=PI/2.0;
    int n_robot = 2;

    int    pw_l=1500, pw_r=1500, pw_laser=1500, laser=0;
    double max_speed = 120.0;

    // Vision state (persisted between frames)
    double opp_nav_ic=IMG_W/2.0, opp_nav_jc=IMG_H/2.0;
    double opp_tgt_ic=IMG_W/2.0, opp_tgt_jc=IMG_H/2.0;
    double opp_heading=0.0;
    int    found=0, heading_valid=0, lost_frames=0;

    // Defence robot does NOT fire (laser always 0)

    // State machine  0=SEARCH  1=FLEE  2=STRAFE  3=HIDE
    // SEARCH: spin to locate opponent
    // FLEE:   run directly away from opponent
    // STRAFE: move perpendicular to opponent line-of-sight (dodge laser)
    // HIDE:   use obstacles as cover — put obstacle between self and opponent
    int    state=0;
    double evade_start=-999.0;
    static const char *snames[] = {"SEARCH","FLEE","STRAFE","HIDE"};

    // ---- Tuning constants (DEFENCE) ----
    const int    V_FWD       = 220;    // forward pw offset — slightly faster to outrun attacker
    const double STEER_GAIN  = 120.0;  // proportional steer
    const int    ROT_PW      = 190;    // rotation pw offset — faster turning for evasion
    const double FLEE_DIST   = 300.0;  // px: start fleeing when opponent closer than this
    const double DANGER_DIST = 200.0;  // px: opponent very close — maximum urgency
    const double BACKUP_DIST = 120.0;  // px: actively back away when this close
    const double OBS_RADIUS  =  95.0;  // px: body-point avoidance (obs ~42px + 53px buffer; body extent already handled)
    const double OBS_EMERG   =  55.0;  // px: emergency zone (obs ~42px + 13px; body point nearly touching)
    const double PF_RADIUS   = 180.0;  // px: potential field soft-deflection (centre-based, not body points)
    const int    WALL_MARGIN = 100;    // px: wall avoidance activation margin
    const int    LOST_THRESH =  25;    // frames without detection before SEARCH
    const double AIM_TOL     = 0.30;   // rad: opponent heading within this → detected as threat
    const double AIM_DIST    = 350.0;  // px: detect threat from this distance
    const double BODY_R      =  60.0;  // px: half robot body length (D/2)

    cout << "\n=== ROBOT " << PLAYER_NUM << " ===";
    cout << "\nPress space to begin.";
    pause();

    activate_vision();
    activate_simulation(width1, height1,
        x_obs, y_obs, N_obs,
        "robot_A.bmp", "robot_B.bmp", "background.bmp",
        obstacle_file, D, Lx, Ly, Ax, Ay, alpha_max, n_robot);

    if (PLAYER_NUM == 0) {
        set_simulation_mode(0);
        set_robot_position(500, 240, PI);
        set_opponent_position(140, 240, 0.0);
        join_player();
    } else if (PLAYER_NUM == 1) {
        set_simulation_mode(1);
        set_robot_position(500, 240, PI);
        wait_for_player();
    } else {
        set_simulation_mode(2);
        set_robot_position(140, 240, 0.0);
        join_player();
    }

    set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

    img_rgb.type=RGB_IMAGE;   img_rgb.width=IMG_W;  img_rgb.height=IMG_H;
    img_rgb0.type=RGB_IMAGE;  img_rgb0.width=IMG_W; img_rgb0.height=IMG_H;
    img_grey.type=GREY_IMAGE; img_grey.width=IMG_W; img_grey.height=IMG_H;
    img_bin.type=GREY_IMAGE;  img_bin.width=IMG_W;  img_bin.height=IMG_H;
    img_tmp.type=GREY_IMAGE;  img_tmp.width=IMG_W;  img_tmp.height=IMG_H;
    img_lbl.type=LABEL_IMAGE; img_lbl.width=IMG_W;  img_lbl.height=IMG_H;
    allocate_image(img_rgb);  allocate_image(img_rgb0);
    allocate_image(img_grey); allocate_image(img_bin);
    allocate_image(img_tmp);  allocate_image(img_lbl);

    double tc0=high_resolution_time(), tc;

    while (1) {

        acquire_image_sim(img_rgb);
        mask_obstacles(img_rgb);    // erase obstacle pixels before any processing or copy
        copy(img_rgb, img_rgb0);
        tc = high_resolution_time() - tc0;

        // My simulator state
        int my_idx = (PLAYER_NUM <= 1) ? 1 : 2;
        double my_x  = S1->P[my_idx]->x[2];
        double my_y  = S1->P[my_idx]->x[3];
        double theta = S1->P[my_idx]->x[1];
        // Soft clamp -- keeps display sensible if simulator allows slight overshoot
        if (my_x < 0) my_x=0; if (my_x > IMG_W) my_x=IMG_W;
        if (my_y < 0) my_y=0; if (my_y > IMG_H) my_y=IMG_H;

        // --- Vision ---
        double tgt_ic, tgt_jc, nav_ic, nav_jc, h;
        int hv;
        found = find_opponent(tgt_ic, tgt_jc, nav_ic, nav_jc, h, hv, my_x, my_y);
        if (found) {
            opp_tgt_ic=tgt_ic; opp_tgt_jc=tgt_jc;
            opp_nav_ic=nav_ic; opp_nav_jc=nav_jc;
            if (hv) { opp_heading=h; heading_valid=1; }
            lost_frames=0;
        } else {
            lost_frames++;
            heading_valid=0;
        }

        // --- Geometry ---
        double nav_dx = opp_nav_ic-my_x, nav_dy = opp_nav_jc-my_y;
        double dist   = sqrt(nav_dx*nav_dx + nav_dy*nav_dy);
        double nav_err = angle_diff(theta, atan2(nav_dy, nav_dx));

        // (Defence robot: no aim_err or laser cooldown needed)

        // --- Live obstacle avoidance: find CLOSEST obstacle within range ---
        // Check FRONT, CENTRE, and REAR of robot body against all obstacles.
        static int  obs_frames  = 0;
        static double obs_esc_h = 0.0;
        bool   near_obs   = false;
        bool   obs_emerg  = false;
        double obs_escape = 0.0;
        double obs_min_d  = 1e9;
        int    obs_closest_k = -1;
        double body_pts[3][2] = {
            { my_x,                        my_y },
            { my_x + BODY_R*cos(theta),    my_y + BODY_R*sin(theta) },
            { my_x - BODY_R*cos(theta),    my_y - BODY_R*sin(theta) }
        };
        for (int k = 0; k < S1->N_obs; k++) {
            for (int bp = 0; bp < 3; bp++) {
                double odx = body_pts[bp][0] - S1->x_obs[k];
                double ody = body_pts[bp][1] - S1->y_obs[k];
                double od  = sqrt(odx*odx + ody*ody);
                if (od < OBS_RADIUS && od < obs_min_d) {
                    obs_min_d  = od;
                    near_obs   = true;
                    obs_escape = atan2(my_y - S1->y_obs[k], my_x - S1->x_obs[k]);
                    obs_emerg  = (od < OBS_EMERG);
                    obs_closest_k = k;
                }
            }
        }
        // Hysteresis: lock escape heading for full duration to prevent oscillation.
        if (near_obs && obs_frames <= 0) {
            obs_frames = 15; obs_esc_h = obs_escape;
        } else if (near_obs) {
            obs_escape = obs_esc_h;
        } else if (obs_frames > 0) {
            obs_frames--; near_obs = true; obs_escape = obs_esc_h;
        }

        // --- Wall check with hysteresis ---
        // Once triggered, stay active for 45 frames so the robot clears the wall
        // before the state machine resumes. Eliminates wall-boundary oscillation.
        static int wall_frames = 0;
        {
            bool hit_wall = (my_x < WALL_MARGIN || my_x > IMG_W-WALL_MARGIN ||
                             my_y < WALL_MARGIN || my_y > IMG_H-WALL_MARGIN);
            if (hit_wall) wall_frames = 25;
            else if (wall_frames > 0) wall_frames--;
        }
        bool near_wall = (wall_frames > 0);

        // --- Threat detection: is opponent aiming at me? ---
        bool opp_aiming = false;
        if (heading_valid && dist < AIM_DIST) {
            double ang_to_me = atan2(my_y-opp_nav_jc, my_x-opp_nav_ic);
            if (fabs(angle_diff(opp_heading, ang_to_me)) < AIM_TOL)
                opp_aiming = true;
        }

        // --- Find best obstacle to hide behind ---
        // Defence strategy: put an obstacle between self and attacker.
        double hide_x = IMG_W/2.0, hide_y = IMG_H/2.0;  // fallback: arena centre
        bool   can_hide = false;
        if (found && S1->N_obs > 0) {
            double best_score = -1e9;
            for (int k = 0; k < S1->N_obs; k++) {
                double ox = S1->x_obs[k], oy = S1->y_obs[k];
                // Target point: on the far side of the obstacle from the opponent
                double opp2obs_x = ox - opp_nav_ic, opp2obs_y = oy - opp_nav_jc;
                double opp2obs_d = sqrt(opp2obs_x*opp2obs_x + opp2obs_y*opp2obs_y);
                if (opp2obs_d < 1.0) continue;
                // Hide spot: 160px past obstacle centre, away from opponent
                double hx = ox + 160.0 * (opp2obs_x / opp2obs_d);
                double hy = oy + 160.0 * (opp2obs_y / opp2obs_d);
                // Clamp to arena bounds
                if (hx < WALL_MARGIN+20) hx = WALL_MARGIN+20;
                if (hx > IMG_W-WALL_MARGIN-20) hx = IMG_W-WALL_MARGIN-20;
                if (hy < WALL_MARGIN+20) hy = WALL_MARGIN+20;
                if (hy > IMG_H-WALL_MARGIN-20) hy = IMG_H-WALL_MARGIN-20;
                // Score: prefer spots that are close to us and far from opponent
                double d_me  = sqrt((hx-my_x)*(hx-my_x) + (hy-my_y)*(hy-my_y));
                double d_opp = sqrt((hx-opp_nav_ic)*(hx-opp_nav_ic) + (hy-opp_nav_jc)*(hy-opp_nav_jc));
                double score = d_opp - 0.5 * d_me;  // prefer far from opp, close to me
                if (score > best_score) { best_score = score; hide_x = hx; hide_y = hy; can_hide = true; }
            }
        }

        // --- Compute flee direction (away from opponent, with obstacle repulsion) ---
        double flee_x = 0, flee_y = 0;
        if (found) {
            // Base direction: directly away from opponent
            double away_x = my_x - opp_nav_ic, away_y = my_y - opp_nav_jc;
            double away_d = sqrt(away_x*away_x + away_y*away_y);
            if (away_d > 1.0) { flee_x = away_x/away_d; flee_y = away_y/away_d; }
            // Add obstacle repulsion so we don't flee into an obstacle
            const double K_REP = 2.0;
            for (int k = 0; k < S1->N_obs; k++) {
                double odx = my_x - S1->x_obs[k];
                double ody = my_y - S1->y_obs[k];
                double od  = sqrt(odx*odx + ody*ody);
                if (od < PF_RADIUS && od > 1.0) {
                    double rep = K_REP * (PF_RADIUS*PF_RADIUS/(od*od) - 1.0);
                    if (rep > 10.0) rep = 10.0;
                    flee_x += rep * (odx / od);
                    flee_y += rep * (ody / od);
                }
            }
            // Add wall repulsion so we don't flee into a wall
            if (my_x < WALL_MARGIN+40) flee_x += 3.0;
            if (my_x > IMG_W-WALL_MARGIN-40) flee_x -= 3.0;
            if (my_y < WALL_MARGIN+40) flee_y += 3.0;
            if (my_y > IMG_H-WALL_MARGIN-40) flee_y -= 3.0;
        }
        double flee_err = angle_diff(theta, atan2(flee_y, flee_x));

        // --- Compute hide direction (toward hide spot, with obstacle repulsion) ---
        double hide_dx = hide_x - my_x, hide_dy = hide_y - my_y;
        double hide_dist = sqrt(hide_dx*hide_dx + hide_dy*hide_dy);
        // Add obstacle repulsion to hide path
        double hide_pf_x = hide_dx, hide_pf_y = hide_dy;
        {
            double mag = sqrt(hide_pf_x*hide_pf_x + hide_pf_y*hide_pf_y);
            if (mag > 1.0) { hide_pf_x /= mag; hide_pf_y /= mag; }
            const double K_REP = 2.0;
            for (int k = 0; k < S1->N_obs; k++) {
                double odx = my_x - S1->x_obs[k];
                double ody = my_y - S1->y_obs[k];
                double od  = sqrt(odx*odx + ody*ody);
                if (od < PF_RADIUS && od > 1.0) {
                    double rep = K_REP * (PF_RADIUS*PF_RADIUS/(od*od) - 1.0);
                    if (rep > 10.0) rep = 10.0;
                    hide_pf_x += rep * (odx / od);
                    hide_pf_y += rep * (ody / od);
                }
            }
        }
        double hide_err = angle_diff(theta, atan2(hide_pf_y, hide_pf_x));

        // --- State transitions (DEFENCE) ---
        // 0=SEARCH: no opponent visible → spin to find them
        // 1=FLEE:   opponent close or aiming → run away fast
        // 2=STRAFE: opponent at medium range and aiming → dodge perpendicular
        // 3=HIDE:   opponent visible but not immediate threat → get behind obstacle
        if (lost_frames > LOST_THRESH) {
            state = 0;  // lost track → search
        } else if (found) {
            if (opp_aiming && dist < DANGER_DIST) {
                state = 1;  // immediate threat: flee!
            } else if (opp_aiming) {
                state = 2;  // aiming but not super close: strafe to dodge
            } else if (dist < FLEE_DIST) {
                state = 1;  // too close: flee
            } else {
                state = 3;  // safe-ish distance: hide behind obstacle
            }
        }

        // --- Drive decisions (DEFENCE) ---
        if (near_wall) {
            // Rotate toward arena centre, then drive inward
            double err_c = angle_diff(theta, atan2(IMG_H/2.0-my_y, IMG_W/2.0-my_x));
            if (fabs(err_c) > 0.3) rotate_inplace(pw_l, pw_r, err_c, ROT_PW);
            else                   drive_forward  (pw_l, pw_r, err_c, V_FWD, STEER_GAIN);

        } else if (near_obs) {
            double err_o = angle_diff(theta, obs_escape);
            if (obs_emerg) {
                if (fabs(err_o) > PI/2.0) drive_backward(pw_l, pw_r, V_FWD);
                else                      drive_forward  (pw_l, pw_r, err_o, V_FWD, STEER_GAIN);
            } else {
                if (fabs(err_o) > 0.4) rotate_inplace(pw_l, pw_r, err_o, ROT_PW);
                else                   drive_forward  (pw_l, pw_r, err_o, V_FWD, STEER_GAIN);
            }

        } else {
            // --- Simulator-based emergency separation ---
            int opp_sim_idx = (PLAYER_NUM <= 1) ? 2 : 1;
            double rr_dx = my_x - S1->P[opp_sim_idx]->x[2];
            double rr_dy = my_y - S1->P[opp_sim_idx]->x[3];
            double sim_rr_dist = sqrt(rr_dx*rr_dx + rr_dy*rr_dy);

            if (!found && sim_rr_dist < BACKUP_DIST) {
                double esc     = atan2(rr_dy, rr_dx);
                double err_esc = angle_diff(theta, esc);
                if (fabs(err_esc) < PI/2.0) drive_forward(pw_l, pw_r, err_esc, V_FWD, STEER_GAIN);
                else                        drive_backward(pw_l, pw_r, V_FWD);
            } else {
                switch (state) {

                case 0: // SEARCH: slow CCW spin until opponent found
                    pw_l = 1500 + ROT_PW;
                    pw_r = 1500 + ROT_PW;
                    break;

                case 1: // FLEE: run away from opponent at full speed
                    // Higher threshold (0.8 rad ≈ 45°) — keep driving even if direction
                    // isn't perfect. Stopping to rotate gives the attacker time to close in.
                    if (fabs(flee_err) > 0.8)
                        rotate_inplace(pw_l, pw_r, flee_err, ROT_PW);
                    else
                        drive_forward(pw_l, pw_r, flee_err, V_FWD, STEER_GAIN);
                    break;

                case 2: // STRAFE: move perpendicular to opponent line-of-sight
                    {
                        double ang_to_me = atan2(my_y-opp_nav_jc, my_x-opp_nav_ic);
                        double opt1 = ang_to_me + PI/2.0;
                        double opt2 = ang_to_me - PI/2.0;
                        double e1 = angle_diff(theta, opt1);
                        double e2 = angle_diff(theta, opt2);
                        double ev = (fabs(e1) < fabs(e2)) ? e1 : e2;
                        if (fabs(ev) > 0.7) rotate_inplace(pw_l, pw_r, ev, ROT_PW);
                        else                drive_forward  (pw_l, pw_r, ev, V_FWD, STEER_GAIN);
                    }
                    break;

                case 3: // HIDE: navigate to hide spot behind obstacle
                    if (hide_dist < 40.0) {
                        // At hide spot — face away from opponent, hold position
                        double away = atan2(my_y-opp_nav_jc, my_x-opp_nav_ic);
                        double err_a = angle_diff(theta, away);
                        if (fabs(err_a) > 0.3) rotate_inplace(pw_l, pw_r, err_a, ROT_PW/2);
                        else { pw_l = 1500; pw_r = 1500; }
                    } else {
                        if (fabs(hide_err) > 0.7) rotate_inplace(pw_l, pw_r, hide_err, ROT_PW);
                        else                      drive_forward(pw_l, pw_r, hide_err, V_FWD, STEER_GAIN);
                    }
                    break;
                }
            }
        }

        // --- Defence robot does NOT fire ---
        laser = 0;

        set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
        if (PLAYER_NUM != 2) view_rgb_image(img_rgb, 1);

        // --- Determine drive reason string for debug ---
        const char *drive_reason = "SM";
        if (near_wall)  drive_reason = "WALL";
        else if (near_obs && obs_emerg) drive_reason = "OBS_EMRG";
        else if (near_obs)              drive_reason = "OBS_WARN";

        static int frame=0;
        if (++frame % 60 == 0) {
            cout << "\n[t=" << (int)tc << "s] " << snames[state]
                 << " drv=" << drive_reason
                 << "  me=(" << (int)my_x << "," << (int)my_y
                 << " th=" << (int)(theta*180.0/PI) << ")"
                 << "  opp=(" << (int)opp_nav_ic << "," << (int)opp_nav_jc << ")"
                 << "  d=" << (int)dist
                 << "  pw=" << pw_l << "/" << pw_r;
            if (obs_closest_k >= 0)
                cout << "  obs" << obs_closest_k << "_d=" << (int)obs_min_d;
            if (opp_aiming) cout << " [THREAT]";
            if (can_hide) cout << "  hide=(" << (int)hide_x << "," << (int)hide_y << ")";
        }
    }

    free_image(img_rgb);  free_image(img_rgb0);
    free_image(img_grey); free_image(img_bin);
    free_image(img_tmp);  free_image(img_lbl);
    deactivate_vision();
    deactivate_simulation();
    cout << "\ndone.\n";
    return 0;
}


// MECH 472/663 - Player 2 DEFENCE
// Evade the opponent robot and avoid getting hit by its laser.
// mode = 2 (two player, player #2 controls robot_B)
//
// Vision pipeline based on prof's examples (vision_example_6, simulator_example2):
//   copy -> lowpass_filter -> scale -> threshold -> invert -> erode -> dialate
//   -> label_image -> centroid2
//
// State machine: SEARCH -> FLEE -> STRAFE -> HIDE
//
// Rules:
// - can't leave the screen (lose the turn)
// - can't hit obstacles (marks deducted)
// - obstacles block the laser - use them as cover

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
// Same pipeline as offence: threshold -> label -> centroid2 -> pair blobs
// Identifies self (closest to known position) vs opponent (farthest)
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

    // prof's vision pipeline
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

        if (is_obstacle_colour(R, G, B)) {
            draw_point_rgb(img_rgb, (int)ic, (int)jc, 255, 255, 0);
            continue;
        }

        b_ic[n_blobs] = ic; b_jc[n_blobs] = jc; b_area[n_blobs] = area;
        b_R[n_blobs] = R; b_G[n_blobs] = G; b_B[n_blobs] = B;
        n_blobs++;
    }
    if (n_blobs == 0) return 0;

    // pair blobs as the two markers of one robot
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

    // build candidate list: pairs first, then singletons
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
        if (cand_d[0] < 60.0) return 0;
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
// drive helpers (based on prof's servo model: left servo flipped)
// -----------------------------------------------------------------------
int clamp_pw(int pw)
{
    if (pw < 1000) return 1000;
    if (pw > 2000) return 2000;
    return pw;
}

void drive_forward(int &pw_l, int &pw_r, double heading_err,
                   int v_fwd, double gain)
{
    int steer = (int)(heading_err * gain);
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

    // --- simulation setup (must match player1) ---
    double width1 = IMG_W, height1 = IMG_H;
    const int N_obs = 2;
    double x_obs[N_obs] = { 320, 320 }; //270.5 135 
    double y_obs[N_obs] = { 240, 240 };
    char obstacle_file[N_obs][S_MAX] = {
        "obstacle_black.bmp", "obstacle_green.bmp"
    };
    double D = 121.0, Lx = 31.0, Ly = 0.0, Ax = 37.0, Ay = 0.0;
    double alpha_max = PI / 2.0;
    int n_robot = 2;

    int pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
    double max_speed = 180.0;

    // --- vision state ---
    double opp_nav_ic = IMG_W/2.0, opp_nav_jc = IMG_H/2.0;
    double opp_tgt_ic = IMG_W/2.0, opp_tgt_jc = IMG_H/2.0;
    double opp_heading = 0.0;
    int found = 0, heading_valid = 0, lost_frames = 0;

    // --- state machine: 0=SEARCH  1=FLEE  2=STRAFE  3=HIDE ---
    int state = 0;
    static const char *snames[] = {"SEARCH", "FLEE", "STRAFE", "HIDE"};

    // --- tuning constants (defence) ---
    const int    V_FWD       = 300;     // forward pw offset (faster to outrun attacker)
    const double STEER_GAIN  = 180.0;   // proportional steer
    const int    ROT_PW      = 190;     // rotation pw offset (faster turning)
    const double FLEE_DIST   = 250.0;   // start fleeing when opponent closer than this
    const double DANGER_DIST = 150.0;   // opponent very close - maximum urgency
    const double BACKUP_DIST = 120.0;   // back away when this close
    const double OBS_RADIUS  =  95.0;   // body-point avoidance radius
    const double OBS_EMERG   =  55.0;   // emergency zone around obstacles
    const double PF_RADIUS   = 180.0;   // potential field deflection radius
    const int    WALL_MARGIN = 100;     // wall avoidance margin (pixels)
    const int    LOST_THRESH =  25;     // frames without detection before SEARCH
    const double AIM_TOL     = 0.30;    // rad: opponent heading tolerance for threat
    const double AIM_DIST    = 350.0;   // detect threat from this distance
    const double BODY_R      =  60.0;   // half robot body length (D/2)

    cout << "\n=== PLAYER 2 - DEFENCE ===";
    cout << "\nState machine: SEARCH -> FLEE / STRAFE / HIDE";
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

    join_player();

    double tc0 = high_resolution_time(), tc;

    while (1) {

        acquire_image_sim(img_rgb);
        mask_obstacles(img_rgb);
        copy(img_rgb, img_rgb0);
        tc = high_resolution_time() - tc0;

        // --- my robot state from simulator (player 2 = index 2) ---
        double my_x  = S1->P[2]->x[2];
        double my_y  = S1->P[2]->x[3];
        double theta = S1->P[2]->x[1];
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
        double nav_dx = opp_nav_ic - my_x, nav_dy = opp_nav_jc - my_y;
        double dist   = sqrt(nav_dx*nav_dx + nav_dy*nav_dy);

        // --- obstacle avoidance: check body front, centre, rear ---
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
        // hysteresis: lock escape heading to prevent oscillation
        if (near_obs && obs_frames <= 0) {
            obs_frames = 15; obs_esc_h = obs_escape;
        } else if (near_obs) {
            obs_escape = obs_esc_h;
        } else if (obs_frames > 0) {
            obs_frames--; near_obs = true; obs_escape = obs_esc_h;
        }

        // --- wall avoidance with hysteresis ---
        static int wall_frames = 0;
        {
            bool hit_wall = (my_x < WALL_MARGIN || my_x > IMG_W-WALL_MARGIN ||
                             my_y < WALL_MARGIN || my_y > IMG_H-WALL_MARGIN);
            if (hit_wall) wall_frames = 25;
            else if (wall_frames > 0) wall_frames--;
        }
        bool near_wall = (wall_frames > 0);

        // --- threat detection: is opponent aiming at me? ---
        bool opp_aiming = false;
        if (heading_valid && dist < AIM_DIST) {
            double ang_to_me = atan2(my_y - opp_nav_jc, my_x - opp_nav_ic);
            if (fabs(angle_diff(opp_heading, ang_to_me)) < AIM_TOL)
                opp_aiming = true;
        }

        // --- find best obstacle to hide behind ---
        double hide_x = IMG_W/2.0, hide_y = IMG_H/2.0;
        bool   can_hide = false;
        if (found && S1->N_obs > 0) {
            double best_score = -1e9;
            for (int k = 0; k < S1->N_obs; k++) {
                double ox = S1->x_obs[k], oy = S1->y_obs[k];
                double opp2obs_x = ox - opp_nav_ic, opp2obs_y = oy - opp_nav_jc;
                double opp2obs_d = sqrt(opp2obs_x*opp2obs_x + opp2obs_y*opp2obs_y);
                if (opp2obs_d < 1.0) continue;
                // hide spot: 160px past obstacle centre, away from opponent
                double hx = ox + 160.0 * (opp2obs_x / opp2obs_d);
                double hy = oy + 160.0 * (opp2obs_y / opp2obs_d);
                // clamp to arena bounds
                if (hx < WALL_MARGIN+20) hx = WALL_MARGIN+20;
                if (hx > IMG_W-WALL_MARGIN-20) hx = IMG_W-WALL_MARGIN-20;
                if (hy < WALL_MARGIN+20) hy = WALL_MARGIN+20;
                if (hy > IMG_H-WALL_MARGIN-20) hy = IMG_H-WALL_MARGIN-20;
                // score: prefer spots far from opponent, close to us
                double d_me  = sqrt((hx-my_x)*(hx-my_x) + (hy-my_y)*(hy-my_y));
                double d_opp = sqrt((hx-opp_nav_ic)*(hx-opp_nav_ic) + (hy-opp_nav_jc)*(hy-opp_nav_jc));
                double score = d_opp - 0.5 * d_me;
                if (score > best_score) { best_score = score; hide_x = hx; hide_y = hy; can_hide = true; }
            }
        }

        // --- line-of-sight check ---
        // Cast a line segment from the opponent to our robot front.
        // If any obstacle is close enough to that line, it blocks the
        // opponent's laser -- we are hidden and can stop moving.
        //
        // The math: project the obstacle centre onto the segment
        // opponent->robot_front, find the closest point on the segment,
        // then measure the perpendicular distance. If that distance
        // is less than LOS_BLOCK_R the obstacle blocks the view.
        //
        // (same approach as Naim's defence, with a more realistic radius)
        const double LOS_BLOCK_R = 55.0; // obstacle ~42px radius + margin
        const double FRONT_OFFSET = 65.0;

        // robot front position (where the laser would hit)
        double front_x = my_x + FRONT_OFFSET * cos(theta);
        double front_y = my_y + FRONT_OFFSET * sin(theta);

        bool hidden = false;
        if (found) {
            for (int k = 0; k < S1->N_obs; k++) {
                double ox = S1->x_obs[k], oy = S1->y_obs[k];

                // only count as hidden if we are actually near this
                // obstacle (within 120px). otherwise we might be on
                // the far side of the arena and an obstacle just
                // happens to be on the line -- that's not cover
                double d_me_obs = sqrt((my_x-ox)*(my_x-ox) + (my_y-oy)*(my_y-oy));
                if (d_me_obs > 120.0) continue;

                // segment vector: opponent -> our front
                double seg_x = front_x - opp_nav_ic;
                double seg_y = front_y - opp_nav_jc;
                double seg_len2 = seg_x*seg_x + seg_y*seg_y + 1e-6;

                // project obstacle onto segment, clamp t to [0,1]
                double t = ((ox - opp_nav_ic)*seg_x + (oy - opp_nav_jc)*seg_y) / seg_len2;
                if (t < 0.0) t = 0.0;
                if (t > 1.0) t = 1.0;

                // closest point on segment to obstacle centre
                double cx = opp_nav_ic + t * seg_x;
                double cy = opp_nav_jc + t * seg_y;

                // perpendicular distance
                double perp = sqrt((ox-cx)*(ox-cx) + (oy-cy)*(oy-cy));

                if (perp < LOS_BLOCK_R) { hidden = true; break; }
            }
        }

        // --- compute flee direction (away from opponent, with obstacle + wall repulsion) ---
        double flee_x = 0, flee_y = 0;
        if (found) {
            double away_x = my_x - opp_nav_ic, away_y = my_y - opp_nav_jc;
            double away_d = sqrt(away_x*away_x + away_y*away_y);
            if (away_d > 1.0) { flee_x = away_x/away_d; flee_y = away_y/away_d; }
            // obstacle repulsion
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
            // wall repulsion
            if (my_x < WALL_MARGIN+40) flee_x += 3.0;
            if (my_x > IMG_W-WALL_MARGIN-40) flee_x -= 3.0;
            if (my_y < WALL_MARGIN+40) flee_y += 3.0;
            if (my_y > IMG_H-WALL_MARGIN-40) flee_y -= 3.0;
        }
        double flee_err = angle_diff(theta, atan2(flee_y, flee_x));

        // --- compute hide direction (toward hide spot, with obstacle repulsion) ---
        double hide_dx = hide_x - my_x, hide_dy = hide_y - my_y;
        double hide_dist = sqrt(hide_dx*hide_dx + hide_dy*hide_dy);
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

        // --- state transitions ---
        if (lost_frames > LOST_THRESH) {
            state = 0;  // lost track -> search
        } else if (found) {
            if (opp_aiming && dist < DANGER_DIST) {
                state = 1;  // immediate threat: flee
            } else if (opp_aiming) {
                state = 2;  // aiming but not close: strafe to dodge
            } else if (dist < FLEE_DIST) {
                state = 1;  // too close: flee
            } else if (can_hide) {
                state = 3;  // safe distance: hide behind obstacle
            } else {
                state = 1;  // no obstacle to hide behind: flee
            }
        }

        // --- drive decisions ---
        // priority: wall > obstacle > state machine
        if (near_wall) {
            double err_c = angle_diff(theta, atan2(IMG_H/2.0-my_y, IMG_W/2.0-my_x));
            if (fabs(err_c) > 0.3) rotate_inplace(pw_l, pw_r, err_c, ROT_PW);
            else                   drive_forward(pw_l, pw_r, err_c, V_FWD, STEER_GAIN);

        } else if (near_obs && state != 3) {
            // suppress obstacle avoidance when in HIDE state --
            // the potential field in hide_pf already handles obstacles,
            // and we NEED to be near the obstacle to use it as cover
            double err_o = angle_diff(theta, obs_escape);
            if (obs_emerg) {
                if (fabs(err_o) > PI/2.0) drive_backward(pw_l, pw_r, V_FWD);
                else                      drive_forward(pw_l, pw_r, err_o, V_FWD, STEER_GAIN);
            } else {
                if (fabs(err_o) > 0.4) rotate_inplace(pw_l, pw_r, err_o, ROT_PW);
                else                   drive_forward(pw_l, pw_r, err_o, V_FWD, STEER_GAIN);
            }

        } else {
            // emergency separation using simulator positions
            double rr_dx = my_x - S1->P[1]->x[2];
            double rr_dy = my_y - S1->P[1]->x[3];
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
                    // always drive forward with proportional steering
                    // never pure rotation -- stopping to spin gives
                    // the attacker time to close in
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
                        else                drive_forward(pw_l, pw_r, ev, V_FWD, STEER_GAIN);
                    }
                    break;

                case 3: // HIDE: navigate to hide spot behind obstacle
                    if (hidden) {
                        // obstacle is blocking opponent's line of sight
                        // stop and hold -- we are safe here
                        pw_l = 1500; pw_r = 1500;
                    } else if (hide_dist < 40.0) {
                        // at hide spot but not yet hidden: face away from opponent
                        double away = atan2(my_y-opp_nav_jc, my_x-opp_nav_ic);
                        double err_a = angle_diff(theta, away);
                        if (fabs(err_a) > 0.3) rotate_inplace(pw_l, pw_r, err_a, ROT_PW/2);
                        else { pw_l = 1500; pw_r = 1500; }
                    } else {
                        // drive toward hiding spot
                        if (fabs(hide_err) > 0.7) rotate_inplace(pw_l, pw_r, hide_err, ROT_PW);
                        else                      drive_forward(pw_l, pw_r, hide_err, V_FWD, STEER_GAIN);
                    }
                    break;
                }
            }
        }

        // --- defence does NOT fire ---
        laser = 0;

        set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
        // player 2 does NOT display -- only player 1 shows the image
        // calling view_rgb_image from both players causes display twitching

        // --- console output ---
        const char *drive_reason = "SM";
        if (near_wall)  drive_reason = "WALL";
        else if (near_obs && obs_emerg) drive_reason = "OBS_EMRG";
        else if (near_obs)              drive_reason = "OBS_WARN";

        static int frame = 0;
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
            if (hidden) cout << " [HIDDEN]";
            if (can_hide) cout << "  hide=(" << (int)hide_x << "," << (int)hide_y << ")";
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


// PLAYER 2

#include <cstdio>
#include <cmath>
#include <iostream>
#include <Windows.h>
using namespace std;
#define KEY(c) ( GetAsyncKeyState((int)(c)) & (SHORT)0x8000 )
#include "image_transfer.h"
#include "vision.h"
#include "robot.h"
#include "vision_simulation.h"
#include "update_simulation.h"
#include "shared_memory.h"

extern robot_system* S1;
extern char* p_shared;

const int    IMG_W = 640, IMG_H = 480;
const double PI = 3.14159265;

const double ROBOT_HARD = 95.0;
const double ROBOT_SOFT = 130.0;

image img_rgb, img_rgb0, img_grey, img_bin, img_tmp, img_lbl;

// -----------------------------------------------------------------------
// centroid2 -- centroid + average RGB of a labelled blob
// -----------------------------------------------------------------------
int centroid2(image& rgb, image& label, int i_label,
    double& ic, double& jc,
    int& R_ave, int& G_ave, int& B_ave, double& n_pixels)
{
    ibyte* p = rgb.pdata;
    i2byte* pl = (i2byte*)label.pdata;
    const double EPS = 1e-7;
    double mi = 0, mj = 0, m = 0, n = 0, Rs = 0, Gs = 0, Bs = 0;
    for (int j = 0; j < IMG_H; j++) for (int i = 0; i < IMG_W; i++) {
        if (pl[j * IMG_W + i] == (i2byte)i_label) {
            ibyte* pc = p + 3 * (j * IMG_W + i);
            Bs += pc[0]; Gs += pc[1]; Rs += pc[2];
            n++; m++; mi += i; mj += j;
        }
    }
    ic = mi / (m + EPS);  jc = mj / (m + EPS);
    R_ave = (int)(Rs / (n + EPS));
    G_ave = (int)(Gs / (n + EPS));
    B_ave = (int)(Bs / (n + EPS));
    n_pixels = n;
    return 0;
}

// -----------------------------------------------------------------------
// is_obstacle_colour -- black and green only.
// Do NOT exclude orange/red: robot_A carries those and we want to detect it.
// -----------------------------------------------------------------------
int is_obstacle_colour(int R, int G, int B)
{
    if (R < 35 && G < 35 && B < 35)          return 1; // black obstacle
    if (G > 80 && G > R + 30 && G > B + 25)      return 1; // green obstacle
    return 0;
}

// -----------------------------------------------------------------------
// mask_obstacles -- paint obstacle footprints gray so they disappear in pipeline
// -----------------------------------------------------------------------
void mask_obstacles(image& img)
{
    const int R = 62;
    ibyte* p = img.pdata;
    for (int k = 0; k < S1->N_obs; k++) {
        int cx = (int)S1->x_obs[k], cy = (int)S1->y_obs[k];
        for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) {
            if (dx * dx + dy * dy > R * R) continue;
            int px = cx + dx, py = cy + dy;
            if (px < 0 || px >= IMG_W || py < 0 || py >= IMG_H) continue;
            ibyte* pix = p + 3 * (py * IMG_W + px);
            pix[0] = pix[1] = pix[2] = 200;
        }
    }
}

// -----------------------------------------------------------------------
// angle_diff -- signed angular difference wrapped to [-pi, pi]
// -----------------------------------------------------------------------
double angle_diff(double a, double b)
{
    double d = b - a;
    while (d > PI) d -= 2 * PI;
    while (d < -PI) d += 2 * PI;
    return d;
}

// -----------------------------------------------------------------------
// draw_line_rgb -- Bresenham line
// -----------------------------------------------------------------------
void draw_line_rgb(image& img, int x0, int y0, int x1, int y1, int R, int G, int B)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    for (;;) {
        if (x0 >= 0 && x0 < img.width && y0 >= 0 && y0 < img.height)
            draw_point_rgb(img, x0, y0, R, G, B);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// -----------------------------------------------------------------------
// clear_shot -- returns 1 if no obstacle blocks the line from (ax,ay) to (bx,by)
// -----------------------------------------------------------------------
int clear_shot(double ax, double ay, double bx, double by)
{
    const double OBS_R = 55.0;
    double dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
    if (len2 < 1.0) return 1;
    for (int k = 0; k < S1->N_obs; k++) {
        double ox = S1->x_obs[k], oy = S1->y_obs[k];
        double t = ((ox - ax) * dx + (oy - ay) * dy) / len2;
        t = t < 0 ? 0 : t>1 ? 1 : t;
        double cx = ax + t * dx, cy = ay + t * dy;
        if ((ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) < OBS_R * OBS_R) return 0;
    }
    return 1;
}

// -----------------------------------------------------------------------
// find_opponent -- vision pipeline.
// Pipeline: lowpass -> scale -> threshold(70) -> invert -> erode -> dilate
// Labels dark blobs. Self = closest blob, opponent = farthest blob.
// Returns 1 and sets opp_x/opp_y. Also sets opp_h/heading_valid if paired.
// -----------------------------------------------------------------------
// set to 1 to print full blob table every frame; 0 for summary only
#define DBG_VISION 0

int find_opponent(double& opp_x, double& opp_y,
    double& opp_h, int& heading_valid,
    double my_x, double my_y)
{
    heading_valid = 0;

    copy(img_rgb0, img_grey);
    lowpass_filter(img_grey, img_tmp); copy(img_tmp, img_grey);
    scale(img_grey, img_grey);
    threshold(img_grey, img_bin, 70);
    invert(img_bin, img_bin);
    erode(img_bin, img_tmp); copy(img_tmp, img_bin);
    dialate(img_bin, img_tmp); copy(img_tmp, img_bin);
    dialate(img_bin, img_tmp); copy(img_tmp, img_bin);

    int nlabels;
    label_image(img_bin, img_lbl, nlabels);

#if DBG_VISION
    printf("\n--- VISION  my=(%.0f,%.0f)  nlabels=%d ---\n", my_x, my_y, nlabels);
#endif

    if (nlabels == 0) {
#if DBG_VISION
        printf("  [no labels] -> not found\n");
#endif
        return 0;
    }

    const int MAX_BLOBS = 40;
    double b_ic[MAX_BLOBS], b_jc[MAX_BLOBS], b_area[MAX_BLOBS];
    int    b_R[MAX_BLOBS], b_G[MAX_BLOBS], b_B[MAX_BLOBS];
    int    b_reject[MAX_BLOBS];   // 0=kept, 1=area, 2=obs_excl, 3=obs_colour, 4=self_excl
    int n_blobs = 0;

    // ---- pass 1: collect all blobs above area threshold ----
    const int MAX_ALL = 40;
    double all_ic[MAX_ALL], all_jc[MAX_ALL], all_area[MAX_ALL];
    int    all_R[MAX_ALL], all_G[MAX_ALL], all_B[MAX_ALL];
    int    all_rej[MAX_ALL];
    int n_all = 0;

    for (int lbl = 1; lbl <= nlabels && n_all < MAX_ALL; lbl++) {
        double ic, jc, area; int R, G, B;
        centroid2(img_rgb0, img_lbl, lbl, ic, jc, R, G, B, area);

        all_ic[n_all] = ic; all_jc[n_all] = jc; all_area[n_all] = area;
        all_R[n_all] = R;   all_G[n_all] = G;   all_B[n_all] = B;
        all_rej[n_all] = 0;

        if (area < 500) { all_rej[n_all] = 1; n_all++; continue; }

        bool obs_excl = false;
        for (int k = 0; k < S1->N_obs; k++) {
            double dx = ic - S1->x_obs[k], dy = jc - S1->y_obs[k];
            if (dx * dx + dy * dy < 30.0 * 30.0) { obs_excl = true; break; }
        }
        if (obs_excl) { all_rej[n_all] = 2; n_all++; continue; }
        if (is_obstacle_colour(R, G, B)) { all_rej[n_all] = 3; n_all++; continue; }

        // passes all filters — check self-exclusion separately below
        b_ic[n_blobs] = ic; b_jc[n_blobs] = jc; b_area[n_blobs] = area;
        b_R[n_blobs] = R;   b_G[n_blobs] = G;   b_B[n_blobs] = B;
        b_reject[n_blobs] = 0;
        double sdx = ic - my_x, sdy = jc - my_y;
        if (sdx * sdx + sdy * sdy < 120.0 * 120.0) b_reject[n_blobs] = 4;  // self-excl
        n_blobs++;
        all_rej[n_all] = 0;
        n_all++;
    }

#if DBG_VISION
    // print ALL blobs (including rejected) with rejection reason
    static const char* rej_str[] = { "OK","<area","obs_pos","obs_col","self_excl" };
    for (int i = 0; i < n_all; i++) {
        double sdx = all_ic[i] - my_x, sdy = all_jc[i] - my_y;
        double dist_me = sqrt(sdx * sdx + sdy * sdy);
        printf("  blob[%d] ic=%5.1f jc=%5.1f area=%6.1f RGB=(%3d,%3d,%3d) distMe=%5.1f  [%s]\n",
            i, all_ic[i], all_jc[i], all_area[i],
            all_R[i], all_G[i], all_B[i], dist_me,
            rej_str[all_rej[i]]);
    }
#endif

    // ---- pass 2: pick largest non-self blob ----
    double best_area = 0;
    int opp_idx = -1;
    for (int i = 0; i < n_blobs; i++) {
        if (b_reject[i] != 0) continue;   // skip self-excluded
        if (b_area[i] > best_area) { best_area = b_area[i]; opp_idx = i; }
    }

    if (opp_idx < 0) {
#if DBG_VISION
        printf("  -> no valid opponent blob (n_blobs=%d, all self-excl or empty)\n", n_blobs);
#endif
        return 0;
    }

    opp_x = b_ic[opp_idx];
    opp_y = b_jc[opp_idx];

#if DBG_VISION
    printf("  PICKED opp_idx=%d  raw=(%5.1f,%5.1f)  area=%.1f  RGB=(%d,%d,%d)\n",
        opp_idx, opp_x, opp_y, b_area[opp_idx],
        b_R[opp_idx], b_G[opp_idx], b_B[opp_idx]);
#endif

    // ---- pass 3: pair nearby blob to estimate opponent heading ----
    int pair_idx = -1; double pair_d = 1e9;
    for (int i = 0; i < n_blobs; i++) {
        if (i == opp_idx) continue;
        double dx = b_ic[i] - opp_x, dy = b_jc[i] - opp_y;
        double d = sqrt(dx * dx + dy * dy);
        if (d >= 45.0 && d <= 140.0 && d < pair_d) { pair_d = d; pair_idx = i; }
    }
    if (pair_idx >= 0) {
        double da = sqrt((b_ic[opp_idx] - my_x) * (b_ic[opp_idx] - my_x) + (b_jc[opp_idx] - my_y) * (b_jc[opp_idx] - my_y));
        double dp = sqrt((b_ic[pair_idx] - my_x) * (b_ic[pair_idx] - my_x) + (b_jc[pair_idx] - my_y) * (b_jc[pair_idx] - my_y));
        double front_ic, front_jc, rear_ic, rear_jc;
        if (da <= dp) { front_ic = b_ic[opp_idx]; front_jc = b_jc[opp_idx]; rear_ic = b_ic[pair_idx]; rear_jc = b_jc[pair_idx]; }
        else { front_ic = b_ic[pair_idx]; front_jc = b_jc[pair_idx]; rear_ic = b_ic[opp_idx]; rear_jc = b_jc[opp_idx]; }
        opp_h = atan2(front_jc - rear_jc, front_ic - rear_ic);
        heading_valid = 1;
#if DBG_VISION
        printf("  HEADING pair_idx=%d  dist=%.1f  opp_h=%.3f rad\n", pair_idx, pair_d, opp_h);
#endif
    }
    else {
#if DBG_VISION
        printf("  no heading pair found\n");
#endif
    }

    return 1;
}

int inside_wall_margin(double x, double y, double wall_margin)
{
    return (x < wall_margin || x > IMG_W - wall_margin ||
            y < wall_margin || y > IMG_H - wall_margin);
}

int inside_obstacle_margin(double x, double y, double obs_margin)
{
    for (int k = 0; k < S1->N_obs; k++) {
        double dx = x - S1->x_obs[k];
        double dy = y - S1->y_obs[k];
        if (dx * dx + dy * dy < obs_margin * obs_margin) return 1;
    }
    return 0;
}

int inside_robot_margin(double my_x, double my_y, double other_x, double other_y, double robot_margin)
{
    double dx = my_x - other_x;
    double dy = my_y - other_y;
    return (dx * dx + dy * dy < robot_margin * robot_margin);
}

double safest_wall_heading(double x, double y)
{
    double dl = x;
    double dr = IMG_W - x;
    double dt = y;
    double db = IMG_H - y;

    double mn = dl;
    double h = 0.0;

    if (dr < mn) { mn = dr; h = PI; }
    if (dt < mn) { mn = dt; h = PI / 2.0; }
    if (db < mn) { mn = db; h = -PI / 2.0; }

    return h;
}

double safest_obstacle_heading(double x, double y)
{
    int closest = 0;
    double best_d2 = 1e18;

    for (int k = 0; k < S1->N_obs; k++) {
        double dx = x - S1->x_obs[k];
        double dy = y - S1->y_obs[k];
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            closest = k;
        }
    }

    return atan2(y - S1->y_obs[closest], x - S1->x_obs[closest]);
}

double safest_robot_heading(double my_x, double my_y, double other_x, double other_y)
{
    return atan2(my_y - other_y, my_x - other_x);
}

int pick_cover_point(double my_x, double my_y,
    double opp_x, double opp_y,
    double& tx, double& ty, int& best_obs_idx)
{
    const double COVER_OFFSET = 65.0;   // smaller so it reaches cover faster
    const double WALL_PAD = 75.0;

    double best_score = -1e18;
    int found = 0;
    best_obs_idx = -1;

    for (int k = 0; k < S1->N_obs; k++) {
        double ox = S1->x_obs[k];
        double oy = S1->y_obs[k];

        // line from opponent to obstacle
        double vx = ox - opp_x;
        double vy = oy - opp_y;
        double vn = sqrt(vx * vx + vy * vy);
        if (vn < 1.0) continue;

        vx /= vn;
        vy /= vn;

        // cover point just behind obstacle from opponent view
        double cx = ox + COVER_OFFSET * vx;
        double cy = oy + COVER_OFFSET * vy;

        if (cx < WALL_PAD) cx = WALL_PAD;
        if (cx > IMG_W - WALL_PAD) cx = IMG_W - WALL_PAD;
        if (cy < WALL_PAD) cy = WALL_PAD;
        if (cy > IMG_H - WALL_PAD) cy = IMG_H - WALL_PAD;

        double d_me = sqrt((cx - my_x) * (cx - my_x) + (cy - my_y) * (cy - my_y));
        double d_opp = sqrt((cx - opp_x) * (cx - opp_x) + (cy - opp_y) * (cy - opp_y));

        // prefer cover that is quick for me and still far from opponent
        double score = -1.4 * d_me + 0.5 * d_opp;

        // slight penalty if target is too close to another obstacle
        for (int j = 0; j < S1->N_obs; j++) {
            if (j == k) continue;
            double ddx = cx - S1->x_obs[j];
            double ddy = cy - S1->y_obs[j];
            double dd = sqrt(ddx * ddx + ddy * ddy);
            if (dd < 95.0) score -= 120.0;
        }

        if (score > best_score) {
            best_score = score;
            tx = cx;
            ty = cy;
            best_obs_idx = k;
            found = 1;
        }
    }

    return found;
}

void enforce_wall_guard(double my_x, double my_y, double my_theta, int& pw_l, int& pw_r)
{
    const double WALL_HARD = 85.0;

    if (my_x >= WALL_HARD && my_x <= IMG_W - WALL_HARD &&
        my_y >= WALL_HARD && my_y <= IMG_H - WALL_HARD) {
        return; // safely inside map
    }

    // always recover toward arena center
    double ang_center = atan2(IMG_H / 2.0 - my_y, IMG_W / 2.0 - my_x);
    double err = angle_diff(my_theta, ang_center);

    // if badly misaligned, rotate in place first
    if (fabs(err) > 0.40) {
        int sign = (err >= 0.0) ? 1 : -1;
        pw_l = 1500 + sign * 360;
        pw_r = 1500 + sign * 360;
    }
    else {
        // once pointing inward enough, drive strongly toward center
        pw_l = 1500 - 320;
        pw_r = 1500 + 320;
    }
}

int near_wall_hard(double x, double y, double pad)
{
    return (x < pad || x > IMG_W - pad || y < pad || y > IMG_H - pad);
}

int near_obstacle_hard(double x, double y, double pad, int& idx_out)
{
    idx_out = -1;
    double best_d2 = 1e18;

    for (int k = 0; k < S1->N_obs; k++) {
        double dx = x - S1->x_obs[k];
        double dy = y - S1->y_obs[k];
        double d2 = dx * dx + dy * dy;
        if (d2 < pad * pad && d2 < best_d2) {
            best_d2 = d2;
            idx_out = k;
        }
    }
    return (idx_out >= 0);
}

void final_safety_guard(double my_x, double my_y, double my_theta, int& pw_l, int& pw_r)
{
    const double WALL_PAD = 88.0;
    const double OBS_PAD = 78.0;
    const double LOOK = 55.0;

    // predict a short point ahead of the robot
    double fx = my_x + LOOK * cos(my_theta);
    double fy = my_y + LOOK * sin(my_theta);

    int obs_idx = -1;

    // priority 1: walls
    if (near_wall_hard(fx, fy, WALL_PAD) || near_wall_hard(my_x, my_y, WALL_PAD - 8.0)) {
        double ang_center = atan2(IMG_H / 2.0 - my_y, IMG_W / 2.0 - my_x);
        double err = angle_diff(my_theta, ang_center);

        if (fabs(err) > 0.35) {
            int sign = (err >= 0.0) ? 1 : -1;
            pw_l = 1500 + sign * 360;
            pw_r = 1500 + sign * 360;
        }
        else {
            pw_l = 1500 - 320;
            pw_r = 1500 + 320;
        }
        return;
    }

    // priority 2: obstacles
    if (near_obstacle_hard(fx, fy, OBS_PAD, obs_idx) ||
        near_obstacle_hard(my_x, my_y, OBS_PAD - 6.0, obs_idx)) {

        double away = atan2(my_y - S1->y_obs[obs_idx], my_x - S1->x_obs[obs_idx]);
        double err = angle_diff(my_theta, away);

        if (fabs(err) > 0.35) {
            int sign = (err >= 0.0) ? 1 : -1;
            pw_l = 1500 + sign * 320;
            pw_r = 1500 + sign * 320;
        }
        else {
            pw_l = 1500 - 260;
            pw_r = 1500 + 260;
        }
        return;
    }
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main()
{
    const int N_obs = 2;
    double x_obs[] = { 200, 440 }, y_obs[] = { 360, 120 };
    char obs_files[N_obs][S_MAX] = { "obstacle_black.bmp", "obstacle_green.bmp" };
    double D = 121, Lx = 31, Ly = 0, Ax = 37, Ay = 0;

    int pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
    double max_speed = 120.0;

    double opp_x = 320, opp_y = 240, opp_h = PI;
    int opp_found = 0, heading_valid = 0, lost_frames = 100;
    double evade_dir = 0.0; int evade_lock = 0;

    double cover_x = 320.0, cover_y = 240.0;
    int cover_obs = -1;
    int cover_lock = 0;

    cout << "\n=== AUTO DEFENCE (Player 2) ===\nPress space to begin.";
    pause();

    activate_vision();
    activate_simulation(IMG_W, IMG_H, x_obs, y_obs, N_obs,
        "robot_B.bmp", "robot_A.bmp", "background.bmp",
        obs_files, D, Lx, Ly, Ax, Ay, PI / 2.0, 2);
    set_simulation_mode(2);
    // Do NOT call set_robot_position / set_opponent_position here.
    // Offense owns position setup; calling these from both sides causes teleportation.
    set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

    img_rgb.type = RGB_IMAGE;   img_rgb.width = IMG_W; img_rgb.height = IMG_H;
    img_rgb0.type = RGB_IMAGE;   img_rgb0.width = IMG_W; img_rgb0.height = IMG_H;
    img_grey.type = GREY_IMAGE;  img_grey.width = IMG_W; img_grey.height = IMG_H;
    img_bin.type = GREY_IMAGE;  img_bin.width = IMG_W; img_bin.height = IMG_H;
    img_tmp.type = GREY_IMAGE;  img_tmp.width = IMG_W; img_tmp.height = IMG_H;
    img_lbl.type = LABEL_IMAGE; img_lbl.width = IMG_W; img_lbl.height = IMG_H;
    allocate_image(img_rgb);  allocate_image(img_rgb0);
    allocate_image(img_grey); allocate_image(img_bin);
    allocate_image(img_tmp);  allocate_image(img_lbl);

    join_player();

    // fetching obstacle data
    {
        int* obs_ready = (int*)(p_shared + 920);
        double* obs_data = (double*)(p_shared + 928);

        while (*obs_ready != 1) Sleep(1);

        x_obs[0] = obs_data[0];
        y_obs[0] = obs_data[1];
        x_obs[1] = obs_data[2];
        y_obs[1] = obs_data[3];

        for (int k = 0; k < N_obs; k++) {
            S1->x_obs[k] = x_obs[k];
            S1->y_obs[k] = y_obs[k];
        }

        printf("\n[P2] obstacles from host:"
            "\n  obs0 = (%.1f, %.1f)"
            "\n  obs1 = (%.1f, %.1f)\n",
            x_obs[0], y_obs[0], x_obs[1], y_obs[1]);
    }

    {
    struct SpawnCandidate {
        double x, y, th;
    };

    // candidate spawn points around the arena edges
    SpawnCandidate cand[4] = {
        {100.0, 240.0, 0.0},      // left, face right
        {540.0, 240.0, PI},       // right, face left
        {320.0, 100.0, PI/2.0},   // top, face down
        {320.0, 380.0, -PI/2.0}   // bottom, face up
    };

    int best_idx = 0;
    double best_score = -1.0;

    for (int i = 0; i < 4; i++) {
        double min_d2 = 1e18;

        for (int k = 0; k < N_obs; k++) {
            double dx = cand[i].x - x_obs[k];
            double dy = cand[i].y - y_obs[k];
            double d2 = dx*dx + dy*dy;
            if (d2 < min_d2) min_d2 = d2;
        }

        if (min_d2 > best_score) {
            best_score = min_d2;
            best_idx = i;
        }
    }

    double spawn_x  = cand[best_idx].x;
    double spawn_y  = cand[best_idx].y;
    double spawn_th = cand[best_idx].th;

    printf("\n[P2 SAFE SPAWN] chosen=(%.1f, %.1f, %.3f)  score=%.1f\n",
           spawn_x, spawn_y, spawn_th, best_score);

    // apply to the correct player slot for defence
    S1->P[2]->x[1] = spawn_th;
    S1->P[2]->x[2] = spawn_x;
    S1->P[2]->x[3] = spawn_y;
    S1->P[2]->x[4] = 0.0;
    S1->P[2]->calculate_outputs();
}

    static int wall_frames = 0;

    while (1) {

        // game-over check
        if (S1->P[1]->laser == 1) {
            pw_l = pw_r = 1500;
            set_inputs(pw_l, pw_r, pw_laser, 0, max_speed);

            cout << "\n=== SHOT DETECTED (GAME OVER) ===";
            pause();
            break;
        }

        acquire_image_sim(img_rgb);
        copy(img_rgb, img_rgb0);
        mask_obstacles(img_rgb0);

        // own pose
        double my_x = S1->P[2]->x[2];
        double my_y = S1->P[2]->x[3];
        double my_theta = S1->P[2]->x[1];

        // true offense pose from shared memory (ground truth for comparison)
        double true_ox = S1->P[1]->x[2];
        double true_oy = S1->P[1]->x[3];
        double true_oh = S1->P[1]->x[1];
        printf("\n=FRAME= me=(%.1f,%.1f,%.3f)  trueOpp=(%.1f,%.1f,%.3f)\n",
            my_x, my_y, my_theta, true_ox, true_oy, true_oh);

        // vision: find opponent
        double new_x, new_y, new_h;
        int hv;
        int found = find_opponent(new_x, new_y, new_h, hv, my_x, my_y);
        if (found) {
            printf("  DETECT raw=(%.1f,%.1f) hv=%d  smooth_before=(%.1f,%.1f)\n",
                new_x, new_y, hv, opp_x, opp_y);
            opp_x = 0.8 * opp_x + 0.2 * new_x;
            opp_y = 0.8 * opp_y + 0.2 * new_y;
            if (hv) {
                double hx = 0.85 * cos(opp_h) + 0.15 * cos(new_h);
                double hy = 0.85 * sin(opp_h) + 0.15 * sin(new_h);
                opp_h = atan2(hy, hx);
                heading_valid = 1;
            }
            opp_found = 1;
            lost_frames = 0;
            printf("  AFTER  smooth=(%.1f,%.1f)  opp_h=%.3f  hv=%d\n",
                opp_x, opp_y, opp_h, heading_valid);
        }
        else {
            lost_frames++;
            heading_valid = 0;
            printf("  NOT FOUND  lf=%d\n", lost_frames);
        }

        // wall avoidance — margin kept at 90 so obstacles (y=120, y=360)
        // stay inside the safe zone and don't conflict with wall avoidance.
        const int WALL_MARGIN = 90;
        const int WALL_CRITICAL = 35;
        {
            bool hit = (my_x < WALL_MARGIN || my_x > IMG_W - WALL_MARGIN ||
                my_y < WALL_MARGIN || my_y > IMG_H - WALL_MARGIN);
            if (hit) wall_frames = 100;  // enough time to spin 180 deg
            else if (wall_frames > 0) wall_frames--;
        }
        bool near_wall = (wall_frames > 0);
        bool critical_wall = (my_x < WALL_CRITICAL || my_x > IMG_W - WALL_CRITICAL ||
            my_y < WALL_CRITICAL || my_y > IMG_H - WALL_CRITICAL);

        // wall escape: face PERPENDICULARLY away from the nearest edge.
        // avoids the "facing wall = facing centre" ambiguity of atan2-to-centre.
        double wall_esc = 0.0;
        {
            double dl = my_x, dr = IMG_W - my_x, dt = my_y, db = IMG_H - my_y;
            double mn = dl; wall_esc = 0.0;        // nearest = left  → face right
            if (dr < mn) { mn = dr; wall_esc = PI; }  // nearest = right → face left
            if (dt < mn) { mn = dt; wall_esc = PI / 2.0; } // nearest = top  → face down
            if (db < mn) { wall_esc = -PI / 2.0; } // nearest = bot  → face up
        }

        // obstacle proximity
        const double OBS_AVOID = 110.0;
        bool near_obs = false;
        int close_obs = 0; double close_d = 1e9;
        for (int k = 0; k < S1->N_obs; k++) {
            double dx = my_x - S1->x_obs[k], dy = my_y - S1->y_obs[k];
            double d = sqrt(dx * dx + dy * dy);
            if (d < close_d) { close_d = d; close_obs = k; }
            if (d < OBS_AVOID) near_obs = true;
        }

        // movement primitives
        auto drive_fwd = [&]() {
            pw_l = 1500 - 260;
            pw_r = 1500 + 260;
            };

        auto drive_bwd = [&]() {
            pw_l = 1500 + 220;
            pw_r = 1500 - 220;
            };

        auto spin_toward = [&](double target) {
            double err = angle_diff(my_theta, target);
            int sign = (err >= 0) ? 1 : -1;
            pw_l = 1500 + sign * 320;
            pw_r = 1500 + sign * 320;
            };

        auto drive_toward = [&](double target) {
            double err = angle_diff(my_theta, target);
            int steer = (int)(180.0 * err);

            if (steer > 180) steer = 180;
            if (steer < -180) steer = -180;

            // move forward while steering instead of spinning in place
            pw_l = 1500 - 240 + steer;
            pw_r = 1500 + 240 + steer;
            };

        auto aligned = [&](double target) {
            return fabs(angle_diff(my_theta, target)) < 0.30;
            };

        // behaviour state label for console
        const char* state_str = "PATROL";

        // threat detection first
        bool threat = false;
        double ang_to_me_dbg = atan2(my_y - opp_y, my_x - opp_x);
        double head_diff_dbg = fabs(angle_diff(opp_h, ang_to_me_dbg));
        int cs_dbg = clear_shot(opp_x, opp_y, my_x, my_y);

        if (heading_valid && lost_frames < 25) {
            threat = (head_diff_dbg < 0.65) && cs_dbg;
        }

        printf("  THREAT: hv=%d lf=%d head_diff=%.2f clear=%d -> threat=%d\n",
            heading_valid, lost_frames, head_diff_dbg, cs_dbg, (int)threat);

        if (critical_wall) {
            double ang_center = atan2(IMG_H / 2.0 - my_y, IMG_W / 2.0 - my_x);
            drive_toward(ang_center);
            state_str = "WALL CRIT -> recover";
        }
        else if (threat) {
            // cover mode has priority over generic obstacle avoidance
            if (cover_lock <= 0 || cover_obs < 0) {
                int new_obs = -1;
                double tx, ty;

                if (pick_cover_point(my_x, my_y, opp_x, opp_y, tx, ty, new_obs)) {
                    cover_x = tx;
                    cover_y = ty;
                    cover_obs = new_obs;
                    cover_lock = 20;
                }
            }
            else {
                cover_lock--;
            }

            double ang_to_cover = atan2(cover_y - my_y, cover_x - my_x);
            double d_cover = sqrt((cover_x - my_x) * (cover_x - my_x) +
                (cover_y - my_y) * (cover_y - my_y));

            // check whether the chosen obstacle is now blocking line of fire
            int in_cover = 0;
            if (cover_obs >= 0) {
                double ox = S1->x_obs[cover_obs];
                double oy = S1->y_obs[cover_obs];
                double dx = ox - opp_x;
                double dy = oy - opp_y;
                double len2 = dx * dx + dy * dy;
                if (len2 > 1.0) {
                    double t = ((my_x - opp_x) * dx + (my_y - opp_y) * dy) / len2;
                    t = t < 0 ? 0 : (t > 1 ? 1 : t);
                    double cx = opp_x + t * dx;
                    double cy = opp_y + t * dy;
                    double dd2 = (my_x - cx) * (my_x - cx) + (my_y - cy) * (my_y - cy);
                    if (dd2 < 60.0 * 60.0) in_cover = 1;
                }
            }

            if (in_cover) {
                pw_l = 1500;
                pw_r = 1500;
                state_str = "EVADE -> hold cover";
            }
            else if (d_cover > 35.0) {
                drive_toward(ang_to_cover);
                state_str = "EVADE -> drive cover";
            }
            else {
                // if close but not properly hidden, sidestep instead of spinning
                double side1 = ang_to_me_dbg + PI / 2.0;
                double side2 = ang_to_me_dbg - PI / 2.0;
                double e1 = fabs(angle_diff(my_theta, side1));
                double e2 = fabs(angle_diff(my_theta, side2));
                double sidestep = (e1 < e2) ? side1 : side2;

                drive_toward(sidestep);
                state_str = "EVADE -> sidestep";
            }
        }
        else if (near_wall) {
            if (aligned(wall_esc)) {
                drive_fwd();
                state_str = "WALL -> drive";
            }
            else {
                drive_toward(wall_esc);
                state_str = "WALL -> steer";
            }
        }
        else if (near_obs) {
            double away = atan2(my_y - S1->y_obs[close_obs], my_x - S1->x_obs[close_obs]);
            drive_toward(away);
            state_str = "OBS -> steer";
        }
        else {
            cover_lock = 0;
            cover_obs = -1;

            if (opp_found && lost_frames < 25) {
                double ang_to_opp = atan2(opp_y - my_y, opp_x - my_x);
                if (fabs(angle_diff(my_theta, ang_to_opp)) < 0.6) {
                    double perp1 = ang_to_opp + PI / 2.0;
                    double perp2 = ang_to_opp - PI / 2.0;
                    double e1 = fabs(angle_diff(my_theta, perp1));
                    double e2 = fabs(angle_diff(my_theta, perp2));
                    double safe_dir = (e1 < e2) ? perp1 : perp2;
                    drive_toward(safe_dir);
                    state_str = "PATROL -> steer";
                }
                else {
                    drive_fwd();
                    state_str = "PATROL";
                }
            }
            else {
                drive_fwd();
                state_str = "PATROL";
            }
        }

        static const char* last_state = "";
        if (state_str != last_state) {
            // state changed: end current line then print a labelled entry
            printf("\n[-->] x=%5.1f  y=%5.1f  th=%6.3f  opp=(%5.1f,%5.1f)  lf=%d  [%s]\n",
                my_x, my_y, my_theta, opp_x, opp_y, lost_frames, state_str);
            last_state = state_str;
        }
        else {
            // no change: overwrite current line with live data
            printf("\r      x=%5.1f  y=%5.1f  th=%6.3f  opp=(%5.1f,%5.1f)  lf=%d  [%s]          ",
                my_x, my_y, my_theta, opp_x, opp_y, lost_frames, state_str);
        }

        enforce_wall_guard(my_x, my_y, my_theta, pw_l, pw_r);

        final_safety_guard(my_x, my_y, my_theta, pw_l, pw_r);

        pw_l = pw_l < 1000 ? 1000 : pw_l > 2000 ? 2000 : pw_l;
        pw_r = pw_r < 1000 ? 1000 : pw_r > 2000 ? 2000 : pw_r;
        laser = 0;
        set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

        // HUD: green arrow = own heading
        draw_line_rgb(img_rgb, (int)my_x, (int)my_y,
            (int)(my_x + 40 * cos(my_theta)), (int)(my_y + 40 * sin(my_theta)), 0, 255, 0);
        view_rgb_image(img_rgb, 2);
        Sleep(10);
    }

    free_image(img_rgb);  free_image(img_rgb0);
    free_image(img_grey); free_image(img_bin);
    free_image(img_tmp);  free_image(img_lbl);
    deactivate_vision();
    deactivate_simulation();
    cout << "\ndone.\n";
    return 0;
}

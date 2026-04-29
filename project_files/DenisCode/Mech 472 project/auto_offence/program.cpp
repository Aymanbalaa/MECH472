
// PLAYER 1

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
#include <ctime>
#include "shared_memory.h"

extern char* p_shared;
extern robot_system* S1;

const int IMG_W   = 640;
const int IMG_H   = 480;
const int MAX_LBL = 50;

const double ROBOT_HARD = 95.0;
const double ROBOT_SOFT = 130.0;

// global images
image img_rgb, img_rgb0, img_grey, img_bin, img_tmp, img_lbl, img_mag, img_theta;

// -----------------------------------------------------------------------
// sobel -- gradient magnitude (mag) and angle (theta) for greyscale image a
// -----------------------------------------------------------------------
int sobel(image& a, image& mag, image& theta)
{
    if (a.height != mag.height || a.width != mag.width ||
        a.height != theta.height || a.width != theta.width ||
        a.type != GREY_IMAGE || mag.type != GREY_IMAGE || theta.type != GREY_IMAGE) {
        printf("\nerror in sobel: size or type mismatch"); return 1;
    }

    i2byte width  = a.width, height = a.height;
    i4byte size   = (i4byte)width * height - 2 * width - 2;

    ibyte *pa1, *pa2, *pa3, *pa4, *pa5, *pa6, *pa7, *pa8, *pa9;
    ibyte *pm = mag.pdata + width + 1, *pt = theta.pdata + width + 1;
    ibyte *p  = a.pdata   + width + 1;

    pa1 = p - width - 1; pa2 = p - width; pa3 = p - width + 1;
    pa4 = p - 1;         pa5 = p;         pa6 = p + 1;
    pa7 = p + width - 1; pa8 = p + width; pa9 = p + width + 1;

    int kx[10] = {0,-1,0,1,-2,0,2,-1,0,1};
    int ky[10] = {0,-1,-2,-1,0,0,0,1,2,1};

    for (i4byte i = 0; i < size; i++) {
        int sx = kx[1]*(*pa1)+kx[2]*(*pa2)+kx[3]*(*pa3)+
                 kx[4]*(*pa4)+kx[5]*(*pa5)+kx[6]*(*pa6)+
                 kx[7]*(*pa7)+kx[8]*(*pa8)+kx[9]*(*pa9);
        int sy = ky[1]*(*pa1)+ky[2]*(*pa2)+ky[3]*(*pa3)+
                 ky[4]*(*pa4)+ky[5]*(*pa5)+ky[6]*(*pa6)+
                 ky[7]*(*pa7)+ky[8]*(*pa8)+ky[9]*(*pa9);
        int M = abs(sx) + abs(sy);  if (M > 255) M = 255;
        *pm = M;
        double A = atan2((double)sy, (double)sx) / 3.14159265 * 180.0;
        *pt = (int)((A + 180.0) / 360.0 * 255.0 + 0.01);
        if (M < 75) *pt = 0;
        pa1++;pa2++;pa3++;pa4++;pa5++;pa6++;pa7++;pa8++;pa9++;pm++;pt++;
    }

    // copy borders from nearest valid row/column
    pm = mag.pdata; pt = theta.pdata;
    i4byte total = (i4byte)width * height;
    for (int i = 0; i < width; i++) {
        pm[i] = pm[i + width];          pm[total-i-1] = pm[total-i-1-width];
        pt[i] = pt[i + width];          pt[total-i-1] = pt[total-i-1-width];
    }
    for (int i = 0, j = 0; i < height; i++, j += width) {
        pm[j] = pm[j+1];               pm[total-j-1] = pm[total-j-2];
        pt[j] = pt[j+1];               pt[total-j-1] = pt[total-j-2];
    }
    return 0;
}

// -----------------------------------------------------------------------
// is_obstacle_colour -- returns 1 if mean colour matches a known obstacle
// -----------------------------------------------------------------------
int is_obstacle_colour(int R, int G, int B)
{
    if (R < 35 && G < 35 && B < 35)                 return 1; // black
    if (G > 80 && G > R + 30 && G > B + 25)         return 1; // green
    if (B > 80 && B > R + 30 && B > G + 25)         return 1; // blue
    if (R > 140 && G > 55 && G < 170 && B < 70)     return 1; // orange
    if (R > 140 && G < 90 && B < 90)                return 1; // red
    return 0;
}

// -----------------------------------------------------------------------
// angle_diff -- returns b - a wrapped to (-pi, pi)
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
// clear_shot -- returns 1 if no obstacle blocks the line from (ax,ay) to (bx,by)
// Uses the closest-point-on-segment test for each obstacle.
// -----------------------------------------------------------------------
int clear_shot(double ax, double ay, double bx, double by)
{
    const double OBS_R = 55.0;   // obstacle radius + clearance buffer
    double dx = bx - ax, dy = by - ay;
    double len2 = dx * dx + dy * dy;
    if (len2 < 1.0) return 1;

    for (int k = 0; k < S1->N_obs; k++) {
        double ox = S1->x_obs[k], oy = S1->y_obs[k];
        // project obstacle centre onto the line segment
        double t = ((ox - ax) * dx + (oy - ay) * dy) / len2;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double cx = ax + t * dx, cy = ay + t * dy;
        double d2 = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        if (d2 < OBS_R * OBS_R) return 0;   // blocked
    }
    return 1;
}

// -----------------------------------------------------------------------
// features -- single labelled blob: centroid, area, mean colour
// -----------------------------------------------------------------------
int features(image& a, image& rgb, image& label, int label_i,
             double& ic, double& jc, double& area,
             double& R_ave, double& G_ave, double& B_ave)
{
    if (rgb.height != label.height || rgb.width != label.width) {
        printf("\nerror in features: size mismatch"); return 1;
    }
    ibyte*  p  = rgb.pdata;
    i2byte* pl = (i2byte*)label.pdata;
    double sx = 0, sy = 0, sR = 0, sG = 0, sB = 0, n = 0;
    for (int j = 0; j < IMG_H; j++) {
        for (int i = 0; i < IMG_W; i++) {
            if (pl[j * IMG_W + i] == label_i) {
                ibyte* px = p + 3 * (j * IMG_W + i);
                sB += px[0]; sG += px[1]; sR += px[2];
                sx += i;     sy += j;     n++;
            }
        }
    }
    const double EPS = 1e-10;
    ic = sx / (n + EPS);   jc = sy / (n + EPS);   area = n;
    R_ave = sR / (n + EPS); G_ave = sG / (n + EPS); B_ave = sB / (n + EPS);
    return 0;
}

// -----------------------------------------------------------------------
// features -- all labels (1-indexed arrays)
// -----------------------------------------------------------------------
int features(image& a, image& rgb, image& label, int n_labels,
             double ic[], double jc[], double area[],
             double R_ave[], double G_ave[], double B_ave[])
{
    for (int i = 1; i <= n_labels; i++)
        features(a, rgb, label, i, ic[i], jc[i], area[i], R_ave[i], G_ave[i], B_ave[i]);
    return 0;
}

// -----------------------------------------------------------------------
// find_opponent -- vision pipeline returning one whole-robot centroid.
// Pipeline: greyscale -> lowpass -> scale -> Sobel -> threshold ->
//           erode x1 -> dilate x2 -> label -> features -> classify.
// Selects the largest non-obstacle blob not near own position.
// Returns 1 and sets opp_x/opp_y if found, 0 otherwise.
// -----------------------------------------------------------------------
int find_opponent(double& opp_x, double& opp_y, double my_x, double my_y)
{
    copy(img_rgb, img_grey);
    lowpass_filter(img_grey, img_tmp);  copy(img_tmp, img_grey);
    scale(img_grey, img_tmp);           copy(img_tmp, img_grey);
    sobel(img_grey, img_mag, img_theta);
    threshold(img_mag, img_bin, 65);
    erode(img_bin, img_tmp);   copy(img_tmp, img_bin);
    dialate(img_bin, img_tmp); copy(img_tmp, img_bin);
    dialate(img_bin, img_tmp); copy(img_tmp, img_bin);

    int nlabels;
    label_image(img_bin, img_lbl, nlabels);
    if (nlabels == 0) return 0;
    if (nlabels > MAX_LBL - 1) nlabels = MAX_LBL - 1;

    double ic[MAX_LBL], jc[MAX_LBL], area[MAX_LBL];
    double R_ave[MAX_LBL], G_ave[MAX_LBL], B_ave[MAX_LBL];
    features(img_bin, img_rgb0, img_lbl, nlabels, ic, jc, area, R_ave, G_ave, B_ave);

    double best = 0;
    int    found = 0;
    for (int i = 1; i <= nlabels; i++) {
        if (area[i] < 200) continue;
        if (is_obstacle_colour((int)R_ave[i], (int)G_ave[i], (int)B_ave[i])) continue;

        // reject blobs centred on a known obstacle position
        bool on_obs = false;
        for (int k = 0; k < S1->N_obs; k++) {
            double dx = ic[i] - S1->x_obs[k], dy = jc[i] - S1->y_obs[k];
            if (dx*dx + dy*dy < 40.0*40.0) { on_obs = true; break; }
        }
        if (on_obs) continue;

        // reject blobs close to own robot (self-exclusion) — 40px, not 60px
        // 60px was too large: rejects the defense robot when it gets close during chase
        double sdx = ic[i] - my_x, sdy = jc[i] - my_y;
        if (sdx*sdx + sdy*sdy < 40.0*40.0) continue;

        // largest remaining blob = opponent whole-body centroid
        if (area[i] > best) {
            best = area[i];
            opp_x = ic[i]; opp_y = jc[i];
            found = 1;
        }
    }
    if (found)
        draw_point_rgb(img_rgb, (int)opp_x, (int)opp_y, 255, 0, 0);
    return found;
}

// -----------------------------------------------------------------------
// draw_line_rgb -- Bresenham line in colour (R,G,B)
// -----------------------------------------------------------------------
void draw_line_rgb(image& img, int x0, int y0, int x1, int y1, int R, int G, int B)
{
    int dx = abs(x1-x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1-y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
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
// main
// -----------------------------------------------------------------------
int main()
{
    srand((unsigned int)time(NULL));
    const double PI = 3.14159265;

    // --- simulation setup ---
    const int N_obs = 2;
    double x_obs[N_obs], y_obs[N_obs];
    char   obs_file[N_obs][S_MAX] = { "obstacle_black.bmp", "obstacle_green.bmp" };

    // random obstacles in the central region of the map
    // keep them away from the left/right spawn sides
    x_obs[0] = 240 + rand() % 161;   // 240 to 400
    y_obs[0] = 100 + rand() % 281;   // 100 to 380

    do {
        x_obs[1] = 240 + rand() % 161;   // 240 to 400
        y_obs[1] = 100 + rand() % 281;   // 100 to 380

        double dx = x_obs[1] - x_obs[0];
        double dy = y_obs[1] - y_obs[0];
        if (dx * dx + dy * dy > 120.0 * 120.0) break;   // keep obstacles apart
    } while (1);

    double D = 121.0, Lx = 31.0, Ly = 0.0, Ax = 37.0, Ay = 0.0;
    int    n_robot = 2;

    int    pw_l = 1500, pw_r = 1500, pw_laser = 1500, laser = 0;
    double max_speed = 120.0;

    double opp_x = IMG_W / 2.0, opp_y = IMG_H / 2.0;
    int    lost_frames = 0, fired = 0;

    double opp_x_f = opp_x, opp_y_f = opp_y;
    double steer_f = 0.0;
    double pw_l_f = 1500.0, pw_r_f = 1500.0;

    // hysteresis flag for obstacle avoidance
    bool avoid_obs_mode = false;

    // states: 0=SEARCH  1=CHASE  2=FIRE
    int state = 0;

    cout << "\n=== AUTO OFFENCE (Player 1) ===\nPress space to begin.";
    pause();



    activate_vision();
    activate_simulation(640, 480, x_obs, y_obs, N_obs,
        "robot_A.bmp", "robot_B.bmp", "background.bmp",
        obs_file, D, Lx, Ly, Ax, Ay, PI / 2.0, 2);

    int* run_id = (int*)(p_shared + 880);
    int* ready = (int*)(p_shared + 904);
    int* obs_ready = (int*)(p_shared + 920);

    // hard reset startup flags for a fresh run
    *ready = 0;
    *obs_ready = 0;

    // new run marker
    *run_id = (int)time(NULL);
    cout << "\n[P1] run_id = " << *run_id << "\n";

    // share obstacle positions for player 2
    {
        int* obs_ready = (int*)(p_shared + 920);
        double* obs_data = (double*)(p_shared + 928);

        *obs_ready = 0;          // invalid while writing
        obs_data[0] = x_obs[0];
        obs_data[1] = y_obs[0];
        obs_data[2] = x_obs[1];
        obs_data[3] = y_obs[1];
        *obs_ready = 1;          // valid
    }

    set_simulation_mode(1);
    set_robot_position(   540.0, 240.0, PI);   // offence on right, facing right
    

    {
        int* ready = (int*)(p_shared + 904);
        *ready = 0; // Lock while writing

        // Pointer for Player 1 Block (Offset 0)
        double* pd1 = (double*)(p_shared + 8); // Skip sample/laser
        pd1[0] = PI; pd1[1] = 540.0; pd1[2] = 240.0;

        // Pointer for Player 2 Block (Offset 500)
        double* pd2 = (double*)(p_shared + 508); // Skip sample/laser
        pd2[0] = 0.0; pd2[1] = 100.0; pd2[2] = 240.0;

        *ready = 1; // Unlock: Player 2 can now safely fetch,
    }

    {
        int* pi = (int*)(p_shared + 500);
        pi += 2;
        double* pd = (double*)pi;

        cout << "\nOFF wrote player2 block:";
        cout << "\n theta = " << pd[0];
        cout << "\n x = " << pd[1];
        cout << "\n y = " << pd[2] << "\n";
    }

    set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);

    img_rgb.type   = RGB_IMAGE;   img_rgb.width   = IMG_W; img_rgb.height   = IMG_H;
    img_rgb0.type  = RGB_IMAGE;   img_rgb0.width  = IMG_W; img_rgb0.height  = IMG_H;
    img_grey.type  = GREY_IMAGE;  img_grey.width  = IMG_W; img_grey.height  = IMG_H;
    img_bin.type   = GREY_IMAGE;  img_bin.width   = IMG_W; img_bin.height   = IMG_H;
    img_tmp.type   = GREY_IMAGE;  img_tmp.width   = IMG_W; img_tmp.height   = IMG_H;
    img_lbl.type   = LABEL_IMAGE; img_lbl.width   = IMG_W; img_lbl.height   = IMG_H;
    img_mag.type   = GREY_IMAGE;  img_mag.width   = IMG_W; img_mag.height   = IMG_H;
    img_theta.type = GREY_IMAGE;  img_theta.width = IMG_W; img_theta.height = IMG_H;

    allocate_image(img_rgb);  allocate_image(img_rgb0);
    allocate_image(img_grey); allocate_image(img_bin);
    allocate_image(img_tmp);  allocate_image(img_lbl);
    allocate_image(img_mag);  allocate_image(img_theta);

    wait_for_player();
    set_opponent_position(100.0, 240.0, 0.0); // defense on left, facing left
    cout << "\njoined.\n";
    double tc0 = high_resolution_time(), tc;

    while (1) {

        // === 1. Capture image ===
        acquire_image_sim(img_rgb);
        copy(img_rgb, img_rgb0);
        tc = high_resolution_time() - tc0;

        // === 2. Own pose from simulation ===
        double my_x   = S1->P[1]->x[2];
        double my_y   = S1->P[1]->x[3];
        double theta  = S1->P[1]->x[1];

        // === 3. Vision: locate opponent centroid ===
        double new_x, new_y;
        int found = find_opponent(new_x, new_y, my_x, my_y);

        if (found) {
            const double alpha_target = 0.20; // smaller = smoother
            opp_x_f = (1.0 - alpha_target) * opp_x_f + alpha_target * new_x;
            opp_y_f = (1.0 - alpha_target) * opp_y_f + alpha_target * new_y;

            opp_x = opp_x_f;
            opp_y = opp_y_f;
            lost_frames = 0;
        }
        else {
            lost_frames++;
        }

        // === 4. Geometry to opponent ===
        double dx = opp_x - my_x, dy = opp_y - my_y;
        double dist        = sqrt(dx * dx + dy * dy);
        double heading_err = angle_diff(theta, atan2(dy, dx));

        // === 5. Proximity checks ===
        const double WALL = 60.0;
        const double LOOKAHEAD = 90.0;

            // current heading look-ahead point
        double future_x = my_x + LOOKAHEAD * cos(theta);
        double future_y = my_y + LOOKAHEAD * sin(theta);

            // robot is near wall already
        bool near_wall = (my_x < WALL || my_x > IMG_W - WALL ||
            my_y < WALL || my_y > IMG_H - WALL);

            // robot is heading toward wall soon
        bool heading_to_wall = (future_x < WALL || future_x > IMG_W - WALL ||
            future_y < WALL || future_y > IMG_H - WALL);

        int    close_obs = 0;
        double close_d = 1e9;
        for (int k = 0; k < S1->N_obs; k++) {
            double odx = my_x - S1->x_obs[k];
            double ody = my_y - S1->y_obs[k];
            double od = sqrt(odx * odx + ody * ody);
            if (od < close_d) {
                close_d = od;
                close_obs = k;
            }
        }

        // hysteresis: enter avoid mode at 150, leave at 180
        if (!avoid_obs_mode && close_d < 150.0) avoid_obs_mode = true;
        if (avoid_obs_mode && close_d > 180.0) avoid_obs_mode = false;

        bool near_obs = avoid_obs_mode;

        // === 6. State transitions ===
        if (lost_frames > 25) {
            state = 0;
        }
        else if (found && state == 0) {
            state = 1;
        }
        else if (found && state == 1 && fabs(heading_err) < 0.15 && dist < 320.0) {
            state = 2;
        }
        else if (state == 2 && fabs(heading_err) > 0.30) {
            state = 1;
        }

        // === 7. Drive commands (wall and obstacle avoidance take priority) ===
        int target_l = 1500;
        int target_r = 1500;
        double steer_cmd = 0.0;

        if (heading_to_wall || near_wall) {
            double ang = atan2(IMG_H / 2.0 - my_y, IMG_W / 2.0 - my_x);
            steer_cmd = 300.0 * angle_diff(theta, ang);

            if (steer_cmd > 450.0) steer_cmd = 450.0;
            if (steer_cmd < -450.0) steer_cmd = -450.0;

            target_l = (int)(1500 - 220 + steer_cmd);
            target_r = (int)(1500 + 220 + steer_cmd);
        }
        else if (near_obs) {
            double away = atan2(my_y - S1->y_obs[close_obs],
                my_x - S1->x_obs[close_obs]);
            steer_cmd = 250.0 * angle_diff(theta, away + PI / 3.0);

            if (steer_cmd > 450.0) steer_cmd = 450.0;
            if (steer_cmd < -450.0) steer_cmd = -450.0;

            target_l = (int)(1500 - 200 + steer_cmd);
            target_r = (int)(1500 + 200 + steer_cmd);
        }
        else if (state == 0) {
            target_l = 1500 + 300;
            target_r = 1500 + 300;
        }
        else if (state == 1) {
            if (dist < 120.0) {
                target_l = 1500 + 200;
                target_r = 1500 - 200;
            }
            else {
                steer_cmd = 160.0 * heading_err;   // reduced from 200

                if (steer_cmd > 350.0) steer_cmd = 350.0;
                if (steer_cmd < -350.0) steer_cmd = -350.0;

                target_l = (int)(1500 - 180 + steer_cmd);
                target_r = (int)(1500 + 180 + steer_cmd);
            }
        }
        else if (state == 2) {
            if (dist > 160.0) {
                steer_cmd = 140.0 * heading_err;   // reduced from 200

                if (steer_cmd > 300.0) steer_cmd = 300.0;
                if (steer_cmd < -300.0) steer_cmd = -300.0;

                target_l = (int)(1500 - 100 + steer_cmd);
                target_r = (int)(1500 + 100 + steer_cmd);
            }
            else {
                int sign = (heading_err >= 0) ? 1 : -1;
                target_l = 1500 + sign * 250;
                target_r = 1500 + sign * 250;
            }
        }

        // clamp targets
        if (target_l < 1000) target_l = 1000;
        if (target_l > 2000) target_l = 2000;
        if (target_r < 1000) target_r = 1000;
        if (target_r > 2000) target_r = 2000;

        // low-pass filter steering and motor commands
        const double alpha_steer = 0.25;
        const double alpha_motor = 0.20;

        steer_f = (1.0 - alpha_steer) * steer_f + alpha_steer * steer_cmd;
        pw_l_f = (1.0 - alpha_motor) * pw_l_f + alpha_motor * target_l;
        pw_r_f = (1.0 - alpha_motor) * pw_r_f + alpha_motor * target_r;

        pw_l = (int)pw_l_f;
        pw_r = (int)pw_r_f;

        // === 8. Fire check: aligned, in range, clear line of sight, no obstacle blocking ===
        if (!fired && tc > 2.0 && state == 2 &&
            fabs(heading_err) < 0.10 && dist < 350.0 &&
            clear_shot(my_x, my_y, opp_x, opp_y)) {

            fired = 1;

            // === SIMPLE HIT CHECK (ground truth) ===
            double dx_hit = S1->P[2]->x[2] - my_x;
            double dy_hit = S1->P[2]->x[3] - my_y;
            double dist_hit = sqrt(dx_hit * dx_hit + dy_hit * dy_hit);

            double ang_to_p2 = atan2(dy_hit, dx_hit);
            double err_hit = fabs(angle_diff(theta, ang_to_p2));

            int hit = (err_hit < 0.15 && dist_hit < 350.0);

            cout << "\n=== SHOT FIRED ===";
            if (hit) cout << "  RESULT: HIT";
            else     cout << "  RESULT: MISS";

            cout << "\nPress space to exit.";

            // stop simulation immediately
            double t_fire = high_resolution_time();

            // show the green beam for 1 second, or until space is pressed
            while (!KEY(' ') && high_resolution_time() - t_fire < 1.0) {
                set_inputs(1500, 1500, 1500, 1, max_speed);
                acquire_image_sim(img_rgb);
                view_rgb_image(img_rgb, 1);
                Sleep(10);
            }

            // then turn laser off and exit
            set_inputs(1500, 1500, 1500, 0, max_speed);
            break;
        }

        // === 9. HUD: heading arrow (green), line to opponent (magenta) ===
        draw_line_rgb(img_rgb, (int)my_x, (int)my_y,
            (int)(my_x + 50 * cos(theta)), (int)(my_y + 50 * sin(theta)), 0, 255, 0);
        if (found)
            draw_line_rgb(img_rgb, (int)my_x, (int)my_y,
                          (int)opp_x, (int)opp_y, 255, 0, 255);

        set_inputs(pw_l, pw_r, pw_laser, laser, max_speed);
        view_rgb_image(img_rgb, 1);
    }

    free_image(img_rgb);  free_image(img_rgb0);
    free_image(img_grey); free_image(img_bin);
    free_image(img_tmp);  free_image(img_lbl);
    free_image(img_mag);  free_image(img_theta);
    deactivate_vision();
    deactivate_simulation();
    cout << "\ndone.\n";
    return 0;
}

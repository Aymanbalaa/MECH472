
// update_simulation.cpp
// Controls dynamic changes to the simulator environment.
// Edit the sections marked below to change obstacle motion,
// background, or lighting for the optional challenge topics.

#include <cstdio>
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
#include "shared_memory.h"
#include "update_simulation.h"

extern robot_system *S1;
extern image rgb_robot, rgb_opponent, rgb_background;
extern image rgb_obstacle[N_MAX];


int update_obstacles()
// Called every frame by the simulator.
// Modify obstacle positions here for the moving-obstacle optional topic.
{
    // --- STATIC OBSTACLES (default) ---
    // Obstacles stay at their initial positions set in main().
    // No changes needed here.

    // --- OPTIONAL: uncomment below for MOVING obstacles ---
    /*
    double t = S1->t;
    S1->x_obs[0] = 320 + 75  * cos(0.25 * t);
    S1->y_obs[0] = 240 + 75  * sin(0.25 * t);
    S1->x_obs[1] = 320 + 195 * cos(0.15 * t);
    S1->y_obs[1] = 240 + 195 * sin(0.15 * t);
    */

    return 0;
}


int update_background()
// Called every frame by the simulator.
// Modify rgb_background here for changing-background optional topic.
{
    // --- DEFAULT: no background changes ---
    // The background stays as loaded from background.bmp.
    return 0;
}


int update_image(image &rgb)
// Called every frame AFTER the full scene is rendered.
// Modify pixel values in rgb here for lighting effects (optional topic).
{
    // --- DEFAULT: no image modification ---
    // The image is passed through unchanged.
    // Uncomment below for a left-to-right brightness gradient (lighting challenge).

    /*
    static int init = 0;
    static image b;
    int width  = rgb.width;
    int height = rgb.height;

    if (!init) {
        b.type   = RGB_IMAGE;
        b.width  = width;
        b.height = height;
        allocate_image(b);
        init = 1;
    }

    ibyte *p  = rgb.pdata;
    ibyte *pb = b.pdata;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            double s = 0.5 * (i / 639.0) + 0.5;  // 0.5 (left) to 1.0 (right)
            int B = (int)(*p       * s); if (B < 0) B = 0; if (B > 255) B = 255;
            int G = (int)(*(p+1)   * s); if (G < 0) G = 0; if (G > 255) G = 255;
            int R = (int)(*(p+2)   * s); if (R < 0) R = 0; if (R > 255) R = 255;
            *pb     = (ibyte)B;
            *(pb+1) = (ibyte)G;
            *(pb+2) = (ibyte)R;
            p  += 3;
            pb += 3;
        }
    }
    copy(b, rgb);
    */

    return 0;
}

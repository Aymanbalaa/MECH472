
// these functions allow update of various items in the simulator

// you can modify them to provide different challenges for
// your project, but make sure that you use the same files
// for both player1 and player2 so their simulator updates
// are the same and sychronized

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
{
	static int init = 0; // initialization flag
	int N_obs, k;
		
	// assume there are no more than 1000 obstacles
	double x_obs[1000+1], y_obs[1000+1], t;

	// get current simulation time
	t = S1->t;
	
	// get the number of obstacles
	N_obs = S1->N_obs;

	// initialization section -- this part only executes the 
	// first time the function is called
	if ( !init ) {
		
		// you don't have to use this section if you don't need it
		
		init = 1; // initialization is complete
	}

	// modify this section below to change the obstacles ////////
	// * don't modify the function above this point ///////////
	
	// set pixel location of obstacle centers
	
	// obstacle #1
	x_obs[0] = 320 + 75*cos(0.25*t);
	y_obs[0] = 240 + 75*sin(0.25*t);
	
	// obstacle #2 (if used)
	x_obs[1] = 320 + 195*cos(0.15*t);
	y_obs[1] = 240 + 195*sin(0.15*t);	
	
	// end of section you can modify //////////////////////////
	// * don't modify the function past this point ////////////
	
	// set simulation obstacle positions to x_obs, y_obs
	for(k=0;k<N_obs;k++) {
		S1->x_obs[k] = x_obs[k];
		S1->y_obs[k] = y_obs[k];
	}
		
	return 0;
}


int update_background()
// background must be an RGB image type
{	
	static int init = 0; // initialization flag
	double t;
	int width, height, i, j;
	
	// note: images used in the function should be static so
	// they are remembered between function calls.
	
	// images should also be initialized and allocated in the
	// initialization section so they only occur once.
	
	static image a; // original background image (*don't modify)
	static image b; // modified background image	
	
	ibyte *pa, *pb;	// pointers to image a and b
	ibyte R, G, B;
	
	// get current simulation time
	t = S1->t;	
	
	// get background image size
	width  = rgb_background.width;
	height = rgb_background.height;
	
	// modify this section below to change the background ////////
	// * don't modify the function above this point ///////////
	
	// initialization section -- this part only executes the 
	// first time the function is called
	if ( !init ) {
		
		// initialize image structures and allocate memory
		
		a.type = RGB_IMAGE;
		a.width = width;
		a.height = height;
		allocate_image(a);

		b.type = RGB_IMAGE;
		b.width = width;
		b.height = height;
		allocate_image(b);
		
		// keep a copy of the original background in image a
		copy(rgb_background,a);
			
		init = 1; // initialization is complete
	}
	
	// note: you shouldn't allocate new images in this function
	// outside of the initialization section above since it will
	// be too slow.  there will be a memory leak if you
	// don't free the images at the end of the function.
	// you don't need to free the images in the initialization
	// section since they are only allocated the first time
	// the function is called.
	
	// initalize image pointers
	pa = a.pdata;
	pb = b.pdata;
		
	// here we use i, j loops to access the image pixels in case 
	// you want to make an effect that depends on the (i,j) 
	// coordinates (eg lighting gradient, etc.)
		
	for(j=0;j<height;j++) {

		for(i=0;i<width;i++) {

			// get pixel RGB values for original background image
			B = *pa;
			G = *(pa+1);
			R = *(pa+2);
			
			// set pixel RGB values for updated background image
			*pb     = (ibyte)(B + i*0.17 - 77);
			*(pb+1) = (ibyte)(G + i*0.17 - 77);
			*(pb+2) = (ibyte)(R + i*0.17 - 77);

			// * note: you should check for overflow of the 
			// calculated RGB colour values to make sure they 
			// go from 0 to 255 (see update_image() below)

			// increment pointers to the next pixel
			pa += 3;
			pb += 3;

		} // end for i

	} // end for j

	// end of section you can modify //////////////////////////
	// * don't modify the function past this point ////////////
	
	// copy updated background image b into the background 
	// image for the vision simulation rgb_background
	copy(b,rgb_background);
	
	return 0;
}


int update_image(image &rgb)
// image rgb must be an RGB image type
{	
	static int init = 0;
	double t, s;
	int width, height, i, j;
	
	static image a; // original background image (*don't modify)
	static image b; // modified rgb image	
	
	ibyte *pa, *pb, *p;	// pointers to image a, b, and rgb
	
	ibyte R, G, B;
	int Ri, Gi, Bi;
	
	// get current simulation time
	t = S1->t;	
	
	// get image size
	width  = rgb.width;
	height = rgb.height;
	
	// modify this section below to change the background ////////
	// * don't modify the function above this point ///////////
	
	if ( !init ) {

		a.type = RGB_IMAGE;
		a.width = width;
		a.height = height;
		allocate_image(a);

		b.type = RGB_IMAGE;
		b.width = width;
		b.height = height;
		allocate_image(b);
		
		// keep a copy of the original background in image a
		copy(rgb_background,a);
			
		init = 1;
	}
	
	// initalize image pointers
	pa = a.pdata;
	pb = b.pdata;
	p  = rgb.pdata;
		
	for(j=0;j<height;j++) {

		for(i=0;i<width;i++) {

			// get colour for image rgb
			B = *p;
			G = *(p+1);
			R = *(p+2);

			// calculate updated pixel colours
			// -- use int Ri, Gi, Bi to protect against overflow
			
			// light gradient calculation
//			Bi = B + i*0.17 - 77;
//			Gi = G + i*0.17 - 77;
//			Ri = R + i*0.17 - 77;

			// alternative more realistic light gradient calculation

			// set s so that it goes from 0.5 to 1.0 with i
			// -- you don't want s < 0 or s > 1 since it will 
			// potentially result in R, G, B being out of range
			s = 0.5*(i/639.0) + 0.5;

			Bi = (ibyte)(B*s);
			Gi = (ibyte)(G*s);
			Ri = (ibyte)(R*s);

			// * correct for Ri, Gi, Bi out of range
			// this part is needed more for update_image()
			// than update_background() because there are 
			// many more objects in the image (robots, etc.)
			// than the background with larger colour ranges
			
			// note: this will only correct for small out of 
			// range errors, large ones will result in colour 
			// errors due to incorrect ratio of R, G, B
			
			if(Bi < 0)   Bi = 0;
			if(Bi > 255) Bi = 255;
			
			if(Gi < 0)   Gi = 0;
			if(Gi > 255) Gi = 255;
			
			if(Ri < 0)   Ri = 0;
			if(Ri > 255) Ri = 255;			

			// set output pixel to corrected Ri, Gi, Bi
			*pb     = (ibyte)Bi;
			*(pb+1) = (ibyte)Gi;
			*(pb+2) = (ibyte)Ri;		
			
			pa += 3;		
			pb += 3;
			p  += 3;
			
		} // end for i

	} // end for j
	
	// end of section you can modify //////////////////////////
	// * don't modify the function past this point ////////////
	
	// copy updated image b into vision simulation image rgb 
	copy(b,rgb);
	
	return 0;
}
